# /*****************************************************************************/
#  * File: export_cnn_weights.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2026-07-23
#  * Description: Exports a bit-accurate, pure-integer reference implementation
#  * of the quantized FalinCNN (Conv+BN-fold+ReLU+Requant+MaxPool x2, Conv+BN-
#  * fold+ReLU+Requant, GlobalAvgPool, Linear+argmax) and dumps weight/
#  * requantization ROMs for the hand-written Verilog CNN. hls4ml conversion
#  * (export_and_convert.py) was abandoned after a chain of QONNX graph
#  * incompatibilities - see Task 7 of misc/docs/fpga_implementation_plan.md.
#  *
#  * Run this BEFORE writing any Verilog: it validates that the pure-integer
#  * arithmetic model matches the QAT/PyTorch accuracy closely, exactly like
#  * gaf_golden_reference.py / gen_sin_phi_rom.py validated the GAF encoder's
#  * integer math before RTL.
#  *
#  * Math summary (see Task 7 for the full derivation):
#  *   - Every Conv has bias=False, so BatchNorm folds directly into it:
#  *     weight' = weight_int8 * bn_scale[out_ch], bias' = bn_shift[out_ch].
#  *   - Weight/activation quantizers are per-TENSOR (Int8WeightPerTensorFloat /
#  *     Int8ActPerTensorFloat), so the only per-output-channel term is BN's.
#  *   - Per conv layer: acc_int32 = conv(x_int8, W_int8) (zero-padded 3x3),
#  *     then y_int8 = clamp(round(acc_int32*M[out_ch] + Bq[out_ch]), 0, 127),
#  *     where M[out_ch] = x_scale*w_scale*bn_scale[out_ch]/next_scale and
#  *     Bq[out_ch] = bn_shift[out_ch]/next_scale. clamp(...,0,127) does ReLU
#  *     and requantization saturation in one step (Int8Act here is effectively
#  *     unsigned post-ReLU).
#  *   - MaxPool runs directly on INT8 codes (order-preserving under a positive
#  *     affine map) - no dequantization needed.
#  *   - GlobalAvgPool (sum of 24x24 INT8 codes / 576) has no learned scale of
#  *     its own, so it collapses into a SINGLE scalar multiplier (no per-
#  *     channel term) applied to the raw INT8 sum.
#  *   - Classifier bias uses Int32Bias, whose scale is DEFINED as
#  *     input_scale*weight_scale, so bias_int32 = round(raw_bias / that scale)
#  *     computed directly from the float .bias parameter (no need to touch
#  *     Brevitas's bias quantizer internals at all). Every output class shares
#  *     the same combined scale, so argmax over raw INT32 logits equals
#  *     argmax over the true dequantized logits - no final rescale needed.
#  *
#  * Multi-dataset support (added 2026-07-24): DATASET_NAME is now read from
#  * argv[1] (defaulting to ElectricDevices for backward compat). Per user
#  * request, ROM_DIR/TB_DATA_DIR now live under results/<dataset>/hw/
#  * (cnn_weights/ and tb_data/ subfolders) instead of src/hw/cnn/ - this
#  * keeps ALL of a dataset's artifacts (software checkpoints under
#  * results/<dataset>/run_1/, hardware ROMs/testbench vectors under
#  * results/<dataset>/hw/) in one place. falin_cnn_top.v's ROM_DIR parameter
#  * and falin_cnn_tb.v's `TB_DIR macro point at these same per-dataset
#  * folders (see Task 8 of misc/docs/fpga_implementation_plan.md). Also
#  * appends a row to results/hw_validation_summary.csv so the per-dataset
#  * int-vs-float accuracy deltas are collected in one place for the paper.
#  * SHIFT/NUM_TB_SAMPLES now come from config.json's "hardware" section
#  * (src/sw/utils/config_loader.py) instead of being hardcoded here.
# /*****************************************************************************/

import csv
import math
import os
import sys

import numpy as np
import torch

SW_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'sw')
sys.path.insert(0, os.path.abspath(SW_DIR))

from main import load_and_encode_ucr, UCRDatasetWrapper  # noqa: E402
from models.falin_cnn_quant import FalinCNNQuant  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from utils.config_loader import load_config  # noqa: E402

CONFIG = load_config()
HW_CFG = CONFIG['hardware']

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..')
DATASET_NAME = sys.argv[1] if len(sys.argv) > 1 else 'ElectricDevices'
UCR_ARCHIVE_PATH = os.path.join(BASE_DIR, 'misc', 'data', 'UCRArchive_2018')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'results', DATASET_NAME, 'run_1', 'FalinCNNQuant_checkpoint.pth')
DATASET_HW_DIR = os.path.join(BASE_DIR, 'results', DATASET_NAME, 'hw')
ROM_DIR = os.path.join(DATASET_HW_DIR, 'cnn_weights')
TB_DATA_DIR = os.path.join(DATASET_HW_DIR, 'tb_data')
HW_SUMMARY_CSV = os.path.join(BASE_DIR, 'results', 'hw_validation_summary.csv')
BATCH_SIZE = 64

# Fixed-point shift for all requantization multipliers (per conv layer and for
# GAP). 20 bits leaves a wide margin: worst-case accumulator magnitudes here
# top out around 2^23 (conv3: 32 in_ch * 9 taps * 127 * 127), so even a 64-bit
# intermediate product (acc * multiplier) comfortably fits signed 64-bit in
# Verilog. Chosen once globally per "One Way" - see feedback on gaf_encoder.v.
SHIFT = HW_CFG['shift']
NUM_TB_SAMPLES = HW_CFG['num_testbench_samples']


# --------------------------------------------------------------------------
# Brevitas introspection helpers
# --------------------------------------------------------------------------

def get_int8_weight(quant_module):
    """
    Returns (weight_int8 ndarray, scale float) for a QuantConv2d/QuantLinear.
    Uses only QuantTensor.value/.scale (dequantized value + scale) - the most
    stable part of Brevitas's QuantTensor API - rather than version-specific
    convenience methods like `.int()`, since this project already got burned
    by API drift across brevitas/qonnx/hls4ml versions (see Task 7 history).
    """
    qw = quant_module.quant_weight()
    scale = float(qw.scale.reshape(-1)[0].item())
    weight_int = torch.round(qw.value / scale).detach().cpu().numpy()
    return weight_int, scale


def get_activation_scales(model, calib_batch):
    """
    Captures the per-tensor activation scale of every Quant layer that sits
    at a layer boundary (quant_inp, layer{1,2,3}'s QuantReLU, classifier's
    QuantIdentity) by temporarily forcing return_quant_tensor=True and
    hooking the forward pass. Restores the original flag afterwards so the
    model's normal forward behavior (plain float tensors between layers) is
    unaffected for any other use of `model`.
    """
    quant_layers = {
        'input': model.quant_inp,
        'layer1_out': model.layer1[2],
        'layer2_out': model.layer2[2],
        'layer3_out': model.layer3[2],
        'classifier_in': model.classifier[1],
    }

    original_flags = {name: mod.return_quant_tensor for name, mod in quant_layers.items()}
    scales = {}

    def make_hook(name):
        def hook(module, inputs, output):
            scales[name] = float(output.scale.reshape(-1)[0].item())
        return hook

    handles = []
    for name, mod in quant_layers.items():
        mod.return_quant_tensor = True
        handles.append(mod.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        model(calib_batch)

    for handle in handles:
        handle.remove()
    for name, mod in quant_layers.items():
        mod.return_quant_tensor = original_flags[name]

    missing = [name for name in quant_layers if name not in scales]
    if missing:
        raise RuntimeError(
            f"Failed to capture activation scale(s) for: {missing}. "
            "This usually means return_quant_tensor wasn't actually honored by "
            "one of these modules in the installed Brevitas version - inspect "
            "the module's forward() to find the right hook point."
        )
    return scales


def quantize_to_fixed_point(multiplier, shift=SHIFT):
    """Converts a real-valued multiplier into (int_multiplier, shift) such that
    int_multiplier / 2**shift approximates `multiplier`."""
    return int(round(multiplier * (1 << shift)))


# --------------------------------------------------------------------------
# Layer parameter extraction
# --------------------------------------------------------------------------

def extract_conv_layer_params(conv, bn, in_scale, out_scale):
    """
    Returns a dict with everything the integer forward pass (and the Verilog
    ROMs) need for one Conv+BN+ReLU+Requant block:
      weight_int8:  (out_ch, in_ch, 3, 3) int8
      mult_int:     (out_ch,) int32 - fixed-point multiplier at 2**SHIFT
      bias_int:     (out_ch,) int32 - additive term at the same 2**SHIFT scale
    such that, for a raw conv accumulator `acc` (int32):
      y_int8 = clamp(round((acc*mult_int[c] + bias_int[c]) / 2**SHIFT), 0, 127)
    """
    weight_int8, w_scale = get_int8_weight(conv)

    gamma = bn.weight.detach().cpu().numpy()
    beta = bn.bias.detach().cpu().numpy()
    running_mean = bn.running_mean.detach().cpu().numpy()
    running_var = bn.running_var.detach().cpu().numpy()
    eps = bn.eps

    bn_scale = gamma / np.sqrt(running_var + eps)          # per-out_ch
    bn_shift = beta - running_mean * bn_scale               # per-out_ch

    combined_scale = in_scale * w_scale * bn_scale           # per-out_ch, real-valued
    multiplier = combined_scale / out_scale                  # per-out_ch, real-valued
    bias_scaled = bn_shift / out_scale                        # per-out_ch, real-valued (in out_scale units, pre-shift)

    mult_int = np.array([quantize_to_fixed_point(m) for m in multiplier], dtype=np.int64)
    bias_int = np.array([quantize_to_fixed_point(b) for b in bias_scaled], dtype=np.int64)

    return {
        'weight_int8': weight_int8.astype(np.int64),
        'mult_int': mult_int,
        'bias_int': bias_int,
    }


def extract_classifier_params(linear, in_scale):
    """
    Returns weight_int8 (num_classes, in_features) and bias_int32
    (num_classes,), already in the SAME integer domain as the INT8 dot
    product (no per-channel multiplier needed - see module header).
    """
    weight_int8, w_scale = get_int8_weight(linear)
    bias_scale = in_scale * w_scale
    raw_bias = linear.bias.detach().cpu().numpy()
    bias_int32 = np.round(raw_bias / bias_scale).astype(np.int64)
    return weight_int8.astype(np.int64), bias_int32


# --------------------------------------------------------------------------
# Pure-integer forward pass (the golden model for the Verilog CNN)
# --------------------------------------------------------------------------

def conv3x3_same_int(x_int8, weight_int8):
    """
    Zero-padded (pad=1) 3x3 conv, integer accumulation. x_int8: (in_ch,H,W).
    weight_int8: (out_ch,in_ch,3,3). Returns (out_ch,H,W) int64 accumulator.
    """
    in_ch, h, w = x_int8.shape
    out_ch = weight_int8.shape[0]
    x_padded = np.zeros((in_ch, h + 2, w + 2), dtype=np.int64)
    x_padded[:, 1:-1, 1:-1] = x_int8

    acc = np.zeros((out_ch, h, w), dtype=np.int64)
    for kh in range(3):
        for kw in range(3):
            # x_padded[:, kh:kh+h, kw:kw+w] has shape (in_ch, h, w)
            patch = x_padded[:, kh:kh + h, kw:kw + w]
            # weight_int8[:, :, kh, kw] has shape (out_ch, in_ch)
            acc += np.einsum('oi,ihw->ohw', weight_int8[:, :, kh, kw], patch)
    return acc


def requantize(acc, mult_int, bias_int, shift=SHIFT):
    """acc: (out_ch,H,W) int64. mult_int/bias_int: (out_ch,). Returns INT8
    (clamped [0,127]) - ReLU + requantization saturation in one step."""
    mult = mult_int.reshape(-1, 1, 1)
    bias = bias_int.reshape(-1, 1, 1)
    scaled = acc * mult + bias
    rounding_bias = 1 << (shift - 1)
    y = (scaled + rounding_bias) >> shift
    return np.clip(y, 0, 127).astype(np.int64)


def maxpool2x2_int(x_int8):
    """x_int8: (ch,H,W), H and W even. Returns (ch,H/2,W/2)."""
    ch, h, w = x_int8.shape
    reshaped = x_int8.reshape(ch, h // 2, 2, w // 2, 2)
    return reshaped.max(axis=(2, 4))


def integer_forward(gaf_image_int8, params):
    """
    Full pure-integer FalinCNN forward pass.
    gaf_image_int8: (1,96,96) int8 (the quantized GAF image - what
    gaf_encoder.v's Q1.7 output should be, see note in main()).
    Returns: predicted class index (int).
    """
    x = gaf_image_int8

    acc1 = conv3x3_same_int(x, params['layer1']['weight_int8'])
    x = requantize(acc1, params['layer1']['mult_int'], params['layer1']['bias_int'])
    x = maxpool2x2_int(x)  # (16,48,48)

    acc2 = conv3x3_same_int(x, params['layer2']['weight_int8'])
    x = requantize(acc2, params['layer2']['mult_int'], params['layer2']['bias_int'])
    x = maxpool2x2_int(x)  # (32,24,24)

    acc3 = conv3x3_same_int(x, params['layer3']['weight_int8'])
    x = requantize(acc3, params['layer3']['mult_int'], params['layer3']['bias_int'])  # (128,24,24)

    channel_sums = x.sum(axis=(1, 2)).astype(np.int64)  # (128,)
    gap_mult = params['gap']['mult_int']
    rounding_bias = 1 << (SHIFT - 1)
    gap_int8 = np.clip((channel_sums * gap_mult + rounding_bias) >> SHIFT, 0, 127)

    logits = params['classifier']['weight_int8'] @ gap_int8 + params['classifier']['bias_int32']
    return int(np.argmax(logits))


# --------------------------------------------------------------------------
# ROM file writers (plain hex text, one value per line, for $readmemh)
# --------------------------------------------------------------------------

def write_hex_rom(path, values, width_bits):
    """Writes signed integers as two's-complement hex, one per line."""
    mask = (1 << width_bits) - 1
    with open(path, 'w') as f:
        for v in np.asarray(values).flatten():
            f.write(f'{int(v) & mask:0{(width_bits + 3) // 4}x}\n')


def write_conv_layer_roms(layer_name, layer_params):
    os.makedirs(ROM_DIR, exist_ok=True)
    write_hex_rom(os.path.join(ROM_DIR, f'{layer_name}_weight.hex'), layer_params['weight_int8'], 8)
    write_hex_rom(os.path.join(ROM_DIR, f'{layer_name}_mult.hex'), layer_params['mult_int'], 32)
    write_hex_rom(os.path.join(ROM_DIR, f'{layer_name}_bias.hex'), layer_params['bias_int'], 32)


def verilog_clog2(n):
    """Mirrors classifier.v's CLASS_BITS localparam: (n<=1) ? 1 : $clog2(n)."""
    return 1 if n <= 1 else math.ceil(math.log2(n))


def write_synth_wrapper_and_script(num_classes, rom_dir_abs, hw_dir_rel):
    """
    Writes a thin, synthesis-only wrapper (falin_cnn_synth_top) that bakes in
    this dataset's NUM_CLASSES/ROM_DIR via standard Verilog instantiation-time
    parameter overrides (`falin_cnn_top #(.NUM_CLASSES(...), .ROM_DIR(...))`),
    plus a Yosys script targeting it. This sidesteps Yosys's `chparam`, whose
    support for STRING parameters (ROM_DIR) is less reliable than integer/
    width parameters - instantiation-time overrides are the same mechanism
    Verilog simulators already use, so both `iverilog` and `yosys` handle
    them identically and unambiguously.

    ROM_DIR is baked in as an ABSOLUTE path (updated 2026-07-24, was
    relative to src/hw/cnn/ before). `$readmemh` paths are resolved by each
    tool relative to ITS OWN notion of "current working directory" at
    elaboration time - `iverilog`/`yosys` both happen to use src/hw/cnn/ (by
    convention, since that's where they're invoked from), but Vivado's GUI
    project flow copies sources into its own project tree and resolves
    `$readmemh` relative to the Vivado process's cwd, not the source file's
    location - a relative path that worked for the first two tools produced
    `[Synth 8-4445] could not open $readmem data file` there. An absolute
    path is unambiguous for all three (no portability need across machines
    for this project, so the simplicity win outweighs the minor loss of
    relocatability). hw_dir_rel (used below for the Yosys script's own
    `read_verilog`/`tee` lines) stays relative, since those only ever run
    via `yosys` with cwd = src/hw/cnn/, unaffected by this.
    """
    class_bits = verilog_clog2(num_classes)
    wrapper_path = os.path.join(DATASET_HW_DIR, 'synth_top.v')
    with open(wrapper_path, 'w', encoding='utf-8') as f:
        f.write(
            '// Auto-generated by export_cnn_weights.py for '
            f'{DATASET_NAME} - do not hand-edit.\n'
            '// Thin synthesis-only wrapper: bakes in this dataset\'s NUM_CLASSES/\n'
            '// ROM_DIR via standard Verilog instantiation-time parameter overrides,\n'
            '// so falin_cnn_top.v itself stays a single, dataset-agnostic file.\n'
            '// ROM_DIR is an absolute path (portable across iverilog/yosys/Vivado -\n'
            '// see write_synth_wrapper_and_script\'s docstring).\n'
            '`timescale 1ns / 1ps\n\n'
            'module falin_cnn_synth_top (\n'
            '    input  wire clk,\n'
            '    input  wire rst_n,\n'
            '    input  wire start,\n'
            '    input  wire signed [1*96*96*8-1:0] in_flat,\n'
            '    output wire done,\n'
            f'    output wire [{class_bits - 1}:0] class_out\n'
            ');\n\n'
            '    falin_cnn_top #(\n'
            f'        .NUM_CLASSES({num_classes}),\n'
            f'        .ROM_DIR("{rom_dir_abs}")\n'
            '    ) dut (\n'
            '        .clk(clk), .rst_n(rst_n), .start(start),\n'
            '        .in_flat(in_flat), .done(done), .class_out(class_out)\n'
            '    );\n\n'
            'endmodule\n'
        )

    area_report_rel = f'{hw_dir_rel}/area_report.json'
    synth_ys_path = os.path.join(DATASET_HW_DIR, 'synth.ys')
    with open(synth_ys_path, 'w', encoding='utf-8') as f:
        f.write(
            f'# Auto-generated by export_cnn_weights.py for {DATASET_NAME} - do not hand-edit.\n'
            '# Coarse area/cell-count check (stops before techmap/abc, same rationale\n'
            '# as the original src/hw/cnn/synth.ys - see its header). Run with\n'
            '# cwd = src/hw/cnn/:\n'
            f'#   yosys {hw_dir_rel}/synth.ys\n\n'
            'read_verilog -sv conv_bn_relu_layer.v\n'
            'read_verilog -sv maxpool2x2.v\n'
            'read_verilog -sv global_avg_pool.v\n'
            'read_verilog -sv classifier.v\n'
            'read_verilog -sv falin_cnn_top.v\n'
            f'read_verilog -sv {hw_dir_rel}/synth_top.v\n'
            'hierarchy -check -top falin_cnn_synth_top\n\n'
            'proc\n'
            'opt\n'
            'memory_collect\n\n'
            f'tee -o {area_report_rel} stat -json\n'
            'stat\n'
        )
    return wrapper_path, synth_ys_path


def write_xilinx_synth_script(hw_dir_rel):
    """
    Writes results/<dataset>/hw/synth_xilinx.ys - a FULL technology-mapped
    synthesis (not the coarse pre-techmap stat.ys above) targeting real
    Xilinx 7-series primitives via Yosys's `synth_xilinx -family xc7`, since
    no Vivado is available locally (confirmed 2026-07-24). This is what
    actually produces LUT/LUTRAM/FF/BRAM/DSP counts - the coarse synth.ys
    only reports generic $mem_v2/$mul/$add cell counts, not real primitives.

    Reuses the SAME synth_top.v wrapper as synth.ys (tool-agnostic Verilog -
    no need for a second wrapper). `clkbufmap` explicitly inserts a top-level
    BUFG on the clock net, since synth_xilinx alone does not do clock-buffer
    insertion (that's normally Vivado implementation's job) - without it,
    BUFG would always read back as 0, which would be misleading rather than
    just "not applicable".

    Caveat (documented here and in the paper-facing summary): this is a real,
    open-source technology-mapped estimate, not Vivado's own synthesis engine
    - expect it to be in the right ballpark, not bit-for-bit identical to
    what `report_utilization` would say. No timing (Fmax) comes out of this
    either, since Yosys's own timing analysis isn't Vivado-grade STA.

    `-widemux 8` (added 2026-07-24, after real-world testing): conv_bn_relu_
    layer.v/maxpool2x2.v/global_avg_pool.v read/write in_flat/out_flat via
    VARIABLE-index part-selects (`in_flat[idx*W +: W]`, idx a runtime FSM
    value) - the right call for `iverilog` simulation performance (Icarus
    treats it as O(1) memory-like access), but for synthesis this lowers to
    a wide RTLIL `$shiftx` cell (no $readmemh/init pattern exists for Yosys
    to recognize it as memory, since in_flat is a live input, not a ROM).
    Confirmed via isolated single-module bisection (even the smallest
    IN_CH=1 case hangs, ruling out scale-of-IN_CH as the driver) that
    mapping a $shiftx this wide (up to ~9216-way for a 96x96 layer) hangs
    indefinitely. First guess (-noabc9, assuming abc9 was the culprit) was
    wrong - this Yosys version (0.66) doesn't default to abc9 at all (it's
    opt-in via -abc9, EXPERIMENTAL), so plain `abc` was already running.
    `-widemux 8` is the actual targeted fix: `synth_xilinx` supports mapping
    wide muxes (>=N inputs) onto dedicated Xilinx MUXF7/MUXF8 hard-mux
    resources instead of forcing plain abc's LUT-based logic synthesis to
    brute-force the huge fan-in cone. Report back whether it actually
    completes.
    """
    utilization_report_rel = f'{hw_dir_rel}/xilinx_utilization_report.json'
    synth_xilinx_ys_path = os.path.join(DATASET_HW_DIR, 'synth_xilinx.ys')
    with open(synth_xilinx_ys_path, 'w', encoding='utf-8') as f:
        f.write(
            f'# Auto-generated by export_cnn_weights.py for {DATASET_NAME} - do not hand-edit.\n'
            '# FULL technology-mapped synthesis to real Xilinx 7-series primitives\n'
            '# (LUT/FF/BRAM/DSP) - no Vivado available locally, so this is the\n'
            '# open-source stand-in (see write_xilinx_synth_script\'s docstring for\n'
            '# caveats). Expect this to take noticeably longer than synth.ys\'s\n'
            '# coarse pre-techmap check. Uses -widemux 8 to map this design\'s wide\n'
            '# variable-index part-selects ($shiftx cells) onto MUXF7/MUXF8 hard\n'
            '# resources instead of hanging plain abc\'s LUT-based mapping - see\n'
            '# docstring for the (wrong) -noabc9 attempt that preceded this.\n'
            '# Run with cwd = src/hw/cnn/:\n'
            f'#   yosys {hw_dir_rel}/synth_xilinx.ys\n\n'
            'read_verilog -sv conv_bn_relu_layer.v\n'
            'read_verilog -sv maxpool2x2.v\n'
            'read_verilog -sv global_avg_pool.v\n'
            'read_verilog -sv classifier.v\n'
            'read_verilog -sv falin_cnn_top.v\n'
            f'read_verilog -sv {hw_dir_rel}/synth_top.v\n\n'
            'synth_xilinx -top falin_cnn_synth_top -family xc7 -widemux 8\n'
            'clkbufmap -buf BUFG O I\n\n'
            f'tee -o {utilization_report_rel} stat -json\n'
            'stat\n'
        )
    return synth_xilinx_ys_path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_test_loader(img_strategy, batch_size=BATCH_SIZE):
    """Mirrors qat_finetune.py's data loading exactly, so this script sees the
    same test set (same LabelEncoder ordering, same GAF encoding) as QAT did."""
    test_path = os.path.join(UCR_ARCHIVE_PATH, DATASET_NAME, f'{DATASET_NAME}_TEST.tsv')
    train_path = os.path.join(UCR_ARCHIVE_PATH, DATASET_NAME, f'{DATASET_NAME}_TRAIN.tsv')
    _, _, X_test, y_test, num_classes, _ = load_and_encode_ucr(train_path, test_path)
    test_loader = DataLoader(UCRDatasetWrapper(X_test, y_test, img_strategy),
                              batch_size=batch_size, shuffle=False)
    return test_loader, num_classes


def main():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    weight_bit_width = checkpoint.get('weight_bit_width', 8)
    act_bit_width = checkpoint.get('act_bit_width', 8)
    print(f"[*] Loaded quantized checkpoint: descriptor_size={checkpoint['descriptor_size']}, "
          f"num_classes={checkpoint['num_classes']}, weight_bit_width={weight_bit_width}, "
          f"act_bit_width={act_bit_width}")

    model = FalinCNNQuant(
        num_classes=checkpoint['num_classes'],
        in_channels=1,
        descriptor_size=checkpoint['descriptor_size'],
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    img_strategy = checkpoint.get('img_strategy', 'gaf')
    test_loader, _ = build_test_loader(img_strategy)
    calib_images, _ = next(iter(test_loader))

    print("[*] Capturing per-layer activation scales (return_quant_tensor probe)...")
    scales = get_activation_scales(model, calib_images)
    for name, s in scales.items():
        print(f"    {name}: scale={s:.8f}")

    q1_7_scale = 1.0 / 128.0
    print(f"[*] gaf_encoder.v's Q1.7 output implies scale={q1_7_scale:.8f} "
          f"(quant_inp learned scale={scales['input']:.8f}, "
          f"ratio={scales['input'] / q1_7_scale:.4f}) - "
          f"if ratio != 1.0, the GAF->CNN interface needs an extra rescale.")

    print("[*] Extracting INT8 weights + folding BatchNorm...")
    params = {
        'layer1': extract_conv_layer_params(model.layer1[0], model.layer1[1], scales['input'], scales['layer1_out']),
        'layer2': extract_conv_layer_params(model.layer2[0], model.layer2[1], scales['layer1_out'], scales['layer2_out']),
        'layer3': extract_conv_layer_params(model.layer3[0], model.layer3[1], scales['layer2_out'], scales['layer3_out']),
    }

    gap_multiplier = scales['layer3_out'] / (576.0 * scales['classifier_in'])
    params['gap'] = {'mult_int': quantize_to_fixed_point(gap_multiplier)}

    weight_int8_fc, bias_int32_fc = extract_classifier_params(model.classifier[2], scales['classifier_in'])
    params['classifier'] = {'weight_int8': weight_int8_fc, 'bias_int32': bias_int32_fc}

    print("[*] Running pure-integer forward pass over the full test set...")
    num_correct_int = 0
    num_correct_float = 0
    num_total = 0
    tb_samples = []

    with torch.no_grad():
        for images, labels in test_loader:
            float_logits = model(images)
            float_preds = torch.argmax(float_logits, dim=1).numpy()

            images_np = images.numpy()
            for i in range(images_np.shape[0]):
                gaf_image_int8 = np.round(images_np[i] / scales['input']).astype(np.int64)
                gaf_image_int8 = np.clip(gaf_image_int8, -128, 127)

                pred_int = integer_forward(gaf_image_int8, params)
                pred_float = int(float_preds[i])
                label = int(labels[i].item())

                num_correct_int += int(pred_int == label)
                num_correct_float += int(pred_float == label)
                num_total += 1

                if len(tb_samples) < NUM_TB_SAMPLES:
                    tb_samples.append((gaf_image_int8.copy(), pred_int))

    int_accuracy = 100.0 * num_correct_int / num_total
    float_accuracy = 100.0 * num_correct_float / num_total

    print(f"[*] PyTorch (FalinCNNQuant) accuracy on {DATASET_NAME} test set: {float_accuracy:.2f}%")
    print(f"[*] Pure-integer (hardware-equivalent) accuracy: {int_accuracy:.2f}%")
    print(f"[*] Accuracy delta (integer - float): {int_accuracy - float_accuracy:+.2f} points")
    if abs(int_accuracy - float_accuracy) > 5.0:
        print("[!] WARNING: delta exceeds 5 points - SHIFT may be too coarse, or a bug in the "
              "requantization math. Do not proceed to Verilog until this is understood.")

    print(f"[*] Writing weight/multiplier/bias ROMs to: {ROM_DIR}")
    for layer_name in ('layer1', 'layer2', 'layer3'):
        write_conv_layer_roms(layer_name, params[layer_name])
    write_hex_rom(os.path.join(ROM_DIR, 'gap_mult.hex'), [params['gap']['mult_int']], 32)
    write_hex_rom(os.path.join(ROM_DIR, 'classifier_weight.hex'), params['classifier']['weight_int8'], 8)
    write_hex_rom(os.path.join(ROM_DIR, 'classifier_bias.hex'), params['classifier']['bias_int32'], 32)

    print(f"[*] Writing {NUM_TB_SAMPLES} testbench stimulus/golden vectors to: {TB_DATA_DIR}")
    os.makedirs(TB_DATA_DIR, exist_ok=True)
    for i, (gaf_image_int8, pred_int) in enumerate(tb_samples):
        np.savetxt(os.path.join(TB_DATA_DIR, f'cnn_stimulus_{i:02d}.txt'),
                   gaf_image_int8.reshape(96, 96), fmt='%d')
        with open(os.path.join(TB_DATA_DIR, f'cnn_golden_{i:02d}.txt'), 'w') as f:
            f.write(f'{pred_int}\n')

    print(f"[*] Appending this dataset's result to: {HW_SUMMARY_CSV}")
    os.makedirs(os.path.dirname(HW_SUMMARY_CSV), exist_ok=True)
    write_header = not os.path.exists(HW_SUMMARY_CSV)
    with open(HW_SUMMARY_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['Dataset', 'NumClasses', 'FloatAccuracy', 'IntAccuracy', 'DeltaPoints'])
        writer.writerow([
            DATASET_NAME, checkpoint['num_classes'],
            f'{float_accuracy:.2f}', f'{int_accuracy:.2f}', f'{int_accuracy - float_accuracy:+.2f}',
        ])

    # Paths in the -D overrides are relative to src/hw/cnn/ (where the RTL
    # lives and where iverilog/vvp are expected to be invoked from), even
    # though the ROMs/testbench vectors physically live under results/ now.
    rtl_dir = os.path.dirname(__file__)
    rom_dir_rel = os.path.relpath(ROM_DIR, start=rtl_dir)
    tb_dir_rel = os.path.relpath(TB_DATA_DIR, start=rtl_dir)

    iverilog_cmd = (
        f'iverilog -g2012 -DNUM_CLASSES={checkpoint["num_classes"]} '
        f'-DROM_DIR=\\"{rom_dir_rel}\\" -DTB_DIR=\\"{tb_dir_rel}\\" '
        f'-o sim_{DATASET_NAME} conv_bn_relu_layer.v maxpool2x2.v global_avg_pool.v '
        f'classifier.v falin_cnn_top.v falin_cnn_tb.v'
    )
    vvp_cmd = f'vvp sim_{DATASET_NAME}'

    # Also written as a standalone script (results/<dataset>/hw/run_sim.sh) so
    # run.sh (repo root) can drive all 10 datasets without re-deriving
    # NUM_CLASSES/paths itself - this stays the single source of truth for
    # that logic. Must be run with cwd = src/hw/cnn/ (same assumption as the
    # printed commands below), which run.sh takes care of.
    run_sim_path = os.path.join(DATASET_HW_DIR, 'run_sim.sh')
    with open(run_sim_path, 'w', encoding='utf-8') as f:
        f.write('#!/usr/bin/env bash\n')
        f.write(f'# Auto-generated by export_cnn_weights.py for {DATASET_NAME} - do not hand-edit.\n')
        f.write('# Run with cwd = src/hw/cnn/ (this is where the RTL sources live).\n')
        f.write('set -euo pipefail\n')
        f.write(iverilog_cmd + '\n')
        f.write(vvp_cmd + '\n')
    os.chmod(run_sim_path, 0o755)

    print("[*] Done. Do NOT run the Verilog testbench for this dataset until int_accuracy above is close to float_accuracy.")
    print(f"[*] Wrote build/run script: {run_sim_path}")
    print(f"[*] Build/run for this dataset (from src/hw/cnn/):")
    print(f'    {iverilog_cmd}')
    print(f'    {vvp_cmd}')

    hw_dir_rel = os.path.relpath(DATASET_HW_DIR, start=rtl_dir)
    rom_dir_abs = os.path.abspath(ROM_DIR)
    wrapper_path, synth_ys_path = write_synth_wrapper_and_script(
        checkpoint['num_classes'], rom_dir_abs, hw_dir_rel
    )
    print(f"[*] Wrote Yosys synth wrapper: {wrapper_path}")
    print(f"[*] Wrote Yosys synth script: {synth_ys_path}")
    print(f"[*] Coarse area check for this dataset (from src/hw/cnn/):")
    print(f'    yosys {hw_dir_rel}/synth.ys')
    print(f"[*] (Or just run ./run_synth.sh from the repo root once export_cnn_weights.py "
          f"has been run for the datasets you care about.)")

    synth_xilinx_ys_path = write_xilinx_synth_script(hw_dir_rel)
    print(f"[*] Wrote real Xilinx-primitive synth script: {synth_xilinx_ys_path}")
    print(f"[*] Real LUT/LUTRAM/FF/BRAM/DSP/BUFG estimate for this dataset (from src/hw/cnn/, "
          f"slower than the coarse check above):")
    print(f'    yosys {hw_dir_rel}/synth_xilinx.ys')
    print(f"[*] (Or just run ./run_synth_xilinx.sh from the repo root.)")


if __name__ == '__main__':
    main()
