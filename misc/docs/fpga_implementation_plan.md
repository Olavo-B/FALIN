# FALIN — FPGA Implementation Plan

Roadmap to port the FALIN pipeline (signal → GAF → FalinCNN) to an FPGA. XGBoost/TreeLUT classification stage is deferred to a later phase.

## Locked decisions

- **Deadline: SBESC 2026, August 2, 2026** (postponed from the original July 10 date). As of 2026-07-24 that's ~9 days. This drives every scope call below.
- **Board:** PYNQ-Z2 (Zynq-7020, part `xc7z020clg400-1`) — no physical board yet, so this is the simulation/synthesis target until hardware arrives.
- **Flow:** hand-written Verilog for both the GAF encoder and the CNN (see Task 7 note) — no Xilinx/Vitis toolchain available, verified with Icarus Verilog, coarse area estimate via Yosys.
- **Image strategy:** GAF, not STFT. STFT needs an FFT core (e.g. Xilinx `xfft`) plus complex-magnitude logic — far more hardware than GAF requires.
- **Paper scope (revised 2026-07-24, per user):** (1) intro, (2) software-level case for GAF+CNN vs. LSTM-FCN, (3) **CNN-only** FPGA implementation. GAF encoding and the GAF→CNN interface stay a software/conceptual step for this paper — no resize/variable-sequence-length hardware is needed, since only the CNN (fixed 96×96 input for every dataset, per Table I) goes to FPGA. The GAF encoder RTL from Task 6 remains built and verified, but isn't the paper's hardware claim this cycle; it's a natural extension for later work.
- **Target datasets: ALL 10 datasets from Table II** (ElectricDevices, StarLightCurves, Wafer, ECG5000, TwoPatterns, FordA, FordB, Yoga, Mallat, FacesUCR), not just ElectricDevices. FALIN's architecture (Table I) is IDENTICAL across all of them — every dataset's transform is resized to 96×96 in software, and only `NUM_CLASSES` + the trained weight ROMs vary. This means the same RTL (already done, Task 7) revalidates for free per dataset; the real remaining work is per-dataset train→QAT→export→simulate, not new hardware design (see Task 8).
- **Scope:** FalinCNN only. **XGBoost is deferred to a future/extended paper** (per explicit user decision 2026-07-24) — it never had a hardware implementation anyway (always a CPU-side sklearn/xgboost classifier on frozen CNN features), so dropping it from this paper's narrative doesn't cost any FPGA work, just simplifies Section III/IV's software framing and figures.
- **LSTM-FCN comparison — decision reversed (2026-07-24):** earlier in this project we concluded the paper's LSTM-FCN numbers (`misc/results_LSTM.csv`, `main.py`'s "Intersection of the 10 biggest datasets between LSTM-FCN paper and UCR Archive" comment) were a "borrowed, non-apples-to-apples" baseline requiring local retraining + an FPGA port (this was the old Task 9). On reflection, given the datasets are the **standard, fixed UCR Archive train/test splits** used identically by the original LSTM-FCN paper (Karim et al. 2018, properly cited as ref [2]), citing their published numbers on these same splits is legitimate, common practice for TSC benchmarking — not something that needs re-implementation. **Task 9 (local LSTM-FCN retrain + FPGA port) is dropped.** This frees essentially all remaining time for Task 8.
- **Folder layout:** `src/` split into `src/sw/` (Python training/quantization pipeline) and `src/hw/` (HLS/RTL work, one subfolder per IP block: `src/hw/gaf_encoder/` for Task 6, `src/hw/cnn/` for Task 7, later joined by the Vivado integration). `src/sw/` itself is further split into `src/sw/models/` (network definitions: `falin_cnn.py`, `falin_cnn_quant.py`, `falin_xgb.py`, the MobileNet/EfficientNet baselines) and `src/sw/aux/` (`gaf_golden_reference.py`, `plot_results.py`), with `main.py`, `trainer.py`, `qat_finetune.py`, `temporal2img.py` staying directly in `src/sw/`. `src/sw/` is the sys.path root everything assumes (matches `test/context.py`): files inside it import each other as bare siblings (`from main import ...`), files inside `models/`/`aux/` are imported as `models.xxx` from anywhere on that root, and `aux/` scripts add `..` to `sys.path` explicitly since they sit one level deeper.

---

## Task 1 — Add checkpoint saving to `trainer.py`

**Why:** `trainer.py` only keeps the best weights as an in-memory `copy.deepcopy(model.state_dict())` (around line 63/127), reloads them into the model (line 161), but never calls `torch.save()`. No `.pth` exists anywhere under `results/`.

**What to do:**
- Right after the best weights are reloaded into the model, add a `torch.save()` call.
- Save to something like `results/<dataset>/run_<n>/checkpoint_<model_name>.pth`.
- Save metadata alongside the weights, not just the raw `state_dict`:
  ```python
  torch.save({
      'state_dict': model.state_dict(),
      'dataset': dataset_name,
      'img_strategy': img_strategy,
      'descriptor_size': descriptor_size,
      'seq_length': seq_length,
      'num_classes': num_classes,
      'label_classes': list(orig_classes),
  }, checkpoint_path)
  ```
- This requires threading `ds_results_dir` (or an explicit save path) into `run_experiment` in `trainer.py`, since it's currently only known in `main.py`.

---

## Task 2 — Retrain FalinCNN on ElectricDevices + GAF, save golden checkpoint

- In `main.py`, set `DATASET = ['ElectricDevices']`, `IMG_STRATEGY = 'gaf'`, `MODELS_TO_RUN = ['FalinCNN']` (skip the other variants to save time).
- Run `python src/sw/main.py`.
- Confirm final test accuracy lands near the ~71–73% already recorded in `results/gafs/summary_metrics.csv` — this is a reproducibility sanity check, not a new result.
- With Task 1 done, this run produces `checkpoint_FalinCNN.pth`.
- Additionally dump a fixed batch of GAF-encoded test images to `.npy` (e.g. first N test samples, using `Temporal2ImageDataset` directly and `np.save` on `self.images`). These become the **golden vectors** used later to verify the HLS testbench.

---

## Task 3 — Implement the GAF encoder as a standalone hardware-oriented golden model

`_gramian_encoding` in `temporal2img.py` currently computes `phi = arccos(x)` then `cos(phi)`, `sin(phi)` — redundant trig, since:

- `cos(arccos(x)) = x`
- `sin(arccos(x)) = sqrt(1 - x²)`

Rewrite without trig:

```python
def gaf_encode_hw(x, min_val, max_val):
    if max_val == min_val:
        x_norm = np.zeros_like(x)
    else:
        x_norm = ((x - min_val) / (max_val - min_val)) * 2 - 1
    x_norm = np.clip(x_norm, -1.0, 1.0)

    cos_phi = x_norm
    sin_phi = np.sqrt(1.0 - x_norm ** 2)

    return np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)
```

- Verify numerically that this matches the original `_gramian_encoding` output on ElectricDevices data elementwise (should match to float precision — write a quick assert comparing both).
- No resize step needed (`seq_len == 96 == target resolution`).
- This becomes the **golden model**: ground truth for bit-exact (or tolerance-bound) verification of the HLS C++ implementation in Task 6.

---

## Task 4 — BN-fold FalinCNN (fuse `BatchNorm2d` into the preceding `Conv2d`)

**Superseded — folded into Task 7's `export_cnn_weights.py` instead of being a standalone float-domain pre-step.** Same identity (`y = gamma*(x-running_mean)/sqrt(running_var+eps)+beta` folds into `weight' = weight*scale`, `bias' = beta - running_mean*scale`), but applied directly against the **quantized** conv weights/scales at export time, combined in the same per-output-channel fixed-point multiplier used for INT8 requantization (see Task 7). Doing it there avoids a redundant float BN-fold pass that would just get re-derived once quantization is added anyway.

---

## Task 5 — Quantize FalinCNN to INT8 and validate accuracy

- Recommended path: **Brevitas** (QAT) + QONNX export — this is the path hls4ml natively supports, more reliable than PyTorch's built-in PTQ for this conversion chain.
- Replace `nn.Conv2d` / `nn.Linear` with `brevitas.nn.QuantConv2d` / `QuantLinear`, `weight_bit_width=8`, `act_bit_width=8` (try lower if accuracy holds).
- Fine-tune (QAT) starting from the BN-folded float checkpoint (Task 4) for a handful of epochs.
- Compare accuracy against the float32 baseline (~71–73%). Flag if the drop exceeds ~5 points — that would mean either a lower bit-width fallback or per-channel quantization needs investigating.
- Export the final quantized model to ONNX/QONNX.

---

## Task 6 — Write the GAF encoder in Verilog + testbench (Yosys, not Vitis HLS)

Superseded the original Vitis HLS C++ plan: no Xilinx toolchain available, so the encoder is hand-written synthesizable Verilog, verified with Icarus Verilog and synthesized (area/LUT estimate only, not full P&R/bitstream — that still goes through Vivado later) with Yosys. Files live under `src/hw/gaf_encoder/`.

- `gen_sin_phi_rom.py`: generates `sin_phi_rom.hex`, the ROM `sin_phi = sqrt(1-x_norm²)` lookup (no sqrt/trig hardware needed).
- `gaf_encoder.v`: 3-stage FSM (MINMAX → NORMALIZE → OUTER_PROD, one sample/row per cycle), 192 parallel 12×12 multipliers in the outer-product stage.
- `gaf_encoder_tb.v`: Icarus Verilog testbench, reads `tb_data/` vectors from `src/sw/aux/gaf_golden_reference.py`.

**Key finding (read `gen_sin_phi_rom.py`'s header for the full analysis):** `sqrt(1-x²)` has a vertical tangent at `x = ±1`, and min-max normalization guarantees every 96-sample window has a sample landing exactly there (its own min and max). An 8-bit ROM address (matching the 8-bit GAF output) gives worst-case per-element error ~0.13 — far worse than the ~1/256 the output width suggests, because error near the singularity shrinks only as `sqrt(step size)`, not linearly. Measured on real ElectricDevices windows:

| ROM address bits | worst-case max error | mean error |
|---|---|---|
| 8  | 0.129 | 0.007 |
| 10 | 0.065 | 0.007 |
| 12 | 0.035 | 0.0065 |
| 14 | 0.019 | 0.0065 |

Chose **12-bit ROM address** (4096 × 12-bit, 6 KB — one small BRAM). Internal precision (`cos_phi`/`sin_phi`, Q1.11) is kept higher than the final 8-bit GAF output; rounding to Q1.7 only happens once, after the outer product — the standard "accumulate wide, round once at the end" rule for fixed-point hardware. Once `gaf_encoder.v`'s exact integer arithmetic (round-half-up bias, arithmetic shift, saturate) was modeled bit-for-bit in Python and checked exhaustively against all 7711 real ElectricDevices test windows, the true worst-case error came out at **0.0114** (better than the 0.035 float-domain estimate above — trust the exhaustive number).

Run (from `src/hw/gaf_encoder/`, after `python ../../sw/aux/gaf_golden_reference.py`):
```
iverilog -g2012 -o sim gaf_encoder.v gaf_encoder_tb.v
vvp sim
yosys synth.ys
```
`synth.ys` deliberately stops before `techmap`/`abc` — 192 parallel 12×12 multipliers plus a 4096-entry ROM is a lot for ABC's gate-level optimization to chew through, and full mapping isn't needed for a coarse area check anyway. Stopping right after `memory_collect` and running `stat` reports raw cell counts (`$mul`, `$mem`, etc.) without expanding them. Full technology-mapped synthesis is Vivado's job later (Task 8), not Yosys's here.

**First simulation run found two real bugs, now fixed:** (1) module ports used unpacked SystemVerilog array syntax (`input ... foo [0:N-1]`), which Yosys's Verilog frontend rejects — flattened to 1D vectors (`signal_in_flat`, `gaf_out_flat`), unpacked/packed internally right at the port boundary. (2) the final rounding step sliced the wrong bit range (`rounded[21:14]` instead of `rounded[22:15]`, a 1-bit shift error) and used a fragile manual bit-disagreement overflow check instead of a plain signed comparison — both replaced with an explicit `SHIFT_AMOUNT`/`ROUND_BIAS`/`OUT_MAX`/`OUT_MIN` computation. Re-run the commands above and report the new results.

---

## Task 7 — Hand-written Verilog CNN (superseded the hls4ml conversion)

**Pivoted, same reasoning as Task 6:** `export_and_convert.py`'s hls4ml route got stuck in a deep chain of QONNX/ONNX-graph incompatibilities (missing `kernel_shape`, `ReduceMean`→`GlobalAveragePool`, `Gemm`→`MatMul`+`Add`, and finally an unquantized classifier bias breaking hls4ml's internal scale propagation on the last `Add` node — traced all the way into `move_scales.py`/`infer_precision.py`). Rather than keep patching an ONNX graph produced by a moving-target exporter, the CNN is now hand-written Verilog, mirroring Task 6's GAF encoder. `export_and_convert.py`/hls4ml stays in the repo for reference but is no longer the active path.

**Math (validated in Python before any RTL, same discipline as Task 6):**

FalinCNN is `(Conv3x3→BN→ReLU→Quant→MaxPool2)×2 → Conv3x3→BN→ReLU→Quant → GlobalAvgPool → Linear(128,7)`. Every Conv has `bias=False`, so BN folds cleanly into it (Task 4's identity, applied in the quantized domain): `weight' = weight_int8 * bn_scale[out_ch]`, `bias' = bn_shift[out_ch]`. Since `Int8WeightPerTensorFloat`/`Int8ActPerTensorFloat` are per-tensor (single scalar scale, not per-channel), the only per-output-channel term is BN's.

Per conv layer, the integer pipeline (mirrors GAF's "accumulate wide, round once" rule):
```
acc_int32[out_ch]   = sum over (in_ch, kh, kw) of x_int8 * W_int8      # zero-padded 3x3 conv
combined_scale       = x_scale * w_scale(per-tensor) * bn_scale[out_ch]
M[out_ch]            = combined_scale / next_layer_activation_scale    # dimensionless
y_int8               = clamp(round(acc_int32*M[out_ch] + bn_shift[out_ch]/next_scale), 0, 127)
```
`M[out_ch]` and the folded bias are pre-computed once in Python and baked into fixed-point `(multiplier_int, SHIFT)` ROMs — same technique as `gaf_encoder.v`'s `ROUND_BIAS`/`SHIFT_AMOUNT`. `clamp(...,0,127)` does ReLU (negatives clamp to 0) and requantization saturation in one step. MaxPool runs directly on the INT8 codes post-requantization — max is scale-invariant under a positive affine map, so no dequantization is needed, just comparators.

GlobalAvgPool (sum of 24×24 INT8 codes ÷ 576, then requantized into the classifier's input scale) collapses into a **single** scalar multiplier (no per-channel term, since GAP has no learned scale/bias) applied directly to the raw INT8 sum.

The classifier (`Linear(128,7)`) uses `Int32Bias`, whose bias scale is *defined* as `input_scale * weight_scale` — so `bias_int32 = round(raw_float_bias / (input_scale*weight_scale))`, added directly to the INT8 dot-product accumulator. Because every output class shares the same combined scale, **argmax over the raw INT32 logits equals argmax over the true dequantized logits** — no final rescale needed, the hardware classifier just needs a 7-way comparator tree.

**Files (under `src/hw/cnn/`):**
- `export_cnn_weights.py`: loads `FalinCNNQuant_checkpoint.pth`, extracts INT8 weights (`module.quant_weight().value / .scale`, robust across Brevitas versions since it only touches the stable `QuantTensor.value`/`.scale` attributes) and activation scales (temporarily flips `return_quant_tensor=True` + forward hooks on `quant_inp`/`layer{1,2,3}[2]`/`classifier[1]`, restored after), folds BN, computes all `(multiplier_int, SHIFT)` ROMs, then **runs the full fixed-point integer pipeline above over the entire real ElectricDevices test set** and reports integer-vs-float accuracy — the gate to pass before writing any Verilog, same as `gen_sin_phi_rom.py`/`gaf_golden_reference.py`'s exhaustive checks. Also writes weight/bias/multiplier `.hex` ROMs (`cnn_weights/`) and testbench stimulus/golden vectors (`tb_data/`).
  - **Result (2026-07-23): 71.09% integer vs 71.09% float — 0.00 point delta.** Also confirmed `quant_inp`'s learned activation scale is exactly `1/128`, matching `gaf_encoder.v`'s Q1.7 output convention bit-for-bit — the GAF encoder's raw output plugs directly into the CNN with **no rescale needed at that interface**.
- `conv_bn_relu_layer.v`: generic (IN_CH/OUT_CH/IMG_SIZE-parameterized) Conv+BN-fold+ReLU+Requant. **Functionally verified (2026-07-23): all 5 testbench samples match the golden model's argmax class exactly.** Initial version did one (in_ch,kh,kw) MAC/cycle (IN_CH*9 cycles/output position) — measured ~33.9M cycles/sample, with conv2 (IN_CH=16) and conv3 (IN_CH=32) alone at 95% of that. Reworked to parallelize across IN_CH (the dominant, layer-varying cost axis — kh*kw is a fixed, already-small 9): for each (kh,kw) tap, all IN_CH channels are multiplied and summed in one cycle via a combinational ripple-add chain, cutting the inner loop to 9 cycles/output position regardless of IN_CH. Same idea as the reference paper's (`misc/docs/reference/1-s2.0-S2542660525002197-main.pdf`) K-parallel-multiplier-plus-tree-adder Kernel module, applied to our dominant axis (IN_CH) instead of theirs (kernel width). New estimate: ~3.1M cycles/sample (~11x), dominated now by conv1 (IN_CH=1, unaffected by this change) and the still-sequential conv2/conv3 out_ch×row×col×9 loop. One instance per conv block (layer1/2/3), only the parameters and ROM filenames differ.
  - **Compile/synth-time lesson (3 rounds):** (1) an unconditional clocked full-array copy every cycle made `vvp` hang 30+ min (simulation-time cost); (2) a `wire in_buf[]` array read via a runtime-computed index made `iverilog` itself slow even after fixing (1) (dynamic index into an array-of-nets forces an N-way decoder, unlike a true `reg` array/memory which is O(1) regardless of size) — fixed by reading input directly off the flat port via a variable-base indexed part-select instead of staging it in an array at all; (3) the procedural once-only pack loop that fixed (1) turned out to make `yosys`'s `proc` pass take 3+ hours, since synthesis must *statically unroll* every procedural for-loop into concrete hardware (fine for a simulator, which just interprets it, very much not fine for synthesis at 147456 iterations) — fixed by dropping the staging buffer entirely and writing each result straight into the output port at its computed bit-offset (same indexed-part-select trick, now on a write). Net effect: no module has an array indexed by a runtime variable, and no module has a procedural loop over more than a handful of iterations. The IN_CH-parallel MAC's `generate` block is unaffected by any of this — IN_CH is small (≤32) and only ever indexed by generate-constant indices, which elaborate as independent point-to-point signals in both tools.
- `maxpool2x2.v`: generic 2x2/stride-2 max-pool directly on INT8 codes (order-preserving, no dequant needed). Separate module from the conv layer on purpose (layer3 has no pooling after it) — Separation of Concerns.
- `global_avg_pool.v`: sum-then-single-scalar-multiply (GAP has no learned per-channel scale, unlike the conv layers' BN).
- `classifier.v`: `Linear(128,7)` + running-max/argmax — no final rescale needed (see math above), just a 7-way comparator.
- `falin_cnn_top.v`: wires the above into the full pipeline with a top-level FSM sequencing each stage's start/done handshake.
- `falin_cnn_tb.v` + `synth.ys`: same Icarus Verilog + Yosys coarse-area-check flow as Task 6. **Expect real wall-clock minutes per simulated sample** — this is a one-MAC-per-cycle sequential accelerator (conv2/conv3 alone are ~10-20M cycles each), not a bug; parallelizing (e.g. across `in_ch`, like the GAF encoder's 96-wide outer product) is a follow-up optimization once correctness is confirmed.

Build & run (from `src/hw/cnn/`, after `export_cnn_weights.py` has generated `cnn_weights/` and `tb_data/`):
```
iverilog -g2012 -o sim conv_bn_relu_layer.v maxpool2x2.v global_avg_pool.v classifier.v falin_cnn_top.v falin_cnn_tb.v
vvp sim
yosys synth.ys
```
**Status (2026-07-24): done.** All 5 testbench samples match the golden model's argmax class exactly. Reworked once for latency (parallelized the MAC across `IN_CH` instead of one tap/cycle — cut cycles/sample from ~33.9M to ~3.37M, a ~10x measured speedup; conv1 is now the largest remaining cost since `IN_CH=1` there leaves nothing to parallelize). Reworked twice more for tool-scaling issues unrelated to correctness — see `conv_bn_relu_layer.v`'s header for the full 3-round history (simulation-time full-array copy, `iverilog` elaboration-time dynamic array indexing, `yosys` `proc`-pass procedural-loop unrolling) — current design has no array indexed by a runtime variable and no procedural loop beyond a handful of iterations anywhere. Yosys coarse area check passed: 1329 cells, 12 `$mem_v2` (the expected ROMs, correctly preserved as memories), ~54 real datapath multipliers by design (not the raw 219 `$mul` count, which is inflated by not-yet-optimized address-index arithmetic since `synth.ys` deliberately stops before `techmap`/`abc`).

---

## Task 8 — CNN-only FPGA validation across all 10 datasets + real Vivado synthesis numbers

**Revised scope (2026-07-24), given the paper now covers CNN-only hardware (GAF encoder stays software-side for this paper) and the Aug 2 deadline.** No full Vivado block design / bitstream / board deployment this cycle — that's future work. The paper's claim for this cycle is: real, hand-written Verilog, functionally verified against the golden integer model across all 10 datasets, with real (not just estimated) resource/timing numbers from Vivado synthesis for at least one configuration.

**Does the 96x96 resize need FPGA hardware? No — confirmed 2026-07-24.** `temporal2img.py`'s `Temporal2ImageDataset` computes the GAF matrix at its *native* resolution (`seq_len x seq_len` - e.g. 500x500 for FordA, 1024x1024 for Mallat/StarLightCurves) and only afterwards bilinearly interpolates it down/up to the fixed 96x96 (`F.interpolate(..., mode='bilinear')`, done once per sample and cached in RAM). Since the paper's FPGA claim is now CNN-only (GAF stays software-side, see Locked decisions above), this resize is entirely absorbed by the software front-end before any INT8 image reaches hardware - the CNN's hardware contract is simply "a 96x96 INT8 image", regardless of whether it came from a native-96x96 GAF matrix (only true for ElectricDevices) or a resized one (the other 9 datasets). No interpolation hardware is needed this cycle. (Caveat for future work: `gaf_encoder.v`, Task 6, only handles the no-resize ElectricDevices case - if GAF-in-hardware is ever revisited for other datasets, it would need real bilinear-interpolation logic added, and the "no rescale needed at the GAF->CNN interface" finding (`quant_inp` scale = exactly 1/128) was a coincidence specific to ElectricDevices, not a general guarantee - other datasets will calibrate to whatever scale QAT learns, which `export_cnn_weights.py` already captures correctly per-dataset, not hardcoded.)

**Hyperparameters now centralized in `config.json` (repo root, added 2026-07-24).** Per user request, a single JSON file is the source of truth for both the software (`main.py`) and hardware (`export_cnn_weights.py`) pipelines - dataset list, epochs/batch size/adaptive LR+patience rules, QAT epochs/LR/bit-widths, and hardware SHIFT/testbench-sample-count. Loaded via `src/sw/config_loader.py`'s `load_config()`. Editing one file now retunes both the software training run and the hardware export/testbench generation - no more hunting across 3 files for a hardcoded constant.

**Hardware artifacts now live under `results/<dataset>/hw/` (moved 2026-07-24, per user request), not `src/hw/cnn/`.** `export_cnn_weights.py`'s `ROM_DIR`/`TB_DATA_DIR` now resolve to `results/<dataset>/hw/cnn_weights/` and `results/<dataset>/hw/tb_data/` - this keeps everything for a given dataset (software checkpoints under `results/<dataset>/run_1/`, hardware ROMs/testbench vectors under `results/<dataset>/hw/`) in one place instead of splitting outputs between `results/` and `src/hw/cnn/`. The script prints the exact `iverilog -D...` command with correct relative paths (computed via `os.path.relpath`, not hand-written) from `src/hw/cnn/` (where the RTL lives and where `iverilog`/`vvp` are still invoked from) to the new `results/` location.

**On a Verilog code generator (considered and rejected, 2026-07-24):** the CNN's structure is identical across every dataset - only `NUM_CLASSES` and ROM *contents* vary, which is exactly the problem Verilog `parameter`s and `` `define``s are built to solve. A separate Python-based code generator would just reimplement that substitution in a new, untested layer, adding a maintenance burden and a fresh source of tooling bugs (this project already spent 3 rounds debugging simulation/synthesis-scaling issues and abandoned hls4ml/HLS entirely for exactly this class of reason) for no structural benefit - it wouldn't let us express anything `-D`/`parameter` overrides can't. Decision: no generator; kept the single canonical RTL set + per-dataset compile-time overrides (standard practice for parameterized HDL IP, e.g. how Xilinx's own IP cores handle configurability).

**8a — Generalize the CNN RTL + export pipeline for any dataset (done 2026-07-24):**
- `export_cnn_weights.py` now takes `DATASET_NAME` from `argv[1]` (default `ElectricDevices`), writes ROMs to `results/<dataset>/hw/cnn_weights/` and testbench vectors to `results/<dataset>/hw/tb_data/`, and appends each run's int-vs-float accuracy to `results/hw_validation_summary.csv`. SHIFT/NUM_TB_SAMPLES now come from `config.json`.
- `falin_cnn_top.v` gained a `ROM_DIR` string parameter, threaded via string concatenation to all 5 submodules' `WEIGHT_FILE`/`MULT_FILE`/`BIAS_FILE` params (defaults preserved, so old invocations are unaffected). `NUM_CLASSES` was already a top-level parameter.
- `falin_cnn_tb.v` replaced its hardcoded `NUM_CLASSES`/path localparams with `` `define``-driven macros (`NUM_CLASSES`, `ROM_DIR`, `TB_DIR`), overridable via `iverilog -D` flags, so the exact same RTL/testbench files recompile per dataset with no duplication (`` `ifndef`` defaults still match the original ElectricDevices invocation).
- `qat_finetune.py` similarly takes `DATASET_NAME` from `argv[1]`; its hyperparameters now come from `config.json`'s `qat` section.
- `main.py`'s `DATASET` list and training hyperparameters now come from `config.json` (previously hardcoded; only `ElectricDevices` was uncommented before).

**8b — Run the pipeline for all 10 datasets (user-run, needs GPU/iverilog/yosys locally):**
1. `python src/sw/main.py` once — trains float `FalinCNN` + saves `FalinCNN_checkpoint.pth` for all 10 datasets (Task 1's checkpoint saving already applies).
2. `./run.sh` (repo root, added 2026-07-24) — orchestrates the rest for every dataset in `config.json` (or a subset, e.g. `./run.sh Wafer FordA`): runs `qat_finetune.py <Dataset>`, then `export_cnn_weights.py <Dataset>` (which also writes a per-dataset `results/<Dataset>/hw/run_sim.sh`, the single source of truth for that dataset's `NUM_CLASSES`/ROM paths), then runs that generated script (`iverilog` + `vvp`) from `src/hw/cnn/`. Skips a dataset with a clear warning if its float checkpoint is missing, continues past a failing dataset instead of aborting the whole run, and prints a pass/fail summary at the end. Expect ~3.37M cycles/sample regardless of dataset (same architecture).
3. Collect `results/hw_validation_summary.csv` (10 rows: dataset, num_classes, float acc, int acc, delta), appended to automatically by `export_cnn_weights.py` — this table is the paper's hardware-validation evidence, replacing the single ElectricDevices data point.

**8c — Coarse Yosys area check for all 10 datasets (added 2026-07-24, can run concurrently with 8b's `./run.sh`):** `export_cnn_weights.py` now also writes a thin synthesis-only wrapper (`results/<dataset>/hw/synth_top.v`, bakes in that dataset's `NUM_CLASSES`/`ROM_DIR` via standard Verilog instantiation-time parameter overrides - avoids relying on Yosys's `chparam` for STRING parameters, which is less reliable than for integer/width params) and a matching `results/<dataset>/hw/synth.ys`. `./run_synth.sh` (repo root) runs Yosys's coarse check (same stop-before-techmap/abc approach as the original manual `src/hw/cnn/synth.ys`) for every dataset that already has a generated `synth.ys`, skipping (not failing) datasets `run.sh` hasn't reached yet - safe to run in parallel with `run.sh` since it only reads artifacts and writes into `area_report.json`, a file `run.sh` never touches. At the end it calls `src/hw/cnn/collect_area_summary.py`, which parses every `results/*/hw/area_report.json` into one `results/hw_area_summary.csv` (dataset, num_classes, total cells, `$mem_v2` count, `$mul` count) - turning the "area should be nearly constant across datasets" expectation into an actual measured table for the paper.

**Analytical GAF+resize transform cost, no new RTL (decided 2026-07-24).** The paper needs to account for the cost of turning a raw sensor signal into the 96x96 image the CNN consumes, but re-scoping the FPGA claim to include GAF hardware for all 10 datasets was explicitly rejected (too much new RTL risk this close to the deadline) in favor of an analytical estimate. `src/sw/utils/estimate_transform_cost.py` (pure stdlib, run directly, writes `results/gaf_transform_cost_estimate.csv`) extrapolates `gaf_encoder.v`'s OWN already-built, already-simulated architecture (Task 6) - `cycles(L) = 3L + O(1)`, `multipliers(L) = 2L`, `MACs(L) = 2L^2`, all directly read off its 3-stage FSM (MINMAX/NORMALIZE/OUTER_PROD, each `L` cycles) - to every dataset's native sequence length, plus a constant ~36,864-MAC bilinear-resize cost (`temporal2img.py`'s `F.interpolate` to 96x96, independent of source length, software-only). Result: even the longest sequences (Mallat/StarLightCurves, L=1024) cost under 9.5% of the CNN's 22.6M MACs and under 0.1% of its ~3.37M cycles - the transform stage is not the bottleneck. **Caveat to carry into the paper:** this is an extrapolation of a real architecture's parametric cost, exact only for ElectricDevices (L=96, the only length actually built/simulated) and analytical for the other 9; the "cycles" figure specifically assumes gaf_encoder.v's full-parallelism (2L multipliers) scales up as-is, which would exceed the xc7z020's 220 DSP48E1 budget for large L without a time-multiplexed redesign - the MACs figures are architecture-independent and are the safer number to lead with.

**No Vivado available (confirmed 2026-07-24 - user tried `vivado` on the command line, not found).** The coarse Yosys check (8c) only reports generic `$mem_v2`/`$mul`/`$add` cell counts - it does NOT answer "how much LUT/LUTRAM/FF/BRAM/DSP/BUFG does this use", which is what the paper actually needs and what a reviewer would expect (Vivado's own `report_utilization` categories). Without Vivado, the open-source stand-in is Yosys's own full technology-mapped synthesis to real Xilinx 7-series primitives:

- `export_cnn_weights.py` now also writes `results/<dataset>/hw/synth_xilinx.ys` per dataset (reusing the same `synth_top.v` wrapper from 8c - no second wrapper needed), running `synth_xilinx -top falin_cnn_synth_top -family xc7` (full techmap+abc9 to LUT6..1/FDRE.../RAMB18E1/RAMB36E1/DSP48E1) followed by `clkbufmap -buf BUFG O I` (synth_xilinx alone doesn't insert a top-level clock buffer - that's normally Vivado implementation's job - so without this, BUFG would misleadingly read back as 0 instead of the expected 1).
- `./run_synth_xilinx.sh` (repo root, same skip-if-not-ready/parallel-safe pattern as `run_synth.sh`) runs this for every dataset, then calls `src/hw/cnn/collect_xilinx_utilization.py`, which buckets the real Yosys cell-type names into the six categories Vivado reports (LUT, LUTRAM, FF, BRAM, DSP, BUFG - plus an "Other" column for anything unrecognized, e.g. `CARRY4`/`IBUF`/`OBUF`, so nothing silently vanishes from the accounting) into `results/hw_xilinx_utilization_summary.csv`.
- **Caveat, to state honestly in the paper:** this is Yosys's own technology mapping, not Vivado's synthesis engine - expect ballpark agreement (it maps to the same real Series-7 primitives Vivado would target), not a bit-for-bit match to what `report_utilization` prints. No Fmax/timing comes out of this either (Yosys's own static timing isn't Vivado-grade STA).
- **Real bug found and fixed (2026-07-24): `abc9` hangs indefinitely on this design.** `conv_bn_relu_layer.v`/`maxpool2x2.v`/`global_avg_pool.v` read/write `in_flat`/`out_flat` via variable-index part-selects (`in_flat[idx*W +: W]`, `idx` a runtime FSM value) - correct for `iverilog` simulation performance (Icarus treats it as O(1)), but for synthesis this lowers to a wide RTLIL `$shiftx` cell (no `$readmemh`/init pattern exists for `in_flat`, since it's a live input, not a ROM - Yosys can't recognize it as memory). `abc9`'s technology mapping of a `$shiftx` this wide (~9216-way for a 96x96 layer) hangs indefinitely - confirmed via isolated single-module bisection (`src/hw/cnn/debug_layer1.ys`/`debug_layer3.ys`, disposable diagnostic files): even the smallest `IN_CH=1` case hangs, ruling out scale-of-`IN_CH` as the driver, and Ctrl+C doesn't recover the process. **Fix:** `write_xilinx_synth_script` now passes `-noabc9` to `synth_xilinx` (falls back to the older, plain `abc` mapper). Time-boxed experiment - if this also doesn't complete in a reasonable time, treat full Xilinx-primitive numbers as out of reach for this deadline and fall back to the coarse Yosys cell counts (8c) + the analytical MAC-based transform cost as the paper's hardware area evidence, which remain valid regardless.

**Update (2026-07-24): Vivado is available after all** - user found the GUI still installed (just not on the command-line `PATH`). This supersedes 8c's Yosys `synth_xilinx` detour, which was root-caused (wide `$shiftx` cells from the RTL's variable-index part-selects hang indefinitely under `abc`'s LUT mapping, confirmed via isolated single-module bisection and staged pipeline bisection) but not worth continuing to chase now that real Vivado numbers are reachable. `src/hw/cnn/vivado_synth.tcl` does an out-of-context `synth_design` (no block design, no bitstream) targeting `xc7z020clg400-1`, reusing the SAME per-dataset `synth_top.v` wrapper `export_cnn_weights.py` already generates (plain Verilog instantiation-time parameter overrides - no Yosys-specific machinery, works unchanged in Vivado). Writes `report_utilization` (flat + hierarchical) and `report_timing_summary` into `results/<dataset>/hw/`. Run from Vivado's Tcl Console with cwd = `src/hw/cnn/` (same relative-path convention as every other tool here): `source vivado_synth.tcl` (edit the `dataset_name` variable at the top for datasets other than ElectricDevices).

**Real bug found and fixed (2026-07-24): Vivado's 1,000,000-bit single-variable elaboration limit.** First GUI `synth_design` attempt (project named "rato") failed with `[Synth 8-4556] size of variable 'conv1_out' is too large to handle; the size of the variable is 1179648, the limit is 1000000` at `falin_cnn_top.v`. `conv1_out` (layer1's output, before any pooling shrinks the spatial size: `16*96*96*8 = 1,179,648` bits) is the widest inter-stage signal in the whole design and the only one over Vivado's ceiling — checked every other one (`pool1_out`=294,912, `conv2_out`=589,824, `pool2_out`=147,456, `conv3_out`=589,824, `gap_out`=1,024 bits), all under. Neither `iverilog` nor Yosys enforce this limit, so it was invisible until reaching Vivado.

- **First fix attempt (unpacked array ports) failed too:** converting `conv_bn_relu_layer.v`'s `out_flat` (and the matching `in_flat` on `maxpool2x2.v`/`global_avg_pool.v`) to a true unpacked `reg`/`wire` array port hit `[Synth 8-9917] port 'out_flat' must not be declared to be an array` — Vivado's `synth_design` rejects unpacked array ports outright, for a different reason than Yosys's older frontend (which also rejects them, in the ANSI port list specifically). Confirmed a dead end, reverted.
- **Actual fix: split `out_flat` into two packed ports, `out_flat_lo`/`out_flat_hi`, each covering `OUT_CH/2` channels.** Worst case (layer1) is `8*96*96*8 = 589,824` bits per half — safely under the 1M limit, and `OUT_CH` (16/32/128) is always even across all 3 instantiations. `ST_REQUANT` in `conv_bn_relu_layer.v` now picks lo/hi based on which half `out_ch_cnt` falls in, using a local (within-half) channel index for the part-select offset — same O(1) variable-base-part-select pattern as before, just against a half-width bus, no change to the MAC/requant datapath itself. `maxpool2x2.v`'s `in_flat_lo`/`in_flat_hi` and `global_avg_pool.v`'s `in_flat_lo`/`in_flat_hi` mirror this on the consumer side (both compute a `ch_is_hi`/`local_ch` pair and index into whichever half `ch_cnt` currently falls in). `falin_cnn_top.v`'s `conv1_out_lo/hi`, `conv2_out_lo/hi`, `conv3_out_lo/hi` wire pairs wire it all together — only `conv1_out` strictly needed the split for size reasons, `conv2_out`/`conv3_out` split too only for port-type consistency with the module definitions they're instantiated from. `maxpool2x2.v`'s own `out_flat` and `global_avg_pool.v`'s own `out_flat`, and `conv_bn_relu_layer.v`'s own `in_flat`, stay single packed vectors throughout (never exceed the limit, no reason to touch them). The disposable `debug_layer1_top.v`/`debug_layer3_top.v` Yosys-hang diagnostic files (already obsolete now that Vivado is the synthesis target, and broken by the port changes either way) were deleted.
- **Not yet verified by an actual `iverilog`/Vivado run** — no synthesis or simulation toolchain available in this environment to confirm. Next step for the user: re-run `run.sh`'s functional simulation for at least one dataset first (confirm the lo/hi split didn't change simulation behavior — the accumulation math is untouched, only how results are addressed/written), then re-run `synth_design` in Vivado.

**8d — (superseded by the Vivado path above; kept as a record) Real Vivado synthesis, only-if-Vivado-available fallback text:**
- Run `synth_design` (not full implementation) on `falin_cnn_top.v` + its 4 submodules alone, targeting `xc7z020clg400-1`, for 1-2 representative `NUM_CLASSES`/`ROM_DIR` configs (e.g. ElectricDevices at 7 classes and FacesUCR at 14, to show the classifier's class-count scaling) — no PS7, no AXI, no bitstream needed for this. Pull the real LUT/FF/BRAM/DSP utilization report and the timing summary (Fmax). This directly answers the "no LUT count, BRAM usage, timing data" gap flagged for this paper, with real synthesis numbers instead of the Yosys coarse cell-count estimate (1329 cells / 12 `$mem_v2` / ~54 real multipliers) or a pure MAC-based argument. Areas should be nearly identical across datasets (only the classifier's `NUM_CLASSES` and comparator width change) — this is a sanity confirmation, not 10 separate synthesis runs.
- Full block design + bitstream + PYNQ board deployment remain explicit future work in the paper (honest framing, not a hidden gap) — not attempted this cycle given the Aug 2 deadline.

**Board status:** no physical PYNQ-Z2 as of 2026-07-24 — moot for this cycle since 8c stops at synthesis, not bitstream/deployment.

---

## Task 9 — LSTM-FCN temporal baseline: DROPPED (2026-07-24, see Locked decisions)

Originally scoped as local LSTM-FCN retraining (software) + a from-scratch FPGA port, to replace the paper's borrowed LSTM-FCN comparison numbers. Reversed on reflection: `misc/results_LSTM.csv` reports Karim et al. 2018's own published numbers on the **standard, fixed UCR Archive train/test splits** — the same splits this project's own experiments use. Citing a prior paper's numbers on the official splits of a public benchmark is standard, accepted TSC practice (not something every follow-up paper is expected to re-implement), provided it's properly cited (already is, ref [2]). No further work needed here; this section is kept only as a decision record.
