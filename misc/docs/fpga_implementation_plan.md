# FALIN — FPGA Implementation Plan

Roadmap to port the FALIN pipeline (signal → GAF → FalinCNN) to an FPGA. XGBoost/TreeLUT classification stage is deferred to a later phase.

## Locked decisions

- **Board:** PYNQ-Z2 (Zynq-7020, part `xc7z020clg400-1`) — no physical board yet, so this is the simulation/synthesis target until hardware arrives.
- **Flow:** hls4ml (Vitis HLS) for the CNN; hand-written Vitis HLS C++ for the GAF encoder.
- **Image strategy:** GAF, not STFT. STFT needs an FFT core (e.g. Xilinx `xfft`) plus complex-magnitude logic — far more hardware than GAF requires.
- **Target dataset:** ElectricDevices. `seq_len = 96`, which equals the 96×96 target resolution used in the paper, so **no resize/interpolation logic is needed** for the first prototype. 7 classes, GAF+FalinCNN baseline accuracy ≈ 71–73% (`results/gafs/summary_metrics.csv`).
- **Scope:** FalinCNN only for this pass. XGBoost/TreeLUT (already cited in the paper via Khataei & Bazargan, FPGA 2025) comes after the CNN is working end-to-end in hardware.
- **Folder layout:** `src/` split into `src/sw/` (the existing Python training/quantization pipeline — unchanged internally, all cross-imports are peer files in the same folder so nothing needed updating) and `src/hw/` (HLS/RTL work, one subfolder per IP block — `src/hw/gaf_encoder/` for Task 6, later joined by the hls4ml CNN project and the Vivado integration).

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

`BatchNorm2d` in eval mode computes `y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta`. Since every conv in `falin_cnn.py` already has `bias=False`, fold as:

```python
scale = gamma / torch.sqrt(running_var + eps)
new_weight = conv.weight * scale.view(-1, 1, 1, 1)
new_bias = beta - running_mean * scale
```

- Apply to all 3 `BatchNorm2d` layers (`layer1`, `layer2`, `layer3`).
- Replace each `BatchNorm2d` with `nn.Identity()`, set `conv.bias = nn.Parameter(new_bias)`.
- Verify: run the folded model in `eval()` on the same test batch as the original and confirm outputs match within ~1e-5.

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
- `gaf_encoder_tb.v`: Icarus Verilog testbench, reads `tb_data/` vectors from `src/sw/gaf_golden_reference.py`.

**Key finding (read `gen_sin_phi_rom.py`'s header for the full analysis):** `sqrt(1-x²)` has a vertical tangent at `x = ±1`, and min-max normalization guarantees every 96-sample window has a sample landing exactly there (its own min and max). An 8-bit ROM address (matching the 8-bit GAF output) gives worst-case per-element error ~0.13 — far worse than the ~1/256 the output width suggests, because error near the singularity shrinks only as `sqrt(step size)`, not linearly. Measured on real ElectricDevices windows:

| ROM address bits | worst-case max error | mean error |
|---|---|---|
| 8  | 0.129 | 0.007 |
| 10 | 0.065 | 0.007 |
| 12 | 0.035 | 0.0065 |
| 14 | 0.019 | 0.0065 |

Chose **12-bit ROM address** (4096 × 12-bit, 6 KB — one small BRAM). Internal precision (`cos_phi`/`sin_phi`, Q1.11) is kept higher than the final 8-bit GAF output; rounding to Q1.7 only happens once, after the outer product — the standard "accumulate wide, round once at the end" rule for fixed-point hardware. Once `gaf_encoder.v`'s exact integer arithmetic (round-half-up bias, arithmetic shift, saturate) was modeled bit-for-bit in Python and checked exhaustively against all 7711 real ElectricDevices test windows, the true worst-case error came out at **0.0114** (better than the 0.035 float-domain estimate above — trust the exhaustive number).

Run (from `src/hw/gaf_encoder/`, after `python ../../sw/gaf_golden_reference.py`):
```
iverilog -g2012 -o sim gaf_encoder.v gaf_encoder_tb.v
vvp sim
yosys -p "read_verilog -sv gaf_encoder.v; synth; stat" 2>&1 | tee yosys_area_report.txt
```

**First simulation run found two real bugs, now fixed:** (1) module ports used unpacked SystemVerilog array syntax (`input ... foo [0:N-1]`), which Yosys's Verilog frontend rejects — flattened to 1D vectors (`signal_in_flat`, `gaf_out_flat`), unpacked/packed internally right at the port boundary. (2) the final rounding step sliced the wrong bit range (`rounded[21:14]` instead of `rounded[22:15]`, a 1-bit shift error) and used a fragile manual bit-disagreement overflow check instead of a plain signed comparison — both replaced with an explicit `SHIFT_AMOUNT`/`ROUND_BIAS`/`OUT_MAX`/`OUT_MIN` computation. Re-run the commands above and report the new results.

---

## Task 7 — Convert the quantized FalinCNN via hls4ml, run C-simulation

- Install `hls4ml` and Vitis HLS (check hls4ml's docs for the currently supported Vitis HLS version — this changes between hls4ml releases).
- Convert:
  ```python
  hls_model = hls4ml.converters.convert_from_onnx_model(
      model, output_dir=..., part='xc7z020clg400-1', ...
  )
  ```
- Generate a starting config with `hls4ml.utils.config_from_onnx_model`, then tune `Precision`/`ReuseFactor` per layer to trade latency vs area.
- Run `hls_model.compile()` + `hls_model.predict()` (C-sim) on the same test set and compare against the quantized PyTorch model — should match closely.
- Run `hls_model.build(csim=True, synth=True)` for the Vivado HLS resource/latency report, compare against the paper's projected 22.6M MACs figure.

---

## Task 8 — Integrate GAF module + CNN HLS project into a Vivado design for PYNQ-Z2

- Export both HLS designs as Vivado IP (RTL export from the GAF Vitis HLS project; hls4ml's `build()` export flag for the CNN).
- Create a Vivado block design targeting `xc7z020clg400-1`: Zynq PS7 block + both custom IPs, wired GAF output → CNN input (typically AXI4-Stream, depending on what hls4ml exposes by default).
- Generate the bitstream.
- **If the board is available:** deploy via a PYNQ overlay (`.bit` + `.hwh`), write a small Python driver on the PS/ARM side feeding raw ElectricDevices test samples, compare predictions to the software baseline.
- **If not:** validate via Vivado's behavioral/post-synthesis co-simulation instead.
- Final check: confirm end-to-end hardware accuracy matches the software baseline (~71–73%) within expected quantization loss.
