# /*****************************************************************************/
#  * File: estimate_transform_cost.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2026-07-24
#  * Description: Analytical (no new RTL, per explicit decision 2026-07-24)
#  * cost estimate of the "raw sensor signal -> 96x96 image" front-end
#  * (GAF encoding + bilinear resize to the fixed target resolution) for all
#  * 10 paper datasets, so the paper can honestly account for this step's
#  * cost alongside the CNN's, without re-scoping the FPGA claim back to
#  * include it (that stays CNN-only, see Locked decisions in
#  * misc/docs/fpga_implementation_plan.md).
#  *
#  * Method - two independently-justified pieces, not guesses:
#  *   1. GAF encoding: src/hw/gaf_encoder/gaf_encoder.v is a REAL, built,
#  *      simulated Verilog design (Task 6), but only for SEQ_LEN=96
#  *      (ElectricDevices). Its architecture is fully parametric in SEQ_LEN
#  *      (L) and has a known, fixed structure - MINMAX (L cycles) + NORM_INIT
#  *      (1) + NORMALIZE (L cycles) + OUTER_PROD (L cycles, L parallel-2
#  *      multiplier columns per row) - so cycles(L) = 3L+O(1) and
#  *      multipliers(L) = 2L follow directly from the SAME already-verified
#  *      design, not a new estimate. MACs(L) = 2*L^2 (L rows x 2L multiplies/
#  *      row in the outer product). This is an EXTRAPOLATION of a real
#  *      architecture's known parametric cost, not a new architecture, and it
#  *      has NOT been built/simulated/synthesized for any L != 96 - the
#  *      numbers below are analytical, not measured, for every dataset except
#  *      ElectricDevices.
#  *   2. Resize: temporal2img.py's Temporal2ImageDataset always bilinearly
#  *      interpolates the native LxL GAF matrix down/up to the fixed 96x96
#  *      target (see its F.interpolate call). Bilinear interpolation costs
#  *      ~4 MACs per OUTPUT pixel (2x2 neighborhood), independent of the
#  *      source resolution L - this stays software-only, no hardware
#  *      equivalent exists or is being built.
#  *
#  * IMPORTANT CAVEAT (state this in the paper): gaf_encoder.v's OUTER_PROD
#  * stage uses 2L multipliers fully in parallel. That is only physically
#  * realizable on the target xc7z020 (220 DSP48E1 slices total) for small L -
#  * already noted in Task 8 that L=96's 192 multipliers alone, combined with
#  * the CNN's own ~54, is near the chip's DSP budget. The MACs figures below
#  * are architecture-independent (total multiply-accumulate work is the same
#  * no matter how much parallelism is thrown at it) and are the primary,
#  * robust comparison point. The "cycles" figures assume the SAME full-
#  * parallelism architecture scaled up, which would NOT fit on this chip for
#  * large L without a redesign trading multipliers for more cycles (time-
#  * multiplexing) - they illustrate a best-case latency bound, not a
#  * feasible-as-is design point, for datasets other than ElectricDevices.
#  *
#  * Pure stdlib - no torch/numpy needed, safe to run standalone. Writes
#  * results/gaf_transform_cost_estimate.csv.
# /*****************************************************************************/

import csv
import os

# config_loader.py lives in this same directory (src/sw/utils/), so this is
# a plain sibling import - no sys.path tricks needed (matches how this
# script would be invoked directly: `python src/sw/utils/estimate_transform_cost.py`).
from config_loader import load_config

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..')
OUTPUT_CSV = os.path.join(BASE_DIR, 'results', 'gaf_transform_cost_estimate.csv')

# Sequence length per dataset (Table II of the paper). Not derived from
# config.json since that only lists dataset names, not their raw lengths -
# these come straight from the UCR Archive TSVs / the paper's own table.
SEQ_LEN_BY_DATASET = {
    'ElectricDevices': 96,
    'StarLightCurves': 1024,
    'Wafer': 152,
    'ECG5000': 140,
    'TwoPatterns': 128,
    'FordA': 500,
    'FordB': 500,
    'Yoga': 426,
    'Mallat': 1024,
    'FacesUCR': 131,
}

TARGET_SIZE = 96
RESIZE_MACS_PER_SAMPLE = TARGET_SIZE * TARGET_SIZE * 4  # bilinear, ~4 MACs/output pixel

# Already-established FalinCNN numbers (paper's Fig. 4 / Task 7 measurement),
# used only as the denominator for a relative-cost comparison.
CNN_MACS = 22_600_000
CNN_CYCLES = 3_370_000


def gaf_hardware_equivalent_cost(seq_len):
    """
    Extrapolates gaf_encoder.v's OWN, already-built-and-verified architecture
    (see module docstring) to a different SEQ_LEN. Returns
    (multipliers, cycles, macs). Exact for seq_len=96 (matches the real,
    simulated design); analytical for every other value.
    """
    multipliers = 2 * seq_len
    cycles = 3 * seq_len + 3  # MINMAX + NORM_INIT + NORMALIZE + OUTER_PROD + small FSM overhead
    macs = 2 * seq_len * seq_len
    return multipliers, cycles, macs


def main():
    config = load_config()
    datasets = config['datasets']

    rows = []
    print(f"{'Dataset':<16}{'SeqLen':>7}{'GAFMults':>10}{'GAFCycles':>11}{'GAFMACs':>12}"
          f"{'ResizeMACs':>12}{'TotalMACs':>12}{'MACs/CNN':>10}{'Cycles/CNN':>12}")

    for dataset in datasets:
        seq_len = SEQ_LEN_BY_DATASET[dataset]
        gaf_mults, gaf_cycles, gaf_macs = gaf_hardware_equivalent_cost(seq_len)
        # ElectricDevices' native length already equals the 96x96 target, so
        # the resize step is a no-op for it (still executed by
        # temporal2img.py today, but with zero actual resampling cost).
        resize_macs = 0 if seq_len == TARGET_SIZE else RESIZE_MACS_PER_SAMPLE
        total_macs = gaf_macs + resize_macs
        macs_ratio = total_macs / CNN_MACS
        cycles_ratio = gaf_cycles / CNN_CYCLES

        rows.append({
            'Dataset': dataset,
            'SeqLen': seq_len,
            'GAF_Multipliers': gaf_mults,
            'GAF_Cycles': gaf_cycles,
            'GAF_MACs': gaf_macs,
            'Resize_MACs': resize_macs,
            'Total_Transform_MACs': total_macs,
            'TransformMACs_over_CNNMACs_pct': round(macs_ratio * 100, 4),
            'GAFCycles_over_CNNCycles_pct': round(cycles_ratio * 100, 5),
        })

        print(f"{dataset:<16}{seq_len:>7}{gaf_mults:>10}{gaf_cycles:>11}{gaf_macs:>12,}"
              f"{resize_macs:>12,}{total_macs:>12,}{macs_ratio*100:>9.3f}%{cycles_ratio*100:>11.4f}%")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[*] Wrote {len(rows)} rows to {OUTPUT_CSV}")
    print("[*] Reminder: GAF numbers are exact only for ElectricDevices (seq_len=96, the only "
          "length gaf_encoder.v has actually been built/simulated for) - analytical extrapolation "
          "for every other dataset. See this file's module docstring for the DSP-budget caveat on "
          "the 'cycles' column.")


if __name__ == '__main__':
    main()
