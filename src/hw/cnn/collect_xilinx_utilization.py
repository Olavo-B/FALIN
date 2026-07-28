# /*****************************************************************************/
#  * File: collect_xilinx_utilization.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2026-07-24
#  * Description: Scans results/<dataset>/hw/xilinx_utilization_report.json
#  * (written by Yosys's `synth_xilinx -family xc7` + `stat -json`, see
#  * export_cnn_weights.py's write_xilinx_synth_script) and buckets the real
#  * Xilinx-primitive cell counts into the categories Vivado's own
#  * report_utilization uses (LUT, LUTRAM, FF, BRAM, DSP, BUFG), writing one
#  * row per dataset to results/hw_xilinx_utilization_summary.csv.
#  *
#  * Caveat (see write_xilinx_synth_script's docstring for the full reasoning):
#  * no Vivado is available locally, so this is Yosys's own technology mapping
#  * to Xilinx 7-series primitives - a real, open-source estimate, not
#  * Vivado's synthesis engine. Expect ballpark agreement, not an exact match
#  * to what `report_utilization` would print. Anything not recognized below
#  * (e.g. CARRY4, IBUF/OBUF, GND/VCC) is summed into "Other" rather than
#  * silently dropped, so nothing goes missing from the accounting.
#  *
#  * Pure stdlib - no torch/numpy needed, safe to run standalone.
# /*****************************************************************************/

import csv
import glob
import json
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
HW_VALIDATION_CSV = os.path.join(RESULTS_DIR, 'hw_validation_summary.csv')
OUTPUT_CSV = os.path.join(RESULTS_DIR, 'hw_xilinx_utilization_summary.csv')

# Cell-type -> bucket, based on Yosys's synth_xilinx -family xc7 output naming.
BUCKETS = {
    'LUT': {'LUT1', 'LUT2', 'LUT3', 'LUT4', 'LUT5', 'LUT6'},
    'LUTRAM': {
        'RAM32X1D', 'RAM32X1S', 'RAM32X2S', 'RAM64X1D', 'RAM64X1S',
        'RAM128X1D', 'RAM128X1S', 'RAM256X1S', 'SRL16E', 'SRLC32E',
        'RAM32M', 'RAM64M',
    },
    'FF': {'FDRE', 'FDSE', 'FDCE', 'FDPE', 'FDRE_1', 'FDSE_1', 'FDCE_1', 'FDPE_1'},
    'BRAM': {'RAMB18E1', 'RAMB36E1', 'RAMB18E2', 'RAMB36E2'},
    'DSP': {'DSP48E1', 'DSP48E2'},
    'BUFG': {'BUFG', 'BUFGCTRL', 'BUFGP'},
}
CELL_TO_BUCKET = {cell: bucket for bucket, cells in BUCKETS.items() for cell in cells}


def load_num_classes_by_dataset():
    mapping = {}
    if not os.path.exists(HW_VALIDATION_CSV):
        return mapping
    with open(HW_VALIDATION_CSV, 'r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            mapping[row['Dataset']] = row.get('NumClasses', '')
    return mapping


def bucket_cell_counts(cells_by_type):
    counts = {bucket: 0 for bucket in BUCKETS}
    other = 0
    for cell_type, count in cells_by_type.items():
        bucket = CELL_TO_BUCKET.get(cell_type)
        if bucket is not None:
            counts[bucket] += count
        else:
            other += count
    counts['Other'] = other
    return counts


def main():
    num_classes_by_dataset = load_num_classes_by_dataset()

    pattern = os.path.join(RESULTS_DIR, '*', 'hw', 'xilinx_utilization_report.json')
    report_paths = sorted(glob.glob(pattern))

    if not report_paths:
        print(f"[!] No xilinx_utilization_report.json files found under {RESULTS_DIR}/*/hw/. "
              "Run ./run_synth_xilinx.sh first.")
        return

    fieldnames = ['Dataset', 'NumClasses', 'LUT', 'LUTRAM', 'FF', 'BRAM', 'DSP', 'BUFG', 'Other']
    rows = []
    for path in report_paths:
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path, 'r', encoding='utf-8') as f:
            report = json.load(f)

        cells_by_type = report['design'].get('num_cells_by_type', {})
        counts = bucket_cell_counts(cells_by_type)

        row = {'Dataset': dataset_name, 'NumClasses': num_classes_by_dataset.get(dataset_name, '')}
        row.update(counts)
        rows.append(row)

        print(f"[*] {dataset_name}: LUT={counts['LUT']}, LUTRAM={counts['LUTRAM']}, "
              f"FF={counts['FF']}, BRAM={counts['BRAM']}, DSP={counts['DSP']}, "
              f"BUFG={counts['BUFG']}, Other={counts['Other']}")

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[*] Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
