# /*****************************************************************************/
#  * File: collect_area_summary.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2026-07-24
#  * Description: Scans results/<dataset>/hw/area_report.json (written by
#  * Yosys's coarse `stat -json` for each dataset's synth.ys, see
#  * export_cnn_weights.py's write_synth_wrapper_and_script) and collects
#  * them into a single results/hw_area_summary.csv - one row per dataset,
#  * so the paper can report area as (expected, per Task 8) nearly constant
#  * across all 10 datasets, backed by real numbers instead of an assumption.
#  * NumClasses is joined in from results/hw_validation_summary.csv (written
#  * by export_cnn_weights.py) when available. Pure stdlib - no torch/numpy
#  * needed, safe to run standalone.
# /*****************************************************************************/

import csv
import glob
import json
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
HW_VALIDATION_CSV = os.path.join(RESULTS_DIR, 'hw_validation_summary.csv')
OUTPUT_CSV = os.path.join(RESULTS_DIR, 'hw_area_summary.csv')


def load_num_classes_by_dataset():
    """Best-effort Dataset -> NumClasses map from hw_validation_summary.csv."""
    mapping = {}
    if not os.path.exists(HW_VALIDATION_CSV):
        return mapping
    with open(HW_VALIDATION_CSV, 'r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            mapping[row['Dataset']] = row.get('NumClasses', '')
    return mapping


def main():
    num_classes_by_dataset = load_num_classes_by_dataset()

    pattern = os.path.join(RESULTS_DIR, '*', 'hw', 'area_report.json')
    report_paths = sorted(glob.glob(pattern))

    if not report_paths:
        print(f"[!] No area_report.json files found under {RESULTS_DIR}/*/hw/. "
              "Run ./run_synth.sh first.")
        return

    rows = []
    for path in report_paths:
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path, 'r', encoding='utf-8') as f:
            report = json.load(f)

        design = report['design']
        total_cells = design['num_cells']
        cells_by_type = design.get('num_cells_by_type', {})
        mem_cells = cells_by_type.get('$mem_v2', 0)
        mul_cells = cells_by_type.get('$mul', 0)

        rows.append({
            'Dataset': dataset_name,
            'NumClasses': num_classes_by_dataset.get(dataset_name, ''),
            'TotalCells': total_cells,
            'MemCells': mem_cells,
            'MulCells': mul_cells,
        })
        print(f"[*] {dataset_name}: total_cells={total_cells}, mem_cells={mem_cells}, mul_cells={mul_cells}")

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Dataset', 'NumClasses', 'TotalCells', 'MemCells', 'MulCells'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[*] Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
