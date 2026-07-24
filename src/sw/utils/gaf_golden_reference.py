# /*****************************************************************************/
#  * File: gaf_golden_reference.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2026-07-23
#  * Description: Standalone, hardware-oriented golden model for the GAF
#  * (Gramian Angular Summation Field) encoding, and stimulus/golden-vector
#  * generator for the Vitis HLS testbench (src/hw/gaf_encoder). Exploits
#  * cos(arccos(x)) = x and sin(arccos(x)) = sqrt(1 - x^2) to drop the
#  * trigonometric unit entirely - only normalization, a square root, and an
#  * outer product remain. Cross-checked against temporal2img.py's original
#  * arccos-based _gramian_encoding (max abs diff ~4e-16, floating point noise)
#  * before being trusted as HLS ground truth.
#  * See Task 3/6 of misc/docs/fpga_implementation_plan.md.
# /*****************************************************************************/

import os
import sys
import numpy as np

# This script lives in src/sw/aux/ - temporal2img.py is one level up, in
# src/sw/. Running this file directly only auto-adds src/sw/aux/ to
# sys.path (Python adds the script's own directory, not its parent), so
# the parent needs to be added explicitly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from temporal2img import Temporal2ImageDataset  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..')
UCR_ARCHIVE_PATH = os.path.join(BASE_DIR, 'misc', 'data', 'UCRArchive_2018')
TB_DATA_DIR = os.path.join(BASE_DIR, 'src', 'hw', 'gaf_encoder', 'tb_data')

DATASET_NAME = 'ElectricDevices'
NUM_GOLDEN_SAMPLES = 5
CROSS_CHECK_TOLERANCE = 1e-4


def gaf_encode_hw(x):
    """
    Hardware-oriented GAF (GASF) encoding: gaf[i,j] = cos(phi_i + phi_j),
    computed as cos_phi[i]*cos_phi[j] - sin_phi[i]*sin_phi[j] without ever
    calling arccos/cos/sin, since cos(arccos(x)) = x and
    sin(arccos(x)) = sqrt(1 - x^2). Output is always exactly in [-1, 1]
    (it's a cosine of a sum of angles), which is what fixes gaf_fixed_t's
    range in gaf_encoder.h.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    min_val = np.min(x)
    max_val = np.max(x)

    if max_val == min_val:
        x_norm = np.zeros_like(x)
    else:
        x_norm = ((x - min_val) / (max_val - min_val)) * 2 - 1
    x_norm = np.clip(x_norm, -1.0, 1.0)

    cos_phi = x_norm
    sin_phi = np.sqrt(1.0 - x_norm ** 2)

    return np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)


def _write_vector(file_path, vector):
    with open(file_path, 'w') as f:
        f.write(' '.join(f'{v:.10f}' for v in vector) + '\n')


def _write_matrix(file_path, matrix):
    with open(file_path, 'w') as f:
        for row in matrix:
            f.write(' '.join(f'{v:.10f}' for v in row) + '\n')


def main():
    test_data = np.loadtxt(
        os.path.join(UCR_ARCHIVE_PATH, DATASET_NAME, f'{DATASET_NAME}_TEST.tsv'),
        delimiter='\t'
    )
    X_test = np.nan_to_num(test_data[:, 1:], nan=0.0)
    seq_len = X_test.shape[1]
    print(f"[*] {DATASET_NAME}: seq_len={seq_len} (must match SEQ_LEN in src/hw/gaf_encoder/gaf_encoder.h)")

    os.makedirs(TB_DATA_DIR, exist_ok=True)

    # Cross-check against the original (redundant-trig) implementation to
    # confirm the hardware-oriented rewrite is mathematically identical
    # before trusting it as ground truth for the HLS testbench. No resize
    # distortion expected here since target_size == seq_len for ElectricDevices.
    reference = Temporal2ImageDataset(
        X_test[:NUM_GOLDEN_SAMPLES], target_size=(seq_len, seq_len), estrategy='gaf'
    )

    for i in range(NUM_GOLDEN_SAMPLES):
        sample = X_test[i]
        gaf_hw = gaf_encode_hw(sample)
        gaf_ref = reference.images[i, 0].numpy()

        max_abs_diff = np.max(np.abs(gaf_hw - gaf_ref))
        assert max_abs_diff < CROSS_CHECK_TOLERANCE, (
            f"Sample {i}: hardware-oriented GAF diverges from temporal2img.py's "
            f"_gramian_encoding by {max_abs_diff} - the trig-identity rewrite is wrong."
        )

        _write_vector(os.path.join(TB_DATA_DIR, f'stimulus_{i:02d}.txt'), sample)
        _write_matrix(os.path.join(TB_DATA_DIR, f'golden_{i:02d}.txt'), gaf_hw)

    print(f"[*] Verified {NUM_GOLDEN_SAMPLES} samples match temporal2img.py's GAF exactly "
          f"(within {CROSS_CHECK_TOLERANCE}).")
    print(f"[*] Wrote stimulus/golden vectors to: {TB_DATA_DIR}")


if __name__ == '__main__':
    main()
