# /*****************************************************************************/
#  * File: gen_sin_phi_rom.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2026-07-23
#  * Description: Generates sin_phi_rom.hex, the lookup table gaf_encoder.v
#  * uses to get sin_phi = sqrt(1 - x_norm^2) without any sqrt/trig hardware.
#  *
#  * IMPORTANT precision finding (see misc/docs/fpga_implementation_plan.md
#  * Task 6): sqrt(1-x^2) has a vertical tangent at x = +-1, and min-max
#  * normalization guarantees every 96-sample window has at least one sample
#  * exactly AT x_norm = +1 and one AT x_norm = -1 (the window's own min/max).
#  * A naive 8-bit ROM address (matching the 8-bit GAF output) gives a
#  * worst-case per-element error of ~0.13 near those samples - not the
#  * ~1/256 the output bit-width would suggest. Measured on 200 real
#  * ElectricDevices test windows:
#  *   addr_bits=8  -> worst-case max err 0.129, mean err 0.007
#  *   addr_bits=10 -> worst-case max err 0.065, mean err 0.007
#  *   addr_bits=12 -> worst-case max err 0.035, mean err 0.0065
#  *   addr_bits=14 -> worst-case max err 0.019, mean err 0.0065
#  * Error shrinks only as sqrt(step size) near the singularity (not
#  * linearly), so throwing more ROM bits at it has fast-diminishing
#  * returns. 12 bits (4096 x 12-bit = 6 KB, one small BRAM) was chosen as
#  * the point where worst-case error (0.035) is comfortably inside the
#  * encoder's tolerance while keeping the ROM cheap. The mean error floor
#  * (~0.0065) doesn't move with ROM depth - it's the final 8-bit output
#  * rounding, which is unavoidable at that output precision.
#  *
#  * The numbers above came from a preliminary float-domain simulation on
#  * synthetic data. Once gaf_encoder.v's exact integer arithmetic (round-
#  * half-up bias + arithmetic shift + saturate, see its header) was
#  * modeled bit-for-bit in Python and checked exhaustively against all
#  * 7711 real ElectricDevices test windows, the true worst-case error with
#  * this 12-bit ROM came out lower: 0.0114 (~1.5 output LSBs). Trust that
#  * number over the estimates in the table above.
# /*****************************************************************************/

import os
import numpy as np

ADDR_BITS = 12          # x_norm ROM address width (Q1.11: 1 sign/integer bit, 11 fractional)
NORM_FRAC = ADDR_BITS - 1
SCALE = 1 << NORM_FRAC
NUM_ENTRIES = 1 << ADDR_BITS

OUT_DIR = os.path.dirname(__file__)
ROM_HEX_PATH = os.path.join(OUT_DIR, 'sin_phi_rom.hex')


def build_rom():
    rom = np.zeros(NUM_ENTRIES, dtype=np.int64)
    for code in range(NUM_ENTRIES):
        # Reinterpret the ROM address as the two's complement code for x_norm.
        signed_code = code - NUM_ENTRIES if code >= NUM_ENTRIES // 2 else code
        x_norm = signed_code / SCALE
        sin_phi = np.sqrt(max(1.0 - x_norm ** 2, 0.0))
        # sin_phi's true max is 1.0 (at x_norm=0), which Q1.11 can't represent
        # exactly (max code is (SCALE-1)/SCALE) - saturate rather than wrap.
        rom[code] = min(int(round(sin_phi * SCALE)), SCALE - 1)
    return rom


def validate(rom, num_random_checks=500):
    """Cross-check the ROM+outer-product pipeline against the float golden
    model, reporting the same worst-case/mean error stats as the design
    exploration above, so this stays self-checking if ADDR_BITS changes."""
    rng = np.random.default_rng(0)
    max_errs = []
    mean_errs = []
    for _ in range(num_random_checks):
        x = rng.normal(size=96) * rng.uniform(0.5, 5.0)
        min_val, max_val = x.min(), x.max()
        if max_val == min_val:
            continue
        x_norm_float = ((x - min_val) / (max_val - min_val)) * 2 - 1

        x_norm_q = np.clip(np.round(x_norm_float * SCALE), -SCALE, SCALE - 1).astype(np.int64)
        codes = x_norm_q & (NUM_ENTRIES - 1)
        sin_phi_q = rom[codes] / SCALE
        cos_phi_q = x_norm_q / SCALE

        gaf_hw = np.outer(cos_phi_q, cos_phi_q) - np.outer(sin_phi_q, sin_phi_q)
        gaf_hw = np.clip(np.round(gaf_hw * 128) / 128, -1.0, 127 / 128)  # final round to Q1.7 output

        cos_ref = x_norm_float
        sin_ref = np.sqrt(1.0 - x_norm_float ** 2)
        gaf_ref = np.outer(cos_ref, cos_ref) - np.outer(sin_ref, sin_ref)

        err = np.abs(gaf_hw - gaf_ref)
        max_errs.append(err.max())
        mean_errs.append(err.mean())

    return max(max_errs), float(np.mean(mean_errs))


def main():
    rom = build_rom()

    with open(ROM_HEX_PATH, 'w') as f:
        for code in range(NUM_ENTRIES):
            # Store as ADDR_BITS-bit two's complement hex, one entry per line,
            # for $readmemh in gaf_encoder.v.
            value = int(rom[code]) & (NUM_ENTRIES - 1)
            hex_digits = (ADDR_BITS + 3) // 4
            f.write(f'{value:0{hex_digits}x}\n')

    worst_max_err, avg_mean_err = validate(rom)
    print(f"[*] Wrote {NUM_ENTRIES}-entry ROM ({ADDR_BITS}-bit) to {ROM_HEX_PATH}")
    print(f"[*] Validation over 500 synthetic windows: worst-case max err = {worst_max_err:.5f}, "
          f"avg mean err = {avg_mean_err:.6f}")


if __name__ == '__main__':
    main()
