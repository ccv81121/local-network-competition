#!/usr/bin/env python3
"""Replay of the single-perovskite relation (Eq. S20 of the source record) on the nine BaSn1-xTixO3 states from their
raw features (sp_s20_raw_inputs_full9.csv, scales in sp_s20_descriptor_scales.csv).

Run with Python 3.11+: python compute_sp_s20_scores.py
Inputs and output are resolved beside this script.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = HERE / "sp_s20_raw_inputs_full9.csv"
SCALES = HERE / "sp_s20_descriptor_scales.csv"
OUTPUT = HERE / "sp_s20_scores_9_states.csv"
TAU = -18.93835559189904
EPSILON = 1.0e-6
TOLERANCE = 1.0e-6
FEATURES = (
    "chem_B_mass_std",
    "chem_X_group_mean",
    "order_species_entropy",
    "order_R",
    "oct_BX6_angle_dev_mean",
    "chem_A_rvdw_mean",
    "chem_B_group_mean",
    "chem_B_mass_mean",
)


def spow(u: float, p: float) -> float:
    """Signed power: sgn(u) * abs(u)**p."""
    return math.copysign(abs(u) ** p, u) if u else 0.0


def sgeom(u: float, v: float) -> float:
    """Signed geometric mean: sgn(u*v) * sqrt(abs(u*v))."""
    product = u * v
    return math.copysign(math.sqrt(abs(product)), product) if product else 0.0


def read_scales(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))
    scales = {r["feature_name"]: float(r["frozen_scale_s_j"]) for r in records}
    missing = [name for name in FEATURES if name not in scales]
    if missing:
        raise ValueError(f"missing frozen scales: {missing}")
    return scales


def main() -> int:
    scales = read_scales(SCALES)
    with INPUT.open(newline="", encoding="utf-8-sig") as f:
        source_rows = list(csv.DictReader(f))
    if len(source_rows) != 9 or len({r["sample_id"] for r in source_rows}) != 9:
        raise ValueError("expected nine unique sample_id rows")
    source_rows.sort(key=lambda r: float(r["x"]))
    out_rows = []
    for row in source_rows:
        z = {name: float(row[name]) / scales[name] for name in FEATURES}
        bracket = (
            3.0 * (z["chem_B_mass_std"] ** (2.0 / 3.0) - z["chem_X_group_mean"])
            - spow(2.0 * z["order_species_entropy"], 2.0)
            - spow(
                (z["order_R"] ** (3.0 / 2.0))
                / (abs(z["oct_BX6_angle_dev_mean"]) + EPSILON),
                3.0 / 2.0,
            )
        )
        pi = bracket / 3.0 + sgeom(
            (2.0 / 3.0)
            * ((1.0 / 3.0) * z["chem_A_rvdw_mean"] + z["chem_B_group_mean"]),
            z["chem_B_mass_mean"],
        )
        distance = -(pi - TAU)  # orientation = -1
        pi_frozen = float(row["Pi_frozen"])
        d_frozen = float(row["d_frozen"])
        pred_v = int(distance > 0.0)
        truth_v = int(row["V_truth"])
        result = dict(row)
        result.update({f"{name}_scaled": z[name] for name in FEATURES})
        result.update({
            "Pi_recomputed": pi,
            "Pi_abs_difference": abs(pi - pi_frozen),
            "tau_exact": TAU,
            "orientation": -1,
            "d_recomputed": distance,
            "d_abs_difference": abs(distance - d_frozen),
            "V_pred_from_d": pred_v,
            "V_correct_from_d": int(pred_v == truth_v),
            "replay_tolerance_pass": int(
                abs(pi - pi_frozen) <= TOLERANCE
                and abs(distance - d_frozen) <= TOLERANCE
            ),
        })
        out_rows.append(result)
    fields = list(out_rows[0])
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    max_pi = max(float(r["Pi_abs_difference"]) for r in out_rows)
    max_d = max(float(r["d_abs_difference"]) for r in out_rows)
    correct = sum(int(r["V_correct_from_d"]) for r in out_rows)
    passed = all(int(r["replay_tolerance_pass"]) == 1 for r in out_rows)
    print(f"rows={len(out_rows)} max_abs_Pi_error={max_pi:.17g} max_abs_d_error={max_d:.17g}")
    print(f"equation_replay={'PASS' if passed else 'FAIL'} tolerance={TOLERANCE:g} V_identity={correct}/9")
    print(f"output={OUTPUT}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
