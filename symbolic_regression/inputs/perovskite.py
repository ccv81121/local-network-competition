"""Single and double perovskites: descriptor definitions and parent-grouped folds."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

SEED = 56
N_FOLDS = 5


def _metadata(name: str) -> tuple[str, str, str, str]:
    """Physical sector, role hint, unit and dimension class from the descriptor name."""
    if name.startswith(("chem_A", "chem_B", "chem_X", "chem_BX", "chem_AX", "dp_BBprime")):
        return "elemental_composition", "onsite", "mixed_primitive", "ELEMENTAL"
    spread = ("std", "cv", "dev", "contrast", "offcenter")
    if name.startswith(("oct_", "dp_oct")):
        role = "dephasing" if any(t in name for t in spread) else "geometry"
        return "local_octahedral_geometry", role, "A_or_dimensionless", "GEOMETRY"
    if name.startswith(("network_", "order_")):
        role = "dephasing" if any(t in name for t in ("std", "deviation")) else "network"
        return "octahedral_network", role, "A_deg_or_dimensionless", "NETWORK"
    if name.startswith("cage_"):
        role = "configuration" if any(t in name for t in ("std", "cv", "offcenter", "dev")) \
            else "scale"
        return "cage_environment", role, "A_or_dimensionless", "ENVIRONMENT"
    role = "scale" if "mean" in name or "volume" in name else "configuration"
    return "global_structure", role, "mixed", "GLOBAL"


def _csv_round_trip(frame: pd.DataFrame) -> pd.DataFrame:
    """Write and re-read a table as the original builders passed it between steps."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, float_format="%.17g")
    buffer.seek(0)
    return pd.read_csv(buffer)


def definitions(names: list[str], X: np.ndarray, round_trips: int = 1) -> pd.DataFrame:
    """Descriptor definitions with the range of each descriptor over the rows.

    The published registries were built from these definitions after
    ``round_trips`` CSV round trips (pandas' default float parser), which can
    move a range value by one unit in the last place; repeating them keeps
    the registry bytes identical.
    """
    rows = []
    for i, name in enumerate(names):
        sector, role, unit, dimension = _metadata(name)
        v = X[:, i]
        rows.append({"name": name, "physical_sector": sector, "unit": unit,
                     "dimension_signature": dimension, "formula_eligible": True,
                     "role_hint": role, "redundancy_group": name.rsplit("_", 1)[0],
                     "minimum": float(v.min()), "maximum": float(v.max()),
                     "median": float(np.median(v)),
                     "allowed_transforms": "raw;abs;power;spower;scale;binary",
                     "definition": name})
    frame = pd.DataFrame(rows)
    for _ in range(round_trips):
        frame = _csv_round_trip(frame)
    return frame


def grouped_folds(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Five class-stratified folds that keep every parent compound in one fold."""
    from sklearn.model_selection import StratifiedGroupKFold
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = np.full(len(y), -1, dtype=np.int16)
    for f, (_, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        folds[test] = f
    return folds


def protocol_rows(samples: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    """Row table of a perovskite protocol: every row is a scored hard-core row."""
    parent = samples.parent_formula.astype(str)
    return pd.DataFrame({
        "sample_id": samples.sample_id.astype(str), "curve_id": samples.curve_id.astype(str),
        "parent_formula": parent, "relation_component_id": parent, "cv_group_id": parent,
        "fold_id": folds,
        "is_multiconfig": samples.mode_design_label.notna().to_numpy(),
        "mode_design_label": samples.mode_design_label.fillna("").astype(str),
        "is_development": True, "is_all_primary": True, "is_hard_core": True,
        "is_soft_all": False, "challenge_target": False, "challenge_guard": False,
        "target": samples.target.astype(np.int8).to_numpy(),
    })
