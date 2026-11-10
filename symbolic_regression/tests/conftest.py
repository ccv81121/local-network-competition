"""A small synthetic data set in the input layout the search reads."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

N_ROWS = 80
DOMAINS = ["POSITIVE", "POSITIVE", "NONNEGATIVE", "SIGNED", "POSITIVE", "NONNEGATIVE"]

TINY = {
    "profile": "small", "initial_component_target": 300, "component_population": 128,
    "component_persist_per_generation": 64, "component_context_candidates": 64,
    "component_context_anchors": 16, "component_context_screen_anchors": 8,
    "component_context_full_audit_candidates": 16, "component_capacity": 3000,
    "component_feedback_reserve": 500, "skeleton_population": 128,
    "skeleton_max_generations": 4, "skeleton_min_generations": 4, "skeleton_patience": 4,
    "feedback_component_proposals": 64, "feedback_component_persist": 32,
    "feedback_component_inject": 24, "feedback_component_migrate": 8,
    "feedback_side_population": 32, "beam_topologies": 32, "performance_archive_cap": 128,
    "topology_archive_cap": 64, "prediction_archive_cap": 64, "final_candidate_cap": 64,
    "jobs": 1,
}


def write_inputs(root: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(7)
    X = np.abs(rng.normal(1.0, 0.5, size=(N_ROWS, len(DOMAINS)))) + 0.1
    X[:, 3] = rng.normal(size=N_ROWS)
    y = (X[:, 0] - 0.8 * X[:, 1] + 0.3 * X[:, 3] + 0.2 * rng.normal(size=N_ROWS) > 0.2)
    protocol = root / "protocol"
    arrays = protocol / "RF_PROTOCOL_BUNDLE" / "arrays"
    features = protocol / "RF_PROTOCOL_BUNDLE" / "features"
    arrays.mkdir(parents=True)
    features.mkdir(parents=True)
    hard = np.arange(N_ROWS) < 60
    for name, value in {
        "X_full_clean_float64": X.astype(np.float64), "y_int8": y.astype(np.int8),
        "mask_all_primary_bool": np.ones(N_ROWS, bool), "mask_hard_core_bool": hard,
        "mask_soft_all_bool": ~hard, "fold_id_int16": (np.arange(N_ROWS) % 5).astype(np.int16),
        "mask_multiconfig_bool": np.arange(N_ROWS) % 4 == 0,
    }.items():
        np.save(arrays / f"{name}.npy", value)
    names = [f"d{i}" for i in range(len(DOMAINS))]
    (features / "formula_full_feature_names.json").write_text(json.dumps(names))
    pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(N_ROWS)],
        "curve_id": [f"c{i // 4}" for i in range(N_ROWS)],
        "parent_formula": [f"p{i // 8}" for i in range(N_ROWS)],
        "relation_component_id": [f"r{i % 3}" for i in range(N_ROWS)],
        "is_multiconfig": np.arange(N_ROWS) % 4 == 0,
    }).to_csv(protocol / "protocol_rows.csv", index=False)
    space = root / "space"
    space.mkdir()
    pd.DataFrame({"feature_index": range(len(names)), "feature_name": names,
                  "input_domain": DOMAINS}).to_csv(space / "terminal_registry.csv", index=False)
    pd.DataFrame({"feature_index": range(len(names)),
                  "primary_channel": ["LOCAL", "NETWORK", "LOCAL", "GEOMETRY", "NETWORK",
                                      "ENVIRONMENT"],
                  "primary_role": ["LOCAL_COHERENT", "NETWORK_COHERENT"] * 3,
                  "allowed_roles": ["", "DEPHASING", "", "", "", ""]}).to_csv(
        space / "PHYSICAL_CHANNEL_ROLE_REGISTRY.csv", index=False)
    pd.DataFrame({"feature_index": range(len(names)), "sampling_weight": [.3, .2, .2, .1, .1, .1],
                  "search_tier": ["A", "A", "B", "B", "C", "C"]}).to_csv(
        space / "RF_SEARCH_GUIDANCE_REGISTRY.csv", index=False)
    return protocol, space


@pytest.fixture(scope="session")
def inputs(tmp_path_factory):
    return write_inputs(tmp_path_factory.mktemp("inputs"))
