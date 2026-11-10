"""Two-edge search: end to end on synthetic paths, resume, replay, metrics, DP edge labels."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from symbolic_regression import dual_edge as de

TABLES = Path(__file__).resolve().parents[2] / "data" / "descriptor_tables"
EDGE_LABELS = Path(__file__).resolve().parents[2] / "data" / "perovskite_relations" / "dp_edge_targets.csv"


def write_edge_inputs(root: Path) -> dict:
    """Eight substitution paths of twelve states; edges at Γ follow two linear rules."""
    rng = np.random.default_rng(56)
    n_traj, per, nf = 8, 12, 8
    X = rng.normal(size=(n_traj * per, nf))
    sid, rows = [], []
    for t in range(n_traj):
        for j in range(per):
            i = t * per + j
            s = f"mp-{1000 + t}_X_k{j:02d}"
            sid.append(s)
            vg = X[i, 0] + 0.7 * X[i, 1] > 0.1
            cg = X[i, 2] - 0.8 * X[i, 3] > -0.2
            vk = [0, 0, 0] if vg else ([0.5, 0, 0] if t % 3 else [0.25, 0.25, 0])
            ck = [0, 0, 0] if cg else ([0.5, 0, 0] if t % 4 else [0.25, 0.25, 0])
            if not vg and not cg and j % 2 == 0:
                ck = vk
            row = {"sample_id": s, "k": j, "VBM_kpoint_frac": str(vk), "CBM_kpoint_frac": str(ck),
                   "S_V_Gamma_minus_nonGamma_eV": float(X[i, 0] + 0.7 * X[i, 1]),
                   "S_C_nonGamma_minus_Gamma_eV": float(X[i, 2] - 0.8 * X[i, 3])}
            if j in {0, 11}:   # endpoints: the primitive-cell k points decide
                row.update(endpoint_canonicalized=True, primitive_VBM_kpoint_frac=str(vk),
                           primitive_CBM_kpoint_frac=str(ck),
                           VBM_kpoint_frac="[0.125, 0.125, 0.125]",
                           CBM_kpoint_frac="[0.375, 0.375, 0.375]")
            rows.append(row)
    X = np.vstack([X, 3 * rng.normal(size=(16, nf))])   # rows outside the discovery set
    sid += [f"extra_{i}" for i in range(16)]
    protocol = root / "protocol"
    (protocol / "arrays").mkdir(parents=True)
    (protocol / "features").mkdir()
    np.save(protocol / "arrays" / "X_full_clean_float64.npy", X)
    np.save(protocol / "arrays" / "sample_id_unicode.npy", np.asarray(sid))
    names = [f"d{i}" for i in range(nf)]
    (protocol / "features" / "formula_full_feature_names.json").write_text(json.dumps(names))
    pd.DataFrame({"sample_id": sid[:n_traj * per]}).to_csv(root / "samples.csv", index=False)
    pd.DataFrame(rows).to_csv(root / "kpoints.csv", index=False)
    pi_s = {"op": "add", "left": {"op": "terminal", "feature_index": 0},
            "right": {"op": "safe_ratio", "left": {"op": "terminal", "feature_index": 2},
                      "right": {"op": "spower", "power": [1, 2],
                                "arg": {"op": "terminal", "feature_index": 5}}}}
    (root / "pi_s.json").write_text(json.dumps({"ast_json": json.dumps(pi_s)}))
    np.save(root / "pi_s_scales.npy", np.abs(rng.normal(size=nf)) + 0.5)
    for edge in ("valence", "conduction"):
        pd.DataFrame({"feature": names, "importance": rng.random(nf)}).to_csv(
            root / f"prior_{edge}.csv", index=False)
    (root / "small.json").write_text(json.dumps({"profile": "small", "jobs": 1}))
    return {"protocol": protocol, "samples": root / "samples.csv", "kpoints": root / "kpoints.csv",
            "pi_s_formula": root / "pi_s.json", "pi_s_scales": root / "pi_s_scales.npy",
            "prior_valence": root / "prior_valence.csv",
            "prior_conduction": root / "prior_conduction.csv"}


@pytest.fixture(scope="module")
def edge_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("edges")
    inputs = write_edge_inputs(root / "in")
    settings = de.load_edge_settings(root / "in" / "small.json")
    record = de.run(settings, inputs, root / "straight")
    return root, inputs, settings, record


def test_targets_use_primitive_endpoints(edge_run):
    root, inputs, settings, _ = edge_run
    data = de.load_edge_data(inputs["protocol"], inputs["samples"], inputs["kpoints"], settings)
    V = data.targets["V"]
    assert V.available.all() and V.is_endpoint.sum() == 16
    assert not np.isin(V.exact_key[V.is_endpoint], ["[0.125000,0.125000,0.125000]"]).any()
    assert not V.has_margin[V.is_endpoint].any()
    assert {"NN_same", "NN_diff"} <= set(data.states.tolist())


def test_run_finishes_with_both_families(edge_run):
    root, _, _, record = edge_run
    assert record["status"] == "finalized"
    for edge in ("valence", "conduction"):
        top = pd.read_csv(root / "straight" / f"top30_{edge}.csv")
        assert len(top) > 0 and float(top.auc.max()) > 0.7
        assert set(top.family) <= set(de.FAMILIES)
    pairs = pd.read_csv(root / "straight" / "pair_audit.csv")
    assert int(pairs.is_independently_recommended_formula_pair.sum()) == 1


def test_resume_continues_exactly(edge_run):
    root, inputs, settings, _ = edge_run
    stopped = de.run(settings, inputs, root / "resumed", stop_after_generation=2)
    assert stopped["status"] == "stopped"
    de.run(settings, inputs, root / "resumed", resume=True)
    for name in ("top30_valence.csv", "top30_conduction.csv", "elite_valence.csv",
                 "components.json", "component_values.float32.npy", "pair_audit.csv"):
        assert (root / "resumed" / name).read_bytes() == (root / "straight" / name).read_bytes()


def test_replay_reproduces_top_lists(edge_run):
    root, inputs, settings, _ = edge_run
    data = de.load_edge_data(inputs["protocol"], inputs["samples"], inputs["kpoints"], settings,
                             pi_s_formula=inputs["pi_s_formula"],
                             pi_s_scales=inputs["pi_s_scales"])
    components = json.loads((root / "straight" / "components.json").read_text())
    for edge in ("valence", "conduction"):
        records, _ = de.load_edge_formulas(root / "straight" / f"top30_{edge}.csv")
        reports, _ = de.replay(records, data, components)
        assert all(r["all_equal"] for r in reports), [r for r in reports if not r["all_equal"]]


def test_rank_auc_and_threshold():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200).astype(np.int8)
    s = np.round(rng.normal(size=200) + y, 1)          # many ties
    assert de.rank_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-15)
    th, bacc, plateau = de.best_threshold(np.array([0, 0, 1, 1], np.int8),
                                          np.array([0.0, 1.0, 2.0, 3.0]), 1)
    assert (th, bacc) == (1.5, 1.0) and 0 <= plateau <= 10


def test_protected_ratio_and_signed_square():
    a, b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 1.0, -2.0, 4.0])
    eps = de.RATIO_EPSILON * float(np.quantile(np.abs(b), 0.75))
    assert np.array_equal(de.protected_ratio(a, b), a / (np.abs(b) + eps))
    X = np.array([[-3.0], [2e6]])
    assert np.array_equal(de.evaluate({"op": "square", "arg": {"op": "feature", "id": 0}},
                                      X, None, None), np.array([-9.0, 1e12]))


def test_four_states(tmp_path):
    base = {"rank": 1, "formula_id": "f", "curve_id": "c", "fold_id": [0, 1, 2, 3],
            "sample_id": ["a", "b", "c", "d"], "score": 0.0, "oof_margin": 0.0}
    pd.DataFrame({**base, "truth": [1, 0, 1, 0], "oof_prediction": [1, 0, 1, 1]}).to_csv(
        tmp_path / "v.csv", index=False)
    pd.DataFrame({**base, "truth": [1, 1, 0, 0], "oof_prediction": [1, 1, 0, 0]}).to_csv(
        tmp_path / "c.csv", index=False)
    table, summary = de.four_states(tmp_path / "v.csv", tmp_path / "c.csv")
    assert table.state.tolist() == ["COLOCATED", "V_MISMATCH_CARRIER", "C_MISMATCH_CARRIER",
                                    "DISTRIBUTED_MISMATCH"]
    assert summary["both_edges_correct"] == 3 and summary["conduction_oof_bacc"] == 1.0


@pytest.mark.skipif(not (TABLES.is_dir() and EDGE_LABELS.is_file()),
                    reason="DP descriptor tables or edge labels not present")
@pytest.mark.parametrize("edge,positives", [("valence", 67), ("conduction", 58)])
def test_dp_edge_protocol(edge, positives, tmp_path):
    """Edge labels replace the gap label only; X, folds and registries stay those of DP."""
    import hashlib
    from symbolic_regression.inputs import prepare
    from .test_inputs import PUBLISHED
    out = tmp_path / edge
    prepare("dp", TABLES / "dp", out, edge, EDGE_LABELS)
    y = np.load(out / "protocol/RF_PROTOCOL_BUNDLE/arrays/y_int8.npy")
    assert int(y.sum()) == positives and len(y) == 85
    for rel, digest in PUBLISHED["dp"].items():
        if rel.endswith(("y_int8.npy", "protocol_rows.csv")):
            continue
        assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == digest, rel
