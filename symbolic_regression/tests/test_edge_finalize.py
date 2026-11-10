"""Post-search steps of the spinel edge coordinates: tree algebra, scans and the config."""
import json
from pathlib import Path

import numpy as np

from symbolic_regression import edge_finalize as ef

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def f(i):
    return {"op": "feature", "id": i}


def test_simplify_merges_scales_and_drops_zeros():
    t = {"op": "scale", "c": 0.5, "arg": {"op": "scale", "c": -2.0, "arg": f(0)}}
    assert ef.simplify(t) == {"op": "scale", "c": -1.0, "arg": f(0)}
    assert ef.simplify({"op": "add", "left": {"op": "scale", "c": 0.0, "arg": f(1)},
                        "right": f(2)}) == f(2)
    assert ef.simplify({"op": "mul", "left": {"op": "const", "value": 0.0},
                        "right": f(3)}) == {"op": "const", "value": 0.0}


def test_prunes_replace_one_node_by_a_child():
    t = {"op": "sub", "left": {"op": "square", "arg": f(0)}, "right": f(1)}
    got = {json.dumps(z, sort_keys=True) for z in ef.prunes(t)}
    want = {json.dumps(z, sort_keys=True) for z in (
        {"op": "square", "arg": f(0)}, {"op": "scale", "c": -1.0, "arg": f(1)},
        {"op": "sub", "left": f(0), "right": f(1)})}
    assert got == want
    assert ef.complexity(t) == 2 + 0.35 * 2


def test_evaluate_constants_and_ratio():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = {"op": "add", "left": {"op": "const", "value": 1.0},
         "right": {"op": "ratio", "left": f(0), "right": f(1)}}
    eps = 1e-9 * np.quantile(np.abs(X[:, 1]), 0.75)
    assert np.allclose(ef.evaluate(t, X, None), 1.0 + X[:, 0] / (X[:, 1] + eps), rtol=0, atol=0)


def test_coefficients_text_cost_and_sum():
    terms = [{"name": "P"}, {"name": "A", "base": 3}, {"name": "B", "base": "-1/6"}]
    assert ef.coefficient_text(terms, (3.0, -1 / 6)) == "P + (3)*A + (-1/6)*B"
    assert ef.coefficient_cost((3.0,)) == 0.35 * 3 + 0.65
    v = {"P": np.array([1.0]), "A": np.array([2.0]), "B": np.array([6.0])}
    assert ef.combine(terms, v, (3.0, -1 / 6)).tolist() == [(1.0 + 3.0 * 2.0) + (-1 / 6) * 6.0]


def test_config_builds_the_published_coordinate_trees():
    """Terms and final coefficients of the config give exactly the published trees."""
    cfg = ef.load_config(CONFIGS / "spinel_edge_finalize.json")
    published = json.loads((CONFIGS / "spinel_edge_coordinates.json").read_text(encoding="utf-8"))
    final = {"V": (3.0, -1.0, -1 / 6)}   # the recommendation of the coefficient scan for V
    for c in published["coordinates"]:
        e = cfg["edges"][c["edge"]]
        coef = (final[c["edge"]] if e["final_coefficients"] == "recommended"
                else [ef.number(x) for x in e["final_coefficients"]])
        assert json.dumps(ef.coordinate_tree(e["terms"], coef)) == json.dumps(c["ast"])
    assert cfg["freeze"]["description"] == published["description"]
    assert '"_why"' not in json.dumps(cfg)
