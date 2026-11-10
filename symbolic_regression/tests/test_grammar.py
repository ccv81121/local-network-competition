import numpy as np
import pytest

from symbolic_regression import grammar as g

T = g.terminal
DOMAINS = ["POSITIVE", "SIGNED", "NONNEGATIVE", "POSITIVE"]


def test_commutative_and_associative_forms_coincide():
    a = {"op": "add", "left": {"op": "add", "left": T(2), "right": T(0)}, "right": T(1)}
    b = {"op": "add", "left": T(1), "right": {"op": "add", "left": T(0), "right": T(2)}}
    assert g.canonicalize_expression(a) == g.canonicalize_expression(b)


def test_redundant_nodes_are_removed():
    assert g.canonicalize_expression({"op": "abs", "arg": {"op": "abs", "arg": T(1)}}) == \
        {"op": "abs", "arg": T(1)}
    squared = {"op": "power", "power": [2, 1], "arg": {"op": "abs", "arg": T(1)}}
    assert g.canonicalize_expression(squared) == {"op": "power", "power": [2, 1], "arg": T(1)}
    nested = {"op": "scale", "coefficient": [2, 1],
              "arg": {"op": "scale", "coefficient": [3, 2], "arg": T(0)}}
    assert g.canonicalize_expression(nested) == {"op": "scale", "coefficient": [3, 1], "arg": T(0)}


def test_illegal_expressions_are_rejected():
    with pytest.raises(g.ExpressionError):
        g.canonicalize_expression({"op": "power", "power": [1, 2],
                                   "arg": {"op": "power", "power": [2, 1], "arg": T(0)}})
    with pytest.raises(g.ExpressionError):
        g.validate_component({"op": "power", "power": [1, 2], "arg": T(1)}, DOMAINS)
    four = {"op": "add", "left": {"op": "add", "left": T(0), "right": T(1)},
            "right": {"op": "add", "left": T(2), "right": T(3)}}
    with pytest.raises(g.ExpressionError):
        g.validate_component(four, DOMAINS)
    two_ratios = {"op": "safe_ratio", "left": T(0),
                  "right": {"op": "safe_ratio", "left": T(2), "right": T(3)}}
    with pytest.raises(g.ExpressionError):
        g.validate_component(two_ratios, DOMAINS)


def test_domains_follow_the_operators():
    assert g.infer_domain({"op": "spower", "power": [1, 2], "arg": T(1)}, DOMAINS) == "SIGNED"
    assert g.infer_domain({"op": "mul", "left": T(0), "right": T(2)}, DOMAINS) == "NONNEGATIVE"
    assert g.infer_domain({"op": "sub", "left": T(0), "right": T(3)}, DOMAINS) == "SIGNED"


def test_ratio_regularizers_of_the_two_levels():
    X = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 4.0]])
    ratio = {"op": "safe_ratio", "left": T(0), "right": T(1)}
    values, diag = g.evaluate_expression(ratio, X, epsilon=1e-6)
    eps = 1e-6 * np.quantile(np.abs(X[1]), 0.75)
    assert np.array_equal(values, X[0] / (np.abs(X[1]) + eps))
    assert diag["finite_fraction"] == 1.0
    skeleton = {"op": "safe_ratio", "left": g.component_leaf(0), "right": g.component_leaf(1)}
    out = g.evaluate_skeleton(skeleton, lambda cid: X[cid], 1e-6, {})
    assert np.array_equal(out, X[0] / (np.abs(X[1]) + 1e-6))


def test_skeleton_identities():
    a = {"op": "sub", "left": g.component_leaf(1),
         "right": {"op": "scale", "coefficient": [2, 1], "arg": g.component_leaf(2)}}
    b = {"op": "sub", "left": g.component_leaf(5),
         "right": {"op": "scale", "coefficient": [3, 1], "arg": g.component_leaf(7)}}
    assert g.topology_id(a) == g.topology_id(b)
    assert g.skeleton_id(a) != g.skeleton_id(b)
    assert g.skeleton_formula(a) == "(C[1] - (2/1)*(C[2]))"
    with pytest.raises(g.ExpressionError):
        g.check_skeleton_shape({"op": "add", "left": g.component_leaf(1),
                                "right": g.component_leaf(1)})


def test_expansion_substitutes_components():
    components = {0: T(0), 1: {"op": "mul", "left": T(2), "right": T(3)}}
    skeleton = {"op": "add", "left": g.component_leaf(1), "right": g.component_leaf(0)}
    can, expanded, complexity = g.validate_skeleton(skeleton, components.__getitem__, DOMAINS)
    assert expanded == g.canonicalize_expression(
        {"op": "add", "left": T(0), "right": {"op": "mul", "left": T(2), "right": T(3)}})
    assert complexity["raw_feature_count"] == 3
    assert complexity["expanded_operator_nodes"] == 2
    assert g.to_formula(expanded, ["a", "b", "c", "d"]) in {"(a + (c * d))", "((c * d) + a)"}
