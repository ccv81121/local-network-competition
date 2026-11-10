"""Typed expression trees used by the single-coordinate search.

Two levels share one operator set.  A component is an expression over scaled
descriptors (``terminal`` leaves); a skeleton is an expression whose leaves
are components (``component`` leaves).  This module defines what is legal at
each level, the canonical forms used for de-duplication, the evaluators and
the printed formulas.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np

SIGNED = "SIGNED"
NONNEGATIVE = "NONNEGATIVE"
POSITIVE = "POSITIVE"
DOMAINS = {SIGNED, NONNEGATIVE, POSITIVE}

POWERS = ((1, 3), (1, 2), (2, 3), (3, 4), (1, 1), (4, 3), (3, 2), (2, 1))
COEFFICIENTS = ((1, 3), (1, 2), (2, 3), (1, 1), (3, 2), (2, 1), (3, 1))

OPERATORS = (
    "terminal", "abs", "power", "spower", "scale", "add", "sub", "mul",
    "geometric", "sgeometric", "saturating", "safe_ratio",
)
UNARY = {"abs", "power", "spower", "scale"}
BINARY = {"add", "sub", "mul", "geometric", "sgeometric", "saturating", "safe_ratio"}
COMMUTATIVE = {"add", "mul", "geometric", "sgeometric"}
DIVISION_LIKE = {"saturating", "safe_ratio"}
LEAVES = {"terminal", "component"}

COMPONENT_BINARY_OPS = (
    "add", "sub", "mul", "sgeometric", "geometric", "saturating", "safe_ratio",
)
SKELETON_BINARY_OPS = ("add", "sub", "mul", "sgeometric", "saturating", "safe_ratio")

MAX_ABS_OUTPUT = 1.0e150
MAX_SKELETON_LEAVES = 8
MAX_SKELETON_DEPTH = 7


class ExpressionError(ValueError):
    """An expression violates the grammar."""


@dataclass(frozen=True)
class Limits:
    features: int
    depth: int
    nodes: int
    powers: int
    coefficients: int
    divisions: int
    repeated_features: bool


# A component is a short expression of at most three distinct descriptors.
COMPONENT_LIMITS = Limits(3, 6, 7, 3, 3, 1, False)
# Limits on a skeleton after its components have been substituted.
EXPANDED_LIMITS = Limits(24, 20, 120, 24, 24, 2, True)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_id(ast: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json_bytes(ast))


def deep_copy(ast: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(ast))


# ---------------------------------------------------------------- paths


def node_paths(ast: Mapping[str, Any]) -> list[tuple[str, ...]]:
    """All node paths in pre-order, root first."""
    out: list[tuple[str, ...]] = []

    def walk(node: Mapping[str, Any], path: tuple[str, ...]) -> None:
        out.append(path)
        op = node["op"]
        if op in UNARY:
            walk(node["arg"], path + ("arg",))
        elif op not in LEAVES:
            walk(node["left"], path + ("left",))
            walk(node["right"], path + ("right",))

    walk(ast, ())
    return out


def node_at(ast: Mapping[str, Any], path: Sequence[str]) -> Mapping[str, Any]:
    node = ast
    for key in path:
        node = node[key]
    return node


def replace_node(ast: Mapping[str, Any], path: Sequence[str],
                 replacement: Mapping[str, Any]) -> dict[str, Any]:
    if not path:
        return deep_copy(replacement)
    out = deep_copy(ast)
    node = out
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = deep_copy(replacement)
    return out


# ------------------------------------------------- descriptor-level trees


def terminal(index: int) -> dict[str, Any]:
    return {"op": "terminal", "feature_index": int(index)}


def wrap_power(ast: Mapping[str, Any], power: Sequence[int], domain: str) -> dict[str, Any]:
    if tuple(power) == (1, 1):
        return dict(ast)
    op = "power" if domain in {POSITIVE, NONNEGATIVE} else "spower"
    return {"op": op, "power": list(power), "arg": dict(ast)}


def feature_indices(ast: Mapping[str, Any]) -> list[int]:
    """Terminal occurrences in traversal order."""
    op = ast["op"]
    if op == "terminal":
        return [int(ast["feature_index"])]
    if op in UNARY:
        return feature_indices(ast["arg"])
    return feature_indices(ast["left"]) + feature_indices(ast["right"])


def _fraction_pair(value: Any, allowed: Sequence[tuple[int, int]]) -> tuple[int, int]:
    if isinstance(value, Fraction):
        f = value
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        f = Fraction(int(value[0]), int(value[1]))
    elif isinstance(value, int):
        f = Fraction(value, 1)
    elif isinstance(value, str):
        f = Fraction(value)
    else:
        raise ExpressionError(f"invalid rational value {value!r}")
    if f <= 0:
        raise ExpressionError("coefficient/power must be positive")
    pair = (f.numerator, f.denominator)
    if pair not in set(allowed):
        raise ExpressionError(f"rational value {pair} not in allowed library")
    return pair


def _flatten(ast: Mapping[str, Any], op: str) -> list[Mapping[str, Any]]:
    if ast.get("op") == op:
        return _flatten(ast["left"], op) + _flatten(ast["right"], op)
    return [ast]


def _rebuild_sorted(op: str, terms: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    ordered = sorted(terms, key=canonical_json_bytes)
    out = ordered[0]
    for term in ordered[1:]:
        out = {"op": op, "left": out, "right": term}
    return out


def canonicalize_expression(expr: Mapping[str, Any], *,
                            n_features: int | None = None) -> dict[str, Any]:
    """Canonical form: sorted associative chains, merged scales, no unit parameters."""
    if not isinstance(expr, Mapping):
        raise ExpressionError("expression must be a mapping")
    op = str(expr.get("op", ""))
    if op not in OPERATORS:
        raise ExpressionError(f"unknown operator {op!r}")
    if op == "terminal":
        idx = int(expr.get("feature_index", -1))
        if idx < 0 or (n_features is not None and idx >= n_features):
            raise ExpressionError(f"terminal feature_index out of range: {idx}")
        return {"op": "terminal", "feature_index": idx}
    if op == "abs":
        arg = canonicalize_expression(expr["arg"], n_features=n_features)
        if arg["op"] == "abs":
            return arg
        return {"op": "abs", "arg": arg}
    if op in {"power", "spower"}:
        power = _fraction_pair(expr.get("power"), POWERS)
        arg = canonicalize_expression(expr["arg"], n_features=n_features)
        if power == (1, 1):
            return arg
        if arg["op"] in {"power", "spower"}:
            raise ExpressionError("nested powers are forbidden")
        if op == "power" and power == (2, 1) and arg["op"] == "abs":
            arg = arg["arg"]
        return {"op": op, "power": list(power), "arg": arg}
    if op == "scale":
        coef = _fraction_pair(expr.get("coefficient"), COEFFICIENTS)
        arg = canonicalize_expression(expr["arg"], n_features=n_features)
        if coef == (1, 1):
            return arg
        if arg["op"] == "scale":
            f = Fraction(*coef) * Fraction(*arg["coefficient"])
            pair = (f.numerator, f.denominator)
            if pair not in set(COEFFICIENTS):
                raise ExpressionError("nested scale product leaves coefficient library")
            return {"op": "scale", "coefficient": list(pair), "arg": arg["arg"]}
        return {"op": "scale", "coefficient": list(coef), "arg": arg}
    left = canonicalize_expression(expr["left"], n_features=n_features)
    right = canonicalize_expression(expr["right"], n_features=n_features)
    if op in {"add", "mul"}:
        return dict(_rebuild_sorted(op, _flatten(left, op) + _flatten(right, op)))
    if op in {"geometric", "sgeometric"}:
        if canonical_json_bytes(right) < canonical_json_bytes(left):
            left, right = right, left
    return {"op": op, "left": left, "right": right}


def _count_nodes(ast: Mapping[str, Any]) -> int:
    if ast["op"] == "terminal":
        return 0
    if ast["op"] in UNARY:
        return 1 + _count_nodes(ast["arg"])
    return 1 + _count_nodes(ast["left"]) + _count_nodes(ast["right"])


def _depth(ast: Mapping[str, Any]) -> int:
    if ast["op"] == "terminal":
        return 1
    if ast["op"] in UNARY:
        return 1 + _depth(ast["arg"])
    return 1 + max(_depth(ast["left"]), _depth(ast["right"]))


def expression_complexity(ast: Mapping[str, Any]) -> dict[str, Any]:
    features = feature_indices(ast)
    counts = {"power_nodes": 0, "nontrivial_power_nodes": 0, "scale_nodes": 0,
              "nonunit_coefficient_count": 0, "division_like_nodes": 0,
              "special_operator_nodes": 0, "ast_token_count": 0,
              "power_complexity": 0, "coefficient_complexity": 0}

    def walk(node: Mapping[str, Any]) -> None:
        counts["ast_token_count"] += 1
        op = node["op"]
        if op in {"power", "spower"}:
            counts["power_nodes"] += 1
            pair = tuple(node["power"])
            if pair != (1, 1):
                counts["nontrivial_power_nodes"] += 1
                counts["power_complexity"] += 1 if pair in {(1, 2), (2, 1)} else 2
        elif op == "scale":
            counts["scale_nodes"] += 1
            pair = tuple(node["coefficient"])
            if pair != (1, 1):
                counts["nonunit_coefficient_count"] += 1
                counts["coefficient_complexity"] += abs(pair[0]) + pair[1] - 2
        elif op in DIVISION_LIKE:
            counts["division_like_nodes"] += 1
            counts["special_operator_nodes"] += 1
        elif op in {"geometric", "sgeometric"}:
            counts["special_operator_nodes"] += 1
        if op in UNARY:
            walk(node["arg"])
        elif op in BINARY:
            walk(node["left"])
            walk(node["right"])

    walk(ast)
    out = {"n_base_features": len(set(features)),
           "n_terminal_occurrences": len(features),
           "n_components": 1,
           "tree_depth": _depth(ast),
           "operator_nodes": _count_nodes(ast)}
    out.update(counts)
    out["complexity_tuple"] = [
        out["n_base_features"], out["n_components"], out["operator_nodes"],
        out["tree_depth"], out["nontrivial_power_nodes"],
        out["division_like_nodes"], out["power_complexity"],
        out["coefficient_complexity"],
    ]
    return out


def infer_domain(ast: Mapping[str, Any], domains: Sequence[str]) -> str:
    op = ast["op"]
    if op == "terminal":
        d = domains[int(ast["feature_index"])]
        if d not in DOMAINS:
            raise ExpressionError(f"unknown terminal domain {d}")
        return d
    if op == "abs":
        return NONNEGATIVE
    if op == "power":
        inner = infer_domain(ast["arg"], domains)
        n, d = ast["power"]
        if d != 1 and inner == SIGNED:
            raise ExpressionError("fractional power requires nonnegative input; use spower")
        if (n, d) == (2, 1) or inner in {NONNEGATIVE, POSITIVE}:
            return NONNEGATIVE if inner != POSITIVE else POSITIVE
        return inner
    if op == "spower":
        inner = infer_domain(ast["arg"], domains)
        return SIGNED if inner == SIGNED else inner
    if op == "scale":
        return infer_domain(ast["arg"], domains)
    left = infer_domain(ast["left"], domains)
    right = infer_domain(ast["right"], domains)
    nonneg = {NONNEGATIVE, POSITIVE}
    if op in {"add", "mul"}:
        return NONNEGATIVE if left in nonneg and right in nonneg else SIGNED
    if op == "sub":
        return SIGNED
    if op == "geometric":
        if left == SIGNED or right == SIGNED:
            raise ExpressionError("geometric requires nonnegative inputs")
        return NONNEGATIVE
    if op == "sgeometric":
        return SIGNED if SIGNED in {left, right} else NONNEGATIVE
    if op in DIVISION_LIKE:
        return left
    raise ExpressionError(f"cannot infer domain for {op}")


def _division_stats(ast: Mapping[str, Any]) -> tuple[int, int]:
    """Total division-like nodes and the maximum along any root-to-leaf path."""
    def walk(node: Mapping[str, Any], current: int) -> tuple[int, int]:
        op = node["op"]
        here = current + (1 if op in DIVISION_LIKE else 0)
        total = 1 if op in DIVISION_LIKE else 0
        if op in UNARY:
            t, m = walk(node["arg"], here)
            return total + t, m
        if op in BINARY:
            tl, ml = walk(node["left"], here)
            tr, mr = walk(node["right"], here)
            return total + tl + tr, max(ml, mr)
        return total, here
    return walk(ast, 0)


def validate_expression(ast: Mapping[str, Any], domains: Sequence[str],
                        limits: Limits) -> dict[str, Any]:
    """Canonicalize ``ast`` and check it against the domain rules and ``limits``."""
    can = canonicalize_expression(ast, n_features=len(domains))
    infer_domain(can, domains)
    c = expression_complexity(can)
    if (c["n_base_features"] > limits.features or c["tree_depth"] > limits.depth
            or c["operator_nodes"] > limits.nodes):
        raise ExpressionError("expression exceeds feature/depth/node limits")
    if c["nontrivial_power_nodes"] > limits.powers:
        raise ExpressionError("too many nontrivial power nodes")
    if c["nonunit_coefficient_count"] > limits.coefficients:
        raise ExpressionError("too many non-unit coefficients")
    features = feature_indices(can)
    if not limits.repeated_features and len(features) != len(set(features)):
        raise ExpressionError("repeated base feature is forbidden")
    total, per_path = _division_stats(can)
    if total > limits.divisions:
        raise ExpressionError("too many division-like nodes")
    if per_path > 1:
        raise ExpressionError("nested division-like nodes are forbidden")
    return can


def validate_component(ast: Mapping[str, Any], domains: Sequence[str]) -> dict[str, Any]:
    return validate_expression(ast, domains, COMPONENT_LIMITS)


def parameter_free(ast: Mapping[str, Any]) -> dict[str, Any]:
    """The tree with powers and coefficients replaced by a placeholder."""
    op = ast["op"]
    if op in LEAVES:
        return dict(ast)
    if op == "scale":
        return {"op": "scale", "coefficient": "PARAM", "arg": parameter_free(ast["arg"])}
    if op in {"power", "spower"}:
        return {"op": op, "power": "PARAM", "arg": parameter_free(ast["arg"])}
    if op in UNARY:
        return {"op": op, "arg": parameter_free(ast["arg"])}
    return {"op": op, "left": parameter_free(ast["left"]),
            "right": parameter_free(ast["right"])}


def to_formula(ast: Mapping[str, Any], names: Sequence[str] | None = None) -> str:
    op = ast["op"]
    if op == "terminal":
        i = int(ast["feature_index"])
        return names[i] if names is not None else f"x{i}"
    if op == "abs":
        return f"abs({to_formula(ast['arg'], names)})"
    if op in {"power", "spower"}:
        n, d = ast["power"]
        inner = to_formula(ast["arg"], names)
        return f"spow({inner},{n}/{d})" if op == "spower" else f"({inner})^({n}/{d})"
    if op == "scale":
        n, d = ast["coefficient"]
        return f"({n}/{d})*({to_formula(ast['arg'], names)})"
    a = to_formula(ast["left"], names)
    b = to_formula(ast["right"], names)
    if op == "add":
        return f"({a} + {b})"
    if op == "sub":
        return f"({a} - {b})"
    if op == "mul":
        return f"({a} * {b})"
    if op == "geometric":
        return f"sqrt(({a})*({b}))"
    if op == "sgeometric":
        return f"sgn(({a})*({b}))*sqrt(abs(({a})*({b})))"
    if op == "saturating":
        return f"({a})/(1 + abs({b}))"
    if op == "safe_ratio":
        return f"({a})/(abs({b}) + epsilon)"
    raise ExpressionError(op)


def evaluate_expression(ast: Mapping[str, Any], terminals: np.ndarray, *,
                        epsilon: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Values of a component over all samples, with numerical diagnostics.

    ``terminals`` is the scaled descriptor matrix with shape (descriptors,
    samples).  In ``safe_ratio`` the regularizer is ``epsilon`` times the
    upper quartile of the denominator magnitude.
    """
    X = np.asarray(terminals, dtype=np.float64)
    if X.ndim != 2:
        raise ExpressionError("terminal values must have shape (descriptor, sample)")
    denominators: list[np.ndarray] = []
    ratio_raw: list[np.ndarray] = []
    saturations: list[np.ndarray] = []

    def value(node: Mapping[str, Any]) -> np.ndarray:
        op = node["op"]
        if op == "terminal":
            return X[int(node["feature_index"])]
        if op in UNARY:
            a = value(node["arg"])
            if op == "abs":
                return np.abs(a)
            if op == "power":
                return np.power(a, int(node["power"][0]) / int(node["power"][1]))
            if op == "spower":
                p = int(node["power"][0]) / int(node["power"][1])
                return np.sign(a) * np.power(np.abs(a), p)
            return a * (int(node["coefficient"][0]) / int(node["coefficient"][1]))
        a = value(node["left"])
        b = value(node["right"])
        if op == "add":
            return a + b
        if op == "sub":
            return a - b
        if op == "mul":
            return a * b
        if op == "geometric":
            return np.sqrt(a * b)
        if op == "sgeometric":
            prod = a * b
            return np.sign(prod) * np.sqrt(np.abs(prod))
        if op == "saturating":
            den = 1.0 + np.abs(b)
            denominators.append(den)
            saturations.append(np.abs(b))
            return a / den
        if op == "safe_ratio":
            raw = np.abs(b)
            scale = float(np.quantile(raw, 0.75))
            eps = epsilon * (scale if scale > 0 and math.isfinite(scale) else 1.0)
            den = raw + eps
            ratio_raw.append(raw)
            denominators.append(den)
            return a / den
        raise ExpressionError(op)

    with np.errstate(all="ignore"):
        values = np.asarray(value(ast), dtype=np.float64)
    # The diagnostics run outside errstate on purpose: with warnings raised as
    # errors (as in the published runs) an expression whose diagnostics
    # overflow is rejected rather than kept.
    return values, _diagnostics(values, denominators, ratio_raw, saturations)


def _diagnostics(values: np.ndarray, denominators: list[np.ndarray],
                 ratio_raw: list[np.ndarray],
                 saturations: list[np.ndarray]) -> dict[str, Any]:
    finite = np.isfinite(values)
    abs_values = np.abs(values[finite]) if finite.any() else np.asarray([], dtype=float)
    if abs_values.size:
        q50 = float(np.quantile(abs_values, 0.50))
        q99 = float(np.quantile(abs_values, 0.99))
        k = max(1, int(math.ceil(0.01 * abs_values.size)))
        top_share = float(np.partition(abs_values, -k)[-k:].sum()
                          / max(abs_values.sum(), np.finfo(float).tiny))
    else:
        q50 = q99 = top_share = float("nan")
    near_zero = 0.0
    min_den = float("nan")
    if ratio_raw:
        raw_den = np.concatenate([np.ravel(d) for d in ratio_raw])
        ref = float(np.quantile(raw_den, 0.75))
        cutoff = 1.0e-3 * max(ref, np.finfo(float).tiny)
        near_zero = float(np.mean(raw_den <= cutoff))
    if denominators:
        den = np.concatenate([np.ravel(d) for d in denominators])
        min_den = float(np.min(den))
    if values.size and (not abs_values.size or float(np.max(abs_values)) < 1e150):
        constant = bool(not finite.all() or (
            finite.all() and (float(np.max(values)) - float(np.min(values)))
            <= 1e-14 * max(1.0, float(np.max(np.abs(values))))))
    else:
        constant = True
    if abs_values.size:
        ratio = 0.0 if q99 == 0.0 else (
            float("inf") if q50 <= np.finfo(float).tiny else float(q99 / q50))
    else:
        ratio = float("nan")
    return {
        "finite_fraction": float(np.mean(finite)),
        "constant_or_near_constant": constant,
        "max_abs": float(np.max(abs_values)) if abs_values.size else float("nan"),
        "q50_abs": q50,
        "q99_abs": q99,
        "q99_to_q50": ratio,
        "top_1pct_absolute_share": top_share,
        "extreme_fraction": float(np.mean(abs_values > 1e8)) if abs_values.size else float("nan"),
        "denominator_near_zero_fraction": near_zero,
        "minimum_denominator": min_den,
        "saturation_fraction": (float(np.mean(np.concatenate(saturations) > 10.0))
                                if saturations else 0.0),
        "overflow_flag": bool(np.any(abs_values > MAX_ABS_OUTPUT)) if abs_values.size else False,
        "underflow_flag": (bool(np.any((abs_values > 0) & (abs_values < np.finfo(float).tiny)))
                           if abs_values.size else False),
    }


# --------------------------------------------------- component-level trees


def component_leaf(component_id: int) -> dict[str, Any]:
    return {"op": "component", "component_id": int(component_id)}


def skeleton_component_ids(ast: Mapping[str, Any]) -> list[int]:
    op = ast["op"]
    if op == "component":
        return [int(ast["component_id"])]
    if op in UNARY:
        return skeleton_component_ids(ast["arg"])
    return skeleton_component_ids(ast["left"]) + skeleton_component_ids(ast["right"])


def skeleton_depth(ast: Mapping[str, Any]) -> int:
    op = ast["op"]
    if op == "component":
        return 1
    if op in UNARY:
        return 1 + skeleton_depth(ast["arg"])
    return 1 + max(skeleton_depth(ast["left"]), skeleton_depth(ast["right"]))


def skeleton_operator_nodes(ast: Mapping[str, Any]) -> int:
    op = ast["op"]
    if op == "component":
        return 0
    if op in UNARY:
        return 1 + skeleton_operator_nodes(ast["arg"])
    return 1 + skeleton_operator_nodes(ast["left"]) + skeleton_operator_nodes(ast["right"])


def skeleton_division_stats(ast: Mapping[str, Any]) -> tuple[int, int]:
    def walk(node: Mapping[str, Any], current: int) -> tuple[int, int]:
        op = node["op"]
        here = current + (1 if op in DIVISION_LIKE else 0)
        total = 1 if op in DIVISION_LIKE else 0
        if op == "component":
            return total, here
        if op in UNARY:
            t, m = walk(node["arg"], here)
            return total + t, m
        tl, ml = walk(node["left"], here)
        tr, mr = walk(node["right"], here)
        return total + tl + tr, max(ml, mr)
    return walk(ast, 0)


def canonicalize_skeleton(ast: Mapping[str, Any]) -> dict[str, Any]:
    op = str(ast.get("op", ""))
    if op == "component":
        cid = int(ast.get("component_id", -1))
        if cid < 0:
            raise ExpressionError("negative component_id")
        return {"op": "component", "component_id": cid}
    if op == "abs":
        arg = canonicalize_skeleton(ast["arg"])
        if arg["op"] == "abs":
            return arg
        return {"op": "abs", "arg": arg}
    if op in {"power", "spower"}:
        pair = tuple(map(int, ast["power"]))
        if pair not in POWERS:
            raise ExpressionError(f"power outside library: {pair}")
        arg = canonicalize_skeleton(ast["arg"])
        if pair == (1, 1):
            return arg
        if arg["op"] in {"power", "spower"}:
            raise ExpressionError("nested skeleton powers")
        return {"op": op, "power": list(pair), "arg": arg}
    if op == "scale":
        pair = tuple(map(int, ast["coefficient"]))
        if pair not in COEFFICIENTS:
            raise ExpressionError(f"coefficient outside library: {pair}")
        arg = canonicalize_skeleton(ast["arg"])
        if pair == (1, 1):
            return arg
        if arg["op"] == "scale":
            raise ExpressionError("nested skeleton scales")
        return {"op": "scale", "coefficient": list(pair), "arg": arg}
    if op not in SKELETON_BINARY_OPS:
        raise ExpressionError(f"unknown skeleton operator: {op}")
    left = canonicalize_skeleton(ast["left"])
    right = canonicalize_skeleton(ast["right"])
    if op in COMMUTATIVE and canonical_json_bytes(right) < canonical_json_bytes(left):
        left, right = right, left
    return {"op": op, "left": left, "right": right}


def check_skeleton_shape(ast: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical skeleton satisfying the leaf, depth and division rules."""
    can = canonicalize_skeleton(ast)
    ids = skeleton_component_ids(can)
    if len(ids) < 1 or len(ids) > MAX_SKELETON_LEAVES:
        raise ExpressionError(f"component leaf count {len(ids)} outside 1..8")
    if len(ids) != len(set(ids)):
        raise ExpressionError("repeated component leaf forbidden")
    if skeleton_depth(can) > MAX_SKELETON_DEPTH:
        raise ExpressionError("skeleton depth exceeds 7")
    total, per_path = skeleton_division_stats(can)
    if total > 2 or per_path > 1:
        raise ExpressionError("division-like path rule violated")
    return can


def expand_skeleton(ast: Mapping[str, Any],
                    component_ast: Callable[[int], Mapping[str, Any]]) -> dict[str, Any]:
    """Substitute component expressions for the component leaves."""
    op = ast["op"]
    if op == "component":
        return component_ast(int(ast["component_id"]))
    if op in UNARY:
        out = {"op": op, "arg": expand_skeleton(ast["arg"], component_ast)}
        if op == "scale":
            out["coefficient"] = list(ast["coefficient"])
        if op in {"power", "spower"}:
            out["power"] = list(ast["power"])
        return canonicalize_expression(out)
    return canonicalize_expression({"op": op,
                                    "left": expand_skeleton(ast["left"], component_ast),
                                    "right": expand_skeleton(ast["right"], component_ast)})


def validate_skeleton(ast: Mapping[str, Any],
                      component_ast: Callable[[int], Mapping[str, Any]],
                      domains: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any],
                                                       dict[str, Any]]:
    """Canonical skeleton, its expanded expression and their complexity."""
    can = check_skeleton_shape(ast)
    ids = skeleton_component_ids(can)
    total, per_path = skeleton_division_stats(can)
    expanded = validate_expression(expand_skeleton(can, component_ast), domains,
                                   EXPANDED_LIMITS)
    full = expression_complexity(expanded)
    complexity = {
        "component_leaf_count": len(ids),
        "component_level_operator_nodes": skeleton_operator_nodes(can),
        "component_level_tree_depth": skeleton_depth(can),
        "component_division_nodes": total,
        "component_division_path_max": per_path,
        "distinct_component_count": len(set(ids)),
        "raw_feature_count": int(full.get("n_base_features", 0)),
        "expanded_operator_nodes": int(full.get("operator_nodes", 0)),
        "expanded_tree_depth": int(full.get("tree_depth", 0)),
        "expanded_nontrivial_power_nodes": int(full.get("nontrivial_power_nodes", 0)),
        "expanded_nonunit_coefficient_count": int(full.get("nonunit_coefficient_count", 0)),
    }
    return can, expanded, complexity


def skeleton_id(ast: Mapping[str, Any]) -> str:
    """Identity of a skeleton up to canonical form (component-level formula id)."""
    return tree_id(canonicalize_skeleton(ast))


def topology_id(ast: Mapping[str, Any]) -> str:
    """Identity of the operator tree, ignoring which components and parameters it uses."""
    def shape(node: Mapping[str, Any]) -> Mapping[str, Any]:
        op = node["op"]
        if op == "component":
            return {"op": "component"}
        if op == "scale":
            return {"op": "scale", "coefficient": "PARAM", "arg": shape(node["arg"])}
        if op in {"power", "spower"}:
            return {"op": op, "power": "PARAM", "arg": shape(node["arg"])}
        if op in UNARY:
            return {"op": op, "arg": shape(node["arg"])}
        return {"op": op, "left": shape(node["left"]), "right": shape(node["right"])}
    return tree_id(shape(canonicalize_skeleton(ast)))


def skeleton_formula(ast: Mapping[str, Any]) -> str:
    op = ast["op"]
    if op == "component":
        return f"C[{int(ast['component_id'])}]"
    if op == "abs":
        return f"abs({skeleton_formula(ast['arg'])})"
    if op in {"power", "spower"}:
        n, d = ast["power"]
        inner = skeleton_formula(ast["arg"])
        return f"spow({inner},{n}/{d})" if op == "spower" else f"({inner})^({n}/{d})"
    if op == "scale":
        n, d = ast["coefficient"]
        return f"({n}/{d})*({skeleton_formula(ast['arg'])})"
    a = skeleton_formula(ast["left"])
    b = skeleton_formula(ast["right"])
    if op == "add":
        return f"({a} + {b})"
    if op == "sub":
        return f"({a} - {b})"
    if op == "mul":
        return f"({a} * {b})"
    if op == "sgeometric":
        return f"sgeom({a},{b})"
    if op == "saturating":
        return f"({a})/(1 + abs({b}))"
    if op == "safe_ratio":
        return f"({a})/(abs({b}) + epsilon)"
    raise ExpressionError(op)


def evaluate_skeleton(node: Mapping[str, Any], component_values: Callable[[int], np.ndarray],
                      epsilon: float,
                      cache: MutableMapping[bytes, np.ndarray]) -> np.ndarray:
    """Values of a skeleton from the stored component values.

    Unlike :func:`evaluate_expression`, ``safe_ratio`` here adds the plain
    ``epsilon`` and ``power`` acts on the magnitude; this is the arithmetic of
    the published runs.  ``cache`` shares sub-trees within one batch.
    """
    key = canonical_json_bytes(node)
    if key in cache:
        return cache[key]
    op = node["op"]
    with np.errstate(all="ignore"):
        if op == "component":
            out = np.asarray(component_values(int(node["component_id"])), dtype=np.float64)
        elif op == "abs":
            out = np.abs(evaluate_skeleton(node["arg"], component_values, epsilon, cache))
        elif op in {"power", "spower"}:
            x = evaluate_skeleton(node["arg"], component_values, epsilon, cache)
            p = float(node["power"][0]) / float(node["power"][1])
            out = np.power(np.abs(x), p)
            if op == "spower":
                out = np.sign(x) * out
        elif op == "scale":
            c = float(node["coefficient"][0]) / float(node["coefficient"][1])
            out = c * evaluate_skeleton(node["arg"], component_values, epsilon, cache)
        else:
            a = evaluate_skeleton(node["left"], component_values, epsilon, cache)
            b = evaluate_skeleton(node["right"], component_values, epsilon, cache)
            if op == "add":
                out = a + b
            elif op == "sub":
                out = a - b
            elif op == "mul":
                out = a * b
            elif op == "sgeometric":
                out = np.sign(a * b) * np.sqrt(np.abs(a * b))
            elif op == "saturating":
                out = a / (1.0 + np.abs(b))
            elif op == "safe_ratio":
                out = a / (np.abs(b) + epsilon)
            else:
                raise ExpressionError(op)
    out = np.asarray(out, dtype=np.float64)
    cache[key] = out
    return out
