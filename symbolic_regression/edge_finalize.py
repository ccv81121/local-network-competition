"""From a two-edge search to the fixed spinel edge coordinates Π_V and Π_C.

Four steps follow the search (``edges run``); none of them searches again.

1. Selection.  From the hall of fame of each edge keep the numerically safe
   formulas (no concentration penalty, q99/q50 ≤ 10⁴), one per expanded
   formula, and name four of them: the AUC champion, the balanced
   recommendation, the simplest near-optimal formula and the margin champion.
2. Pruning.  One seed formula per edge (chosen by the rule in the config) is
   written out over descriptors, with Π_S kept as one terminal.  Every
   single deletion of a node is tried; a pruned formula survives if it stays
   safe, loses at most 0.015 AUC and 0.08 margin agreement against the seed
   and passes the path gate.  A beam of the simplest survivors is pruned
   again, for six rounds.
3. Coefficients.  For a fixed structure (a sum of terms taken from the
   pruning set) the free coefficients are scanned on a grid of simple
   fractions and on a dense grid around the best of them; the simplest
   fractions within 0.003 AUC of the best are recommended.
4. Freeze.  The final coefficients (the recommendation, unless the config
   records a different choice by the authors) give the coordinates.  They are
   evaluated as published: protected-ratio offsets 10⁻⁶ × the development
   upper quartile of each denominator, Π_S with its offsets fitted on the
   development states, margin agreement with tie-averaged ranks.  The result
   is a coordinate file that ``edges replay`` re-evaluates.

Steps 1–3 evaluate in double precision with the ratio offsets of the search
(10⁻⁹) and the development-fitted Π_S, as the published post-search runs did.
Every choice the authors made by hand sits in the config file, with its
reason.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import dual_edge as de

UNARY = ("abs", "ssqrt", "square")
SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "ratio": "/p", "sgeom": "sgeom"}
SEARCH_ONLY = ("star_eta2", "star_group_n")   # valley diagnostics of the search, not used here


def number(x: Any) -> float:
    """A coefficient from the config: a number or a fraction such as "-1/6"."""
    return float(Fraction(x)) if isinstance(x, str) else float(x)


def metrics(score: np.ndarray, target: de.EdgeTarget, complexity: float) -> dict[str, Any]:
    m = de.candidate_metrics(np.asarray(score, dtype=float), target, complexity)
    for k in SEARCH_ONLY:
        m.pop(k, None)
    return m


# ------------------------------------------------------- descriptor-level trees


def evaluate(node: Mapping[str, Any], X: np.ndarray, pi_s: np.ndarray,
             epsilon: float = de.RATIO_EPSILON) -> np.ndarray:
    """Values of a tree over descriptors, Π_S ("rc8") and constants."""
    op = node["op"]
    if op == "const":
        return np.full(X.shape[0], float(node["value"]))
    if op == "feature":
        return X[:, int(node["id"])]
    if op == "rc8":
        return pi_s
    if op == "scale":
        return float(node["c"]) * evaluate(node["arg"], X, pi_s, epsilon)
    if op in UNARY:
        a = evaluate(node["arg"], X, pi_s, epsilon)
        if op == "abs":
            return np.abs(a)
        if op == "ssqrt":
            return np.sign(a) * np.sqrt(np.abs(a))
        return np.sign(a) * np.minimum(np.abs(a), 1e6) ** 2
    a = evaluate(node["left"], X, pi_s, epsilon)
    b = evaluate(node["right"], X, pi_s, epsilon)
    with np.errstate(all="ignore"):
        if op == "add":
            return a + b
        if op == "sub":
            return a - b
        if op == "mul":
            return a * b
        if op == "ratio":
            return de.protected_ratio(a, b, epsilon)
        if op == "sgeom":
            return np.sign(a * b) * np.sqrt(np.abs(a * b))
    raise ValueError(f"unknown operator {op}")


def expand(node: Mapping[str, Any], components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Replace component leaves by their trees; the Π_S component stays one leaf."""
    op = node["op"]
    if op == "component":
        a = components[int(node["id"])]
        return {"op": "rc8"} if a.get("op") == "rc8" else expand(a, components)
    if op in {"feature", "rc8", "const"}:
        return copy.deepcopy(node)
    z = copy.deepcopy(node)
    if op == "scale" or op in UNARY:
        z["arg"] = expand(node["arg"], components)
    else:
        z["left"], z["right"] = expand(node["left"], components), expand(node["right"], components)
    return z


def simplify(n: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested scales, fold constants, drop additive zeros and unit scales."""
    op = n["op"]
    if op in {"feature", "rc8", "const"}:
        return copy.deepcopy(n)
    if op == "scale":
        a, c = simplify(n["arg"]), float(n["c"])
        if abs(c) < 1e-14:
            return {"op": "const", "value": 0.0}
        if abs(c - 1) < 1e-14:
            return a
        if a.get("op") == "scale":
            return simplify({"op": "scale", "c": c * float(a["c"]), "arg": a["arg"]})
        if a.get("op") == "const":
            return {"op": "const", "value": c * float(a["value"])}
        return {"op": "scale", "c": c, "arg": a}
    if op in UNARY:
        a = simplify(n["arg"])
        if a.get("op") == "const":
            x = float(a["value"])
            v = (abs(x) if op == "abs" else math.copysign(math.sqrt(abs(x)), x) if op == "ssqrt"
                 else math.copysign(min(abs(x), 1e6) ** 2, x))
            return {"op": "const", "value": v}
        return {"op": op, "arg": a}
    a, b = simplify(n["left"]), simplify(n["right"])
    if a.get("op") == "const" and b.get("op") == "const":
        x, y = float(a["value"]), float(b["value"])
        if op == "add":
            v = x + y
        elif op == "sub":
            v = x - y
        elif op == "mul":
            v = x * y
        elif op == "ratio":
            v = x / (abs(y) + de.RATIO_EPSILON)
        else:
            v = math.copysign(math.sqrt(abs(x * y)), x * y) if x * y else 0.0
        return {"op": "const", "value": v}
    az = a.get("op") == "const" and abs(float(a["value"])) < 1e-14
    bz = b.get("op") == "const" and abs(float(b["value"])) < 1e-14
    if op == "add" and (az or bz):
        return b if az else a
    if op == "sub":
        if bz:
            return a
        if az:
            return simplify({"op": "scale", "c": -1.0, "arg": b})
    if op == "mul" and (az or bz):
        return {"op": "const", "value": 0.0}
    return {"op": op, "left": a, "right": b}


def stats(n: Mapping[str, Any]) -> dict[str, Any]:
    op = n["op"]
    if op in {"feature", "rc8", "const"}:
        return {"ops": 0, "leaves": 1, "ratios": 0, "depth": 1,
                "features": {int(n["id"])} if op == "feature" else set()}
    if op == "scale" or op in UNARY:
        q = stats(n["arg"])
        return {**q, "ops": q["ops"] + 1, "depth": q["depth"] + 1}
    a, b = stats(n["left"]), stats(n["right"])
    return {"ops": a["ops"] + b["ops"] + 1, "leaves": a["leaves"] + b["leaves"],
            "ratios": a["ratios"] + b["ratios"] + (op == "ratio"),
            "depth": max(a["depth"], b["depth"]) + 1, "features": a["features"] | b["features"]}


def complexity(n: Mapping[str, Any]) -> float:
    """Operators + 0.35 x leaves + 0.8 x protected ratios (the search's measure)."""
    q = stats(n)
    return float(q["ops"] + 0.35 * q["leaves"] + 0.8 * q["ratios"])


def text(n: Mapping[str, Any], names: Sequence[str]) -> str:
    op = n["op"]
    if op == "feature":
        return names[int(n["id"])]
    if op == "rc8":
        return "Pi_S"
    if op == "const":
        return f"{float(n['value']):.6g}"
    if op == "scale":
        return f"({float(n['c']):.6g})*({text(n['arg'], names)})"
    if op in UNARY:
        return f"{op}({text(n['arg'], names)})"
    return f"({text(n['left'], names)} {SYMBOLS[op]} {text(n['right'], names)})"


def tree_json(n: Mapping[str, Any]) -> str:
    return json.dumps(n, separators=(",", ":"))


def paths(n: Mapping[str, Any], path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """All node paths in pre-order, root first."""
    op = n["op"]
    if op == "scale" or op in UNARY:
        return [path] + paths(n["arg"], path + ("arg",))
    if op in {"feature", "rc8", "const"}:
        return [path]
    return [path] + paths(n["left"], path + ("left",)) + paths(n["right"], path + ("right",))


def prunes(n: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every tree with one node replaced by one of its children (a - b also by a or -b)."""
    out = []
    for p in paths(n):
        x = n
        for k in p:
            x = x[k]
        op = x["op"]
        if op == "scale" or op in UNARY:
            repl = [x["arg"]]
        elif op in {"add", "mul", "ratio", "sgeom"}:
            repl = [x["left"], x["right"]]
        elif op == "sub":
            repl = [x["left"], {"op": "scale", "c": -1.0, "arg": x["right"]}]
        else:
            repl = []
        for r in repl:
            out.append(simplify(de.replace_at(n, p, copy.deepcopy(r))))
    return list({de.tree_hash(z): z for z in out}.values())


# ------------------------------------------------------------ 1. selection


def _bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _col(df: pd.DataFrame, c: str, default: float) -> pd.Series:
    if c not in df:
        return pd.Series(default, index=df.index, dtype=float)
    x = pd.to_numeric(df[c], errors="coerce")
    return x.where(np.isfinite(x), default)


def one_per_formula(df: pd.DataFrame) -> pd.DataFrame:
    """One row per expanded formula text: the highest AUC, then margin, then simplest."""
    x = df.copy()
    x["_key"] = x["formula"].astype(str).str.replace(r"\s+", "", regex=True)
    x = x.sort_values(["auc", "margin_spearman", "complexity"], ascending=[False, False, True],
                      kind="mergesort")
    return x.drop_duplicates("_key", keep="first").drop(columns=["_key"]).reset_index(drop=True)


def balanced_score(df: pd.DataFrame) -> pd.Series:
    """0.67 AUC + 0.23 (margin agreement + 1)/2 + 0.10/(1 + 0.035 complexity)."""
    margin = (_col(df, "margin_spearman", 0.0) + 1.0) / 2.0
    simplicity = 1.0 / (1.0 + 0.035 * np.maximum(0.0, _col(df, "complexity", 100.0)))
    return 0.67 * _col(df, "auc", 0.5) + 0.23 * margin + 0.10 * simplicity


def select(elite: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = one_per_formula(elite)
    ok = (_bool(raw["safe"]) & (_col(raw, "physical_penalty", 1.0) <= cfg["max_penalty"])
          & (_col(raw, "q99_q50", np.inf) <= cfg["max_q99_q50"])
          & np.isfinite(_col(raw, "auc", -np.inf)) & raw["ast_json"].notna())
    safe = raw.loc[ok].copy()
    safe["postselect_hard_safe"] = True
    safe = one_per_formula(safe)
    if safe.empty:
        raise RuntimeError("no numerically safe formula")
    safe["postselect_score"] = balanced_score(safe)
    best = float(safe["auc"].max())
    gate = max(0.70, best - float(cfg["auc_delta"]))
    near = safe[_col(safe, "auc", 0.5) >= gate].copy()
    npath = pd.to_numeric(near.get("trajectory_auc_n", 0), errors="coerce").fillna(0).astype(int)
    path = _col(near, "trajectory_auc", 0.5)
    used, relaxed, warning = float(cfg["trajectory_gate"]), False, ""
    eligible = near[(npath < 3) | (path >= used)].copy()
    if eligible.empty:   # relax the path gate, never the numerical safety
        relaxed, used = True, float(cfg["trajectory_relaxed"])
        eligible = near[(npath < 3) | (path >= used)].copy()
    if eligible.empty:
        eligible, warning = near.copy(), "NO_SAFE_NEAR_OPTIMAL_FORMULA_PASSES_TRAJECTORY_GATE"
    eligible["postselect_recommendation_eligible"] = True
    eligible["postselect_auc_gate"] = gate
    eligible["postselect_trajectory_gate"] = used
    eligible["postselect_trajectory_gate_relaxed"] = relaxed
    eligible["postselect_warning"] = warning
    eligible = eligible.sort_values(["postselect_score", "auc", "margin_spearman", "complexity"],
                                    ascending=[False, False, False, True], kind="mergesort")
    recommended = eligible.iloc[0].copy()
    champion = safe.sort_values(["auc", "margin_spearman", "complexity"],
                                ascending=[False, False, True], kind="mergesort").iloc[0].copy()
    simple = near.sort_values(["complexity", "raw_feature_count", "auc", "margin_spearman"],
                              ascending=[True, True, False, False], kind="mergesort").iloc[0].copy()
    margin = safe.sort_values(["margin_spearman", "auc", "complexity"],
                              ascending=[False, False, True], kind="mergesort").iloc[0].copy()
    key = pd.DataFrame([{**champion.to_dict(), "postselect_role": "SAFE_AUC_CHAMPION"},
                        {**recommended.to_dict(), "postselect_role": "BALANCED_RECOMMENDED"},
                        {**simple.to_dict(), "postselect_role": "SIMPLE_NEAR_OPTIMAL"},
                        {**margin.to_dict(), "postselect_role": "SAFE_MARGIN_CHAMPION"}]
                       ).drop_duplicates(["formula_id", "postselect_role"])
    safe = safe.sort_values(["auc", "postselect_score"], ascending=[False, False],
                            kind="mergesort").reset_index(drop=True)
    return {"pool": safe, "key": key,
            "summary": {"formulas": len(raw), "safe": len(safe), "best_safe_auc": best,
                        "near_optimal_auc_gate": gate, "near_optimal": len(near),
                        "trajectory_gate": used, "trajectory_gate_relaxed": relaxed,
                        "eligible": len(eligible), "warning": warning}}


# --------------------------------------------------------------- 2. pruning


def choose_seed(sel: Mapping[str, Any], rule: Mapping[str, Any], gate: float) -> pd.Series:
    if "key_formula" in rule:
        key = sel["key"]
        return key[key.postselect_role.astype(str).eq(rule["key_formula"])].iloc[0]
    pool = sel["pool"]
    p = pool[pool.postselect_hard_safe.astype(bool) & pool.family.astype(str).eq(rule["family"])
             & (pool.trajectory_auc.fillna(0.5) >= gate)].copy()
    if not len(p):
        p = pool[pool.postselect_hard_safe.astype(bool)].copy()
    p = p[p.auc >= float(p.auc.max()) - float(rule["within_auc"])].copy()
    p = p.sort_values(["raw_feature_count", "complexity", "margin_spearman", "auc"],
                      ascending=[True, True, False, False])
    return p.iloc[0]


def passes(m: Mapping[str, Any], seed: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    if not m.get("safe", False) or float(m.get("physical_penalty", 1)) > 0:
        return False
    if float(m.get("q99_q50", float("inf"))) > cfg["max_q99_q50"]:
        return False
    if float(m["auc"]) < float(seed["auc"]) - cfg["max_auc_drop"]:
        return False
    sr, mr = float(seed.get("margin_spearman", np.nan)), float(m.get("margin_spearman", np.nan))
    if math.isfinite(sr) and math.isfinite(mr) and mr < sr - cfg["max_margin_drop"]:
        return False
    if int(m.get("trajectory_auc_n", 0)) >= 3 and float(m.get("trajectory_auc", 0.5)) < cfg["trajectory_gate"]:
        return False
    return True


def prune(seed: pd.Series, target: de.EdgeTarget, X: np.ndarray, pi_s: np.ndarray,
          components: Sequence[Mapping[str, Any]], names: Sequence[str],
          cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """All accepted prunings of one seed and the recommended one."""
    start = simplify(expand(json.loads(seed.ast_json), components))
    seed_m = metrics(evaluate(start, X, pi_s), target, complexity(start))
    h0 = de.tree_hash(start)
    seed_m.update(formula_id=h0, source_formula_id=str(seed.formula_id), stage="SEED_REPLAY",
                  formula=text(start, names), ast_json=tree_json(start),
                  raw_feature_count=len(stats(start)["features"]))
    accepted = {h0: dict(seed_m)}
    beam = [start]
    for rnd in range(1, int(cfg["rounds"]) + 1):
        cand: dict[str, dict[str, Any]] = {}
        for n in beam:
            for z in prunes(n):
                cand.setdefault(de.tree_hash(z), z)
        new = []
        for h, z in cand.items():
            if h in accepted:
                continue
            m = metrics(evaluate(z, X, pi_s), target, complexity(z))
            if passes(m, seed_m, cfg):
                q = stats(z)
                accepted[h] = dict(m, formula_id=h, stage=f"PRUNE_R{rnd}", formula=text(z, names),
                                   ast_json=tree_json(z), raw_feature_count=len(q["features"]),
                                   expanded_operator_nodes=q["ops"],
                                   expanded_ratio_nodes=q["ratios"], expanded_depth=q["depth"])
                new.append(z)
        if not new:
            break
        # simplest first; AUC and margin agreement break ties
        new.sort(key=lambda z: (complexity(z), -accepted[de.tree_hash(z)]["auc"],
                                -accepted[de.tree_hash(z)].get("margin_spearman", 0)))
        beam = new[:int(cfg["beam"])]
    df = pd.DataFrame(list(accepted.values()))
    df["auc_drop_vs_seed"] = float(seed_m["auc"]) - df.auc
    df["margin_drop_vs_seed"] = float(seed_m.get("margin_spearman", np.nan)) - df.margin_spearman
    df = df.sort_values(["complexity", "auc", "margin_spearman"],
                        ascending=[True, False, False]).reset_index(drop=True)
    # recommended: within one complexity unit of the simplest, best 0.72 AUC + 0.28 margin
    short = df[df.complexity <= float(df.complexity.min()) + 1.0].copy()
    short["simp_recommend_score"] = 0.72 * short.auc + 0.28 * ((short.margin_spearman.fillna(0) + 1) / 2)
    best = short.sort_values(["simp_recommend_score", "complexity"], ascending=[False, True]).iloc[0]
    return df, best


# ---------------------------------------------------------- 3. coefficients


def free_terms(terms: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [t for t in terms if "base" in t]


def combine(terms: Sequence[Mapping[str, Any]], values: Mapping[str, np.ndarray],
            coef: Sequence[float]) -> np.ndarray:
    """Left-to-right sum; free terms carry their coefficient, fixed terms enter as they are."""
    it = iter(coef)
    s = None
    for t in terms:
        v = values[t["name"]] if "base" not in t else next(it) * values[t["name"]]
        s = v if s is None else s + v
    return s


def fraction(x: float, max_den: int = 24) -> Fraction:
    return Fraction(float(x)).limit_denominator(max_den)


def coefficient_cost(coef: Sequence[float]) -> float:
    """0.35 |numerator| + 0.65 denominator per coefficient; non-fractions cost 8 + 20 x error."""
    cost = 0.0
    for x in coef:
        f = fraction(x)
        err = abs(float(f) - float(x))
        cost += 0.35 * abs(f.numerator) + 0.65 * f.denominator + (0 if err < 1e-10 else 8 + 20 * err)
    return float(cost)


def coefficient_text(terms: Sequence[Mapping[str, Any]], coef: Sequence[float]) -> str:
    it = iter(coef)
    parts = []
    for t in terms:
        if "base" not in t:
            parts.append(t["name"])
            continue
        f = fraction(next(it))
        parts.append(f"({f.numerator if f.denominator == 1 else f'{f.numerator}/{f.denominator}'})"
                     f"*{t['name']}")
    return " + ".join(parts)


def scan(edge, terms, values, cands, target, stage, cfg) -> pd.DataFrame:
    letters = "abcdefgh"
    names = [f"{letters[i]}_{t['name']}" for i, t in enumerate(free_terms(terms))]
    rows = []
    for co in cands:
        m = metrics(combine(terms, values, co), target, 0.0)
        if not m.get("safe", False):
            continue
        ok = (float(m["physical_penalty"]) == 0.0 and float(m["q99_q50"]) <= cfg["max_q99_q50"]
              and (int(m["trajectory_auc_n"]) < 3 or float(m["trajectory_auc"]) >= cfg["trajectory_gate"]))
        rows.append({**m, "edge": edge, "stage": stage, "eligible": bool(ok), "coefficient_cost": coefficient_cost(co),
                     "formula": coefficient_text(terms, co), **dict(zip(names, co))})
    return pd.DataFrame(rows)


def roles(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Best AUC, best margin agreement near it, and the simplest fractions near it."""
    q = df[df.eligible.astype(bool)].copy()
    top = float(q.auc.max())
    perf = q.sort_values(["auc", "margin_spearman", "coefficient_cost"], ascending=[False, False, True]).iloc[0]
    near = q[q.auc >= top - cfg["near_auc"][0]].copy()
    bal = near.sort_values(["margin_spearman", "auc", "coefficient_cost"], ascending=[False, False, True]).iloc[0]
    rat = q[q.stage.eq("RATIONAL")].copy()
    rr = rat[rat.auc >= top - cfg["near_auc"][0]].copy()
    if not len(rr):
        rr = rat[rat.auc >= top - cfg["near_auc"][1]].copy()
    if not len(rr):
        rr = rat[rat.auc >= float(rat.auc.max()) - 0.002].copy()
    if math.isfinite(float(bal.margin_spearman)):   # margin must not collapse
        r2 = rr[rr.margin_spearman >= float(bal.margin_spearman) - cfg["max_margin_loss"]]
        if len(r2):
            rr = r2
    simple = rr.sort_values(["coefficient_cost", "auc", "margin_spearman"], ascending=[True, False, False]).iloc[0]
    out = []
    for role, r in (("PERFORMANCE_CHAMPION", perf), ("BALANCED_CHAMPION", bal),
                    ("SIMPLE_RATIONAL_RECOMMENDED", simple)):
        d = r.to_dict()
        d["role"] = role
        d["auc_gap_to_performance"] = float(perf.auc) - float(r.auc)
        out.append(d)
    return pd.DataFrame(out)


def refine(edge: str, ecfg: Mapping[str, Any], cfg: Mapping[str, Any], pruned: pd.DataFrame,
           data: de.EdgeData) -> dict[str, Any]:
    target, terms = data.targets[edge], ecfg["terms"]
    values = {t["name"]: evaluate(de.resolve_names(t["ast"], data.names), data.X, data.pi_s)
              for t in terms}
    # The structure is a member of the pruning set; its base coefficients must replay it.
    st = ecfg["structure"]
    q = pruned[pruned.safe.astype(bool) & (pruned.physical_penalty == 0)
               & (pruned.raw_feature_count == st["descriptors"]) & (pruned.auc >= st["min_auc"])]
    member = q.sort_values(["raw_feature_count", "auc"], ascending=[True, False]).iloc[0]
    base = [number(t["base"]) for t in free_terms(terms)]
    bm = metrics(combine(terms, values, base), target, 0.0)
    y = target.y[target.available]
    quantum = 1.0 / max(1, int((y == 1).sum()) * int((y == 0).sum()))
    if (abs(float(bm["auc"]) - float(member.auc)) > 2.1 * quantum
            or abs(float(bm["margin_spearman"]) - float(member.margin_spearman)) > 1e-6):
        raise ValueError(f"{edge}: the terms do not reproduce the pruning member {member.formula}")
    grids = [[number(x) for x in t["grid"]] for t in free_terms(terms)]
    rational = scan(edge, terms, values, list(itertools.product(*grids)), target, "RATIONAL", cfg)
    q = rational[rational.eligible.astype(bool)].copy()
    q["seed_score"] = 0.78 * q.auc + 0.22 * ((q.margin_spearman.fillna(0) + 1) / 2)
    letters = "abcdefgh"
    cols = [f"{letters[i]}_{t['name']}" for i, t in enumerate(free_terms(terms))]
    seeds = q.sort_values(["seed_score", "coefficient_cost"], ascending=[False, True]).head(cfg["local_seeds"])
    steps = [np.arange(*t["local_step"]) for t in free_terms(terms)]
    local = sorted({tuple(round(s + d, 8) for s, d in zip(seed, delta))
                    for seed in seeds[cols].itertuples(index=False, name=None)
                    for delta in itertools.product(*steps)})
    dense = scan(edge, terms, values, [tuple(map(float, z)) for z in local], target, "LOCAL_DENSE", cfg)
    allc = pd.concat([rational, dense], ignore_index=True).drop_duplicates(subset=["formula"]).reset_index(drop=True)
    r = roles(allc, cfg)
    rec = r[r.role.eq("SIMPLE_RATIONAL_RECOMMENDED")].iloc[0]
    return {"member": member, "baseline": {**bm, "formula": coefficient_text(terms, base)},
            "candidates": allc, "rational": rational, "roles": r,
            "recommended": [float(rec[c]) for c in cols]}


# --------------------------------------------------------------- 4. freeze


def coordinate_tree(terms: Sequence[Mapping[str, Any]], coef: Sequence[float]) -> dict[str, Any]:
    it = iter(coef)
    tree = None
    for t in terms:
        node = (copy.deepcopy(t["ast"]) if "base" not in t
                else {"op": "scale", "c": float(next(it)), "arg": copy.deepcopy(t["ast"])})
        tree = node if tree is None else {"op": "add", "left": tree, "right": node}
    return tree


def coordinate_metrics(tree: Mapping[str, Any], data: de.EdgeData, edge: str,
                       epsilon: float) -> dict[str, Any]:
    """Metrics stored with a published coordinate (margin agreement with tie-averaged ranks)."""
    target = data.targets[edge]
    s = evaluate(de.resolve_names(tree, data.names), data.X, data.pi_s, epsilon)
    m = de.candidate_metrics(s, target, 0.0)
    mm = target.has_margin & target.available
    ra = pd.Series(s[mm] * m["orientation"]).rank(method="average").to_numpy()
    rb = pd.Series(target.margin[mm]).rank(method="average").to_numpy()
    y = target.y[target.available]
    p = de.predict(s[target.available], m).astype(bool)
    return {"auc": m["auc"], "orientation": int(m["orientation"]), "threshold": m["threshold"],
            "bacc": m["bacc"], "tp": int(np.sum(p & (y == 1))), "tn": int(np.sum(~p & (y == 0))),
            "fp": int(np.sum(p & (y == 0))), "fn": int(np.sum(~p & (y == 1))),
            "acc": float(np.mean(p == (y == 1))),
            "margin_spearman": float(np.corrcoef(ra, rb)[0, 1]),
            "trajectory_auc": m["trajectory_auc"], "trajectory_auc_n": int(m["trajectory_auc_n"])}


# -------------------------------------------------------------------- run


def load_config(path: Path) -> dict[str, Any]:
    """The config with its "_comment" / "_why" entries removed."""
    def strip(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if not k.startswith("_")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return strip(json.loads(Path(path).read_text(encoding="utf-8")))


def finalize(cfg: Mapping[str, Any], data: de.EdgeData, run_dir: Path, outdir: Path) -> dict[str, Any]:
    """Run the four steps on a finished search; write the stage tables and the coordinate file."""
    run_dir, outdir = Path(run_dir), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    components = json.loads((run_dir / "components.json").read_text(encoding="utf-8"))
    summary: dict[str, Any] = {}
    coordinates = []
    for edge in de.EDGES:
        name = de.EDGE_NAMES[edge]
        ecfg = cfg["edges"][edge]
        target = data.targets[edge]
        # 1. selection over the hall of fame of the search
        elite = pd.read_csv(run_dir / f"elite_{name}.csv", low_memory=False)
        sel = select(elite, cfg["selection"])
        sel["pool"].to_csv(outdir / f"safe_pool_{name}.csv", index=False)
        sel["key"].to_csv(outdir / f"key_formulas_{name}.csv", index=False)
        # 2. pruning of the seed formula
        seed = choose_seed(sel, ecfg["seed"], cfg["pruning"]["trajectory_gate"])
        pruned, best = prune(seed, target, data.X, data.pi_s, components, data.names, cfg["pruning"])
        pruned.to_csv(outdir / f"pruning_{name}.csv", index=False)
        # 3. coefficients of the fixed structure
        ref = refine(edge, ecfg, cfg["coefficients"], pruned, data)
        order = ["eligible", "auc", "margin_spearman", "coefficient_cost"]
        ref["candidates"].sort_values(order, ascending=[False, False, False, True]).head(
            cfg["coefficients"]["save_top"]).to_csv(outdir / f"coefficients_{name}.csv", index=False)
        ref["roles"].to_csv(outdir / f"coefficient_roles_{name}.csv", index=False)
        # 4. final coefficients and the frozen coordinate
        final = ecfg["final_coefficients"]
        coef = ref["recommended"] if final == "recommended" else [number(x) for x in final]
        tree = coordinate_tree(ecfg["terms"], coef)
        eps = float(cfg["freeze"]["ratio_epsilon"])
        coordinates.append({"edge": edge, "name": ecfg["name"], "ratio_epsilon": eps,
                            "spearman_ranks": "average", "ast": tree,
                            "stored": coordinate_metrics(tree, data, edge, eps)})
        summary[edge] = {
            "selection": sel["summary"],
            "seed": {"formula_id": str(seed.formula_id), "auc": float(seed.auc), "formula": str(seed.formula)},
            "pruning": {"accepted": len(pruned), "recommended_formula": str(best.formula),
                        "recommended_auc": float(best.auc), "recommended_complexity": float(best.complexity)},
            "structure": {"member_formula": str(ref["member"].formula), "member_auc": float(ref["member"].auc),
                          "baseline": ref["baseline"]["formula"], "baseline_auc": ref["baseline"]["auc"]},
            "coefficients": {"recommended": ref["recommended"], "final": coef,
                             "final_differs_from_recommended": [float(x) for x in coef] != ref["recommended"]},
            "coordinate": coordinates[-1]["stored"]}
    doc = {"description": cfg["freeze"]["description"], "pi_s_fit_on_samples": True,
           "coordinates": coordinates}
    with (outdir / "edge_coordinates.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    de.write_json(outdir / "finalize_summary.json", summary)
    return summary
