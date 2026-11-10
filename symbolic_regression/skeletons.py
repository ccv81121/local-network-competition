"""Skeletons: typed trees over components, and the archives that keep them.

Skeletons evolve by mutation and crossover of their trees.  A protected
baseline lineage uses only ordinary components; skeletons built with
feedback components evolve in a side branch and join the active population
only when they beat their parent (see :func:`feedback_gate`).
"""
from __future__ import annotations

import json
import math
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from . import grammar as g
from .metrics import performance_key

SEED_CATEGORIES = (("PHYSICAL_COMPETITION", .30), ("RF_GUIDED", .25),
                   ("CHANNEL_COMPLEMENT", .20), ("TYPED_RANDOM", .15), ("GLOBAL_ESCAPE", .10))
RECORD_METRICS = (
    "threshold", "orientation", "hard_bacc", "hard_worst_recall", "threshold_margin",
    "all_bacc", "all_worst_recall", "all_acc", "all_f1", "all_mcc", "all_auc",
    "soft_bacc", "soft_worst_recall", "soft_acc", "soft_f1", "soft_mcc", "soft_auc",
    "prediction_id",
)
OOF_RECORD_METRICS = (
    "oof_bacc", "oof_worst_recall", "oof_acc", "oof_f1", "oof_mcc", "oof_auc",
    "oof_orientation_agreement", "oof_threshold_std_scaled", "full_fit_bacc",
    "full_fit_worst_recall",
)


class ComponentSampler(Protocol):
    def sample_component(self, rng: np.random.Generator, mode: str = "rf",
                         exclude: set[int] | None = None) -> int: ...


class SnapshotSampler:
    """Component draws inside proposal workers, from arrays fixed for one generation."""

    def __init__(self, ids: np.ndarray, cdf: np.ndarray, channel_codes: np.ndarray):
        self.ids = ids
        self.cdf = cdf
        self.channel_codes = channel_codes

    def sample_component(self, rng: np.random.Generator, mode: str = "rf",
                         exclude: set[int] | None = None) -> int:
        excluded = set(exclude or set())
        n = len(self.ids)
        for _ in range(64):
            if mode == "escape":
                pos = int(rng.integers(n))
            else:
                pos = min(n - 1, int(np.searchsorted(self.cdf, float(rng.random()),
                                                     side="right")))
            cid = int(self.ids[pos])
            if cid not in excluded:
                return cid
        allowed = np.asarray([i for i, cid in enumerate(self.ids) if int(cid) not in excluded],
                             dtype=np.int32)
        if allowed.size == 0:
            raise RuntimeError("no component available after exclusions")
        if mode == "escape":
            return int(self.ids[int(allowed[int(rng.integers(len(allowed)))])])
        raw = np.empty(len(allowed), dtype=np.float64)
        for j, pos in enumerate(allowed):
            prev = float(self.cdf[int(pos) - 1]) if int(pos) > 0 else 0.0
            raw[j] = max(1e-15, float(self.cdf[int(pos)]) - prev)
        raw /= raw.sum()
        return int(self.ids[int(rng.choice(allowed, p=raw))])

    def sample_different_channel(self, rng: np.random.Generator, anchor: int,
                                 exclude: set[int] | None = None) -> int:
        excluded = set(exclude or set()) | {int(anchor)}
        pos = int(np.searchsorted(self.ids, int(anchor)))
        found = pos < len(self.ids) and int(self.ids[pos]) == int(anchor)
        anchor_code = int(self.channel_codes[pos]) if found else -1
        mask = self.channel_codes != anchor_code
        if excluded:
            mask &= ~np.isin(self.ids, np.fromiter(excluded, dtype=np.int32,
                                                   count=len(excluded)))
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return self.sample_component(rng, "rf", excluded)
        return int(self.ids[int(candidates[int(rng.integers(len(candidates)))])])


def seed_template(rng: np.random.Generator, sampler: SnapshotSampler,
                  category: str) -> dict[str, Any]:
    """An initial skeleton of one of the five seed categories."""
    leaf = g.component_leaf
    mode = "escape" if category == "GLOBAL_ESCAPE" else "rf"
    a = sampler.sample_component(rng, mode)
    b = sampler.sample_component(rng, mode, exclude={a})
    A = leaf(a)
    B = leaf(b)
    if category == "PHYSICAL_COMPETITION":
        alpha = g.COEFFICIENTS[int(rng.integers(len(g.COEFFICIENTS)))]
        core = {"op": "sub", "left": A,
                "right": {"op": "scale", "coefficient": list(alpha), "arg": B}}
        if rng.random() < .45:
            c = sampler.sample_component(rng, "rf", exclude={a, b})
            return {"op": "saturating", "left": core, "right": leaf(c)}
        return core
    if category == "RF_GUIDED":
        op = str(rng.choice(["sub", "add", "mul", "sgeometric", "saturating"],
                            p=[.35, .15, .15, .10, .25]))
        if op == "saturating":
            c = sampler.sample_component(rng, "rf", exclude={a, b})
            return {"op": "saturating", "left": {"op": "sub", "left": A, "right": B},
                    "right": leaf(c)}
        return {"op": op, "left": A, "right": B}
    if category == "CHANNEL_COMPLEMENT":
        # b above is drawn and discarded, as in the published runs.
        b = sampler.sample_different_channel(rng, a, exclude={a})
        B = leaf(b)
        c = sampler.sample_component(rng, "rf", exclude={a, b})
        C = leaf(c)
        left = {"op": "sub", "left": A, "right": B}
        coefficient = g.COEFFICIENTS[int(rng.integers(len(g.COEFFICIENTS)))]
        return {"op": "add", "left": left,
                "right": {"op": "scale", "coefficient": list(coefficient), "arg": C}}
    if category == "TYPED_RANDOM":
        op = str(rng.choice(g.SKELETON_BINARY_OPS))
        out = {"op": op, "left": A, "right": B}
        if rng.random() < .5:
            c = sampler.sample_component(rng, "rf", exclude={a, b})
            bop = str(rng.choice(["add", "mul", "saturating"]))
            out = {"op": bop, "left": out, "right": leaf(c)}
        return out
    return {"op": str(rng.choice(["sub", "add", "mul", "sgeometric"])), "left": A, "right": B}


def seed_quotas(target: int) -> dict[str, int]:
    quotas = {name: int(math.floor(target * f)) for name, f in SEED_CATEGORIES}
    quotas[SEED_CATEGORIES[0][0]] += target - sum(quotas.values())
    return quotas


MUTATIONS = ("replace_component", "add_branch", "delete_branch", "operator", "parameter",
             "wrap", "rotate")


def mutate_skeleton(ast: Mapping[str, Any], rng: np.random.Generator,
                    sampler: ComponentSampler, *,
                    joint: bool = False) -> tuple[dict[str, Any], str]:
    base = g.deep_copy(ast)
    paths = g.node_paths(base)
    probs = np.asarray([.25, .17, .10, .18, .14, .10, .06])
    if joint:
        probs = np.asarray([.20, .14, .08, .15, .16, .12, .05])
        probs = probs / probs.sum()
    kind = str(rng.choice(list(MUTATIONS), p=probs))
    used = set(g.skeleton_component_ids(base))

    def binary_paths(ops: set[str] | None = None) -> list[tuple[str, ...]]:
        if ops is None:
            return [p for p in paths if g.node_at(base, p)["op"] not in g.UNARY | {"component"}]
        return [p for p in paths if g.node_at(base, p)["op"] in ops]

    if kind == "replace_component":
        leaves = [p for p in paths if g.node_at(base, p)["op"] == "component"]
        p = leaves[int(rng.integers(len(leaves)))]
        old = int(g.node_at(base, p)["component_id"])
        new = sampler.sample_component(rng, "rf", exclude=used | {old})
        return g.replace_node(base, p, g.component_leaf(new)), kind
    if kind == "add_branch":
        if len(g.skeleton_component_ids(base)) >= g.MAX_SKELETON_LEAVES:
            return mutate_skeleton(base, rng, sampler, joint=joint)
        cid = sampler.sample_component(rng, "rf", exclude=used)
        op = str(rng.choice(["add", "sub", "mul", "saturating"], p=[.25, .25, .25, .25]))
        return {"op": op, "left": base, "right": g.component_leaf(cid)}, kind
    if kind == "delete_branch":
        binary = binary_paths()
        if not binary:
            return base, kind
        p = binary[int(rng.integers(len(binary)))]
        node = g.node_at(base, p)
        keep = node["left"] if rng.random() < .5 else node["right"]
        return g.replace_node(base, p, keep), kind
    if kind == "operator":
        binary = binary_paths()
        if not binary:
            return base, kind
        p = binary[int(rng.integers(len(binary)))]
        node = g.deep_copy(g.node_at(base, p))
        node["op"] = str(rng.choice(g.SKELETON_BINARY_OPS))
        return g.replace_node(base, p, node), kind
    if kind == "parameter":
        p = paths[int(rng.integers(len(paths)))]
        node = g.node_at(base, p)
        if node["op"] == "scale":
            repl = dict(node)
            repl["coefficient"] = list(g.COEFFICIENTS[int(rng.integers(len(g.COEFFICIENTS)))])
        elif node["op"] in {"power", "spower"}:
            repl = dict(node)
            repl["power"] = list(g.POWERS[int(rng.integers(len(g.POWERS)))])
        else:
            coefficient = g.COEFFICIENTS[int(rng.integers(len(g.COEFFICIENTS)))]
            repl = {"op": "scale", "coefficient": list(coefficient), "arg": node}
        return g.replace_node(base, p, repl), kind
    if kind == "wrap":
        p = paths[int(rng.integers(len(paths)))]
        node = g.node_at(base, p)
        wrapper = str(rng.choice(["abs", "spower", "saturating"]))
        if wrapper == "abs":
            repl = {"op": "abs", "arg": node}
        elif wrapper == "spower":
            repl = {"op": "spower", "power": list(g.POWERS[int(rng.integers(len(g.POWERS)))]),
                    "arg": node}
        else:
            cid = sampler.sample_component(rng, "rf", exclude=used)
            repl = {"op": "saturating", "left": node, "right": g.component_leaf(cid)}
        return g.replace_node(base, p, repl), kind
    # Reassociate (a o b) o c into a o (b o c).
    binary = binary_paths({"add", "sub", "mul"})
    if not binary:
        return base, kind
    p = binary[int(rng.integers(len(binary)))]
    node = g.node_at(base, p)
    if node["left"]["op"] != node["op"]:
        return base, kind
    repl = {"op": node["op"], "left": node["left"]["left"],
            "right": {"op": node["op"], "left": node["left"]["right"], "right": node["right"]}}
    return g.replace_node(base, p, repl), kind


def crossover_skeletons(a: Mapping[str, Any], b: Mapping[str, Any],
                        rng: np.random.Generator) -> dict[str, Any]:
    pa = g.node_paths(a)
    pb = g.node_paths(b)
    path = pa[int(rng.integers(len(pa)))]
    donor = g.node_at(b, pb[int(rng.integers(len(pb)))])
    return g.replace_node(a, path, donor)


def parameter_neighbors(ast: Mapping[str, Any], max_neighbors: int,
                        rng: np.random.Generator) -> list[dict[str, Any]]:
    """The skeleton and up to ``max_neighbors - 1`` variants of its powers and coefficients."""
    out = [g.canonicalize_skeleton(ast)]
    seen = {g.skeleton_id(out[0])}
    paths = g.node_paths(ast)
    parameter_paths = [p for p in paths if g.node_at(ast, p)["op"] in {"scale", "power", "spower"}]
    attempts = 0
    while len(out) < max_neighbors and attempts < max_neighbors * 20:
        attempts += 1
        if parameter_paths:
            p = parameter_paths[int(rng.integers(len(parameter_paths)))]
            node = g.deep_copy(g.node_at(ast, p))
            if node["op"] == "scale":
                node["coefficient"] = list(g.COEFFICIENTS[int(rng.integers(len(g.COEFFICIENTS)))])
            else:
                node["power"] = list(g.POWERS[int(rng.integers(len(g.POWERS)))])
            candidate = g.replace_node(ast, p, node)
        else:
            p = paths[int(rng.integers(len(paths)))]
            node = g.node_at(ast, p)
            coefficient = g.COEFFICIENTS[int(rng.integers(len(g.COEFFICIENTS)))]
            candidate = g.replace_node(ast, p, {"op": "scale", "coefficient": list(coefficient),
                                                "arg": node})
        try:
            can = g.canonicalize_skeleton(candidate)
            fid = g.skeleton_id(can)
        except Exception:
            continue
        if fid not in seen:
            seen.add(fid)
            out.append(can)
    return out


def simplification_candidates(ast: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every skeleton obtained by deleting one operator node."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(x: Mapping[str, Any]) -> None:
        try:
            can = g.canonicalize_skeleton(x)
            fid = g.skeleton_id(can)
        except Exception:
            return
        if fid not in seen:
            seen.add(fid)
            out.append(can)

    for p in g.node_paths(ast):
        node = g.node_at(ast, p)
        op = node["op"]
        if op in g.UNARY:
            add(g.replace_node(ast, p, node["arg"]))
        elif op != "component":
            add(g.replace_node(ast, p, node["left"]))
            add(g.replace_node(ast, p, node["right"]))
    return out


def archive_record(result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                   names: Sequence[str], source: str, generation: int,
                   island: int = -1) -> dict[str, Any]:
    """Flat record of an evaluated skeleton, as stored in archives and outputs."""
    c = dict(result["complexity"])
    ids = sorted(set(g.skeleton_component_ids(result["component_ast"])))

    def tokens(field: str) -> str:
        return ";".join(sorted({t for i in ids for t in str(rows[i].get(field, "")).split(";")
                                if t}))

    expanded = result["expanded_ast"]
    record = {
        "formula_id": result["formula_id"],
        "component_level_formula_id": result["component_level_formula_id"],
        "structure_id": result["structure_id"], "topology_id": result["topology_id"],
        "formula": g.to_formula(expanded, names),
        "component_level_formula": g.skeleton_formula(result["component_ast"]),
        "ast_json": json.dumps(expanded, sort_keys=True, separators=(",", ":")),
        "component_ast_json": json.dumps(result["component_ast"], sort_keys=True,
                                         separators=(",", ":")),
        "component_ids": ";".join(map(str, ids)),
        "component_sources": ";".join(sorted({str(rows[i]["source"]) for i in ids})),
        "component_channels": tokens("channels"),
        "component_roles": tokens("role_hints"),
        "dynamic_component_count": sum(1 for i in ids if int(rows[i]["created_generation"]) > 0),
        "raw_feature_count": c["raw_feature_count"],
        "raw_features": ";".join(names[i] for i in sorted(set(g.feature_indices(expanded)))),
        "source": source, "generation": int(generation), "skeleton_island_id": int(island),
        "safe": bool(result["safe"]),
        **{k: result[k] for k in RECORD_METRICS},
        **{k: result[k] for k in OOF_RECORD_METRICS if k in result},
        "tree_depth": c["expanded_tree_depth"],
        "component_level_tree_depth": c["component_level_tree_depth"],
        "component_leaf_count": c["component_leaf_count"],
        "operator_nodes": c["expanded_operator_nodes"],
        "component_level_operator_nodes": c["component_level_operator_nodes"],
        "complexity": json.dumps(c, sort_keys=True),
    }
    return record


class Archives:
    """Best records by formula, by topology and by prediction pattern.

    The baseline triple is updated only by the ordinary skeleton lineage, the
    feedback triple only by the feedback side branch; the merged triple sees
    both and is what the run reports.
    """

    BRANCHES = ("performance", "topology", "prediction")

    def __init__(self, caps: tuple[int, int, int], scoring: str):
        self.caps = caps
        self.scoring = scoring
        self.performance: dict[str, dict[str, Any]] = {}
        self.topology: dict[str, dict[str, Any]] = {}
        self.prediction: dict[str, dict[str, Any]] = {}
        self.baseline_performance: dict[str, dict[str, Any]] = {}
        self.baseline_topology: dict[str, dict[str, Any]] = {}
        self.baseline_prediction: dict[str, dict[str, Any]] = {}
        self.feedback_performance: dict[str, dict[str, Any]] = {}
        self.feedback_topology: dict[str, dict[str, Any]] = {}
        self.feedback_prediction: dict[str, dict[str, Any]] = {}

    def key(self, record: Mapping[str, Any]) -> tuple[Any, ...]:
        return performance_key(record, self.scoring)

    def _triple(self, prefix: str) -> tuple[dict, dict, dict]:
        return tuple(getattr(self, f"{prefix}{name}") for name in self.BRANCHES)

    def _update(self, triple: tuple[dict, dict, dict],
                records: Sequence[Mapping[str, Any]]) -> set[str]:
        performance, topology, prediction = triple
        entered: set[str] = set()
        for r0 in records:
            r = dict(r0)
            if not bool(r.get("safe")):
                continue
            fid = str(r["formula_id"])
            old = performance.get(fid)
            if old is None or self.key(r) < self.key(old):
                performance[fid] = r
                entered.add(fid)
            tid = str(r["topology_id"])
            old = topology.get(tid)
            if old is None or self.key(r) < self.key(old):
                topology[tid] = r
                entered.add(fid)
            pid = str(r["prediction_id"])
            old = prediction.get(pid)
            if old is None or self.key(r) < self.key(old):
                prediction[pid] = r
                entered.add(fid)
        return entered

    def _trim(self, triple: tuple[dict, dict, dict]) -> None:
        for archive, cap, field in zip(triple, self.caps,
                                       ("formula_id", "topology_id", "prediction_id")):
            kept = sorted(archive.values(), key=self.key)[:cap]
            archive.clear()
            archive.update((str(r[field]), r) for r in kept)

    def _add(self, prefix: str, records: Sequence[Mapping[str, Any]]) -> set[str]:
        triple = self._triple(prefix)
        entered = self._update(triple, records)
        self._trim(triple)
        return entered

    def add_baseline(self, records: Sequence[Mapping[str, Any]]) -> set[str]:
        entered = self._add("baseline_", records)
        self._add("", records)
        return entered

    def add_feedback_side(self, records: Sequence[Mapping[str, Any]]) -> set[str]:
        return self._add("feedback_", records)

    def add_migrations(self, records: Sequence[Mapping[str, Any]]) -> set[str]:
        entered = self._add("feedback_", records)
        self._add("", records)
        return entered

    def best(self, prefix: str = "") -> dict[str, Any] | None:
        values = sorted(getattr(self, f"{prefix}performance").values(), key=self.key)
        return values[0] if values else None

    def feedback_sources(self) -> list[dict[str, Any]]:
        """Records whose components feedback mutates: baseline first, then side branch."""
        seen: set[str] = set()
        out = []
        for collection in self._triple("baseline_") + self._triple("feedback_"):
            for r in sorted(collection.values(), key=self.key):
                fid = str(r.get("formula_id", ""))
                if fid in seen:
                    continue
                seen.add(fid)
                out.append(dict(r))
        return out

    def state(self) -> dict[str, Any]:
        return {f"{prefix}{name}": getattr(self, f"{prefix}{name}")
                for prefix in ("", "baseline_", "feedback_") for name in self.BRANCHES}

    def restore(self, state: Mapping[str, Any]) -> None:
        for key in self.state():
            setattr(self, key, {str(k): dict(v) for k, v in state.get(key, {}).items()})


def _layered_selection(records: Sequence[Mapping[str, Any]], target: int,
                       rng: np.random.Generator, key: Any,
                       fractions: tuple[float, float, float],
                       first_minimum: int) -> list[dict[str, Any]]:
    """Best records, then one per topology, then one per prediction pattern,
    then a random remainder, each layer filling up to its fraction of ``target``."""
    valid = list(records)
    valid.sort(key=key)
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(seq: Iterable[Mapping[str, Any]], n: int) -> None:
        for r in seq:
            fid = str(r["formula_id"])
            if fid not in seen:
                seen.add(fid)
                chosen.append(dict(r))
            if len(chosen) >= n:
                break

    def first_per(field: str) -> list[Mapping[str, Any]]:
        out = []
        found = set()
        for r in valid:
            value = str(r[field])
            if value not in found:
                found.add(value)
                out.append(r)
        return out

    add(valid, max(first_minimum, int(target * fractions[0])))
    add(first_per("topology_id"), int(target * fractions[1]))
    add(first_per("prediction_id"), int(target * fractions[2]))
    remain = [r for r in valid if str(r["formula_id"]) not in seen]
    rng.shuffle(remain)
    add(remain, target)
    return chosen[:target]


def select_skeleton_population(records: Sequence[Mapping[str, Any]], target: int,
                               rng: np.random.Generator, key: Any) -> list[dict[str, Any]]:
    valid = [dict(r) for r in records if bool(r.get("safe"))]
    if not valid:
        raise RuntimeError("no safe skeleton candidate")
    chosen = _layered_selection(valid, target, rng, key, (.15, .50, .75), 1)
    return [json.loads(r["component_ast_json"]) for r in chosen]


def select_side_population(records: Sequence[Mapping[str, Any]], target: int,
                           rng: np.random.Generator, key: Any) -> list[dict[str, Any]]:
    valid = [dict(r) for r in records if bool(r.get("safe"))]
    if not valid or target <= 0:
        return []
    return _layered_selection(valid, target, rng, key, (.30, .60, .80), 1)


def select_migrations(records: Sequence[Mapping[str, Any]], cap: int,
                      rng: np.random.Generator, key: Any) -> list[dict[str, Any]]:
    eligible = [dict(r) for r in records
                if bool(r.get("feedback_migrate", False)) and bool(r.get("safe", False))]
    if not eligible or cap <= 0:
        return []
    return _layered_selection(eligible, cap, rng, key, (.50, .75, .90), 1)


def feedback_gate(result: Mapping[str, Any], meta: Mapping[str, Any]) -> tuple[bool, str, float]:
    """Whether a feedback skeleton may leave the side branch.

    It must raise BACC over its parent skeleton, or tie on BACC with a
    higher worst-class recall and no larger expanded tree.
    """
    if result.get("status") not in {"VALID", "UNSAFE"} or not bool(result.get("safe", False)):
        return False, "UNSAFE", float("-inf")
    child = float(result.get("all_bacc", float("-inf")))
    parent = float(meta.get("parent_bacc", float("-inf")))
    delta = child - parent
    tol = 1e-15
    if delta > tol:
        return True, "POSITIVE_BACC", delta
    if abs(delta) <= tol:
        worst = float(result.get("all_worst_recall", -1.0))
        parent_worst = float(meta.get("parent_worst_recall", -1.0))
        comp = dict(result.get("complexity", {}))
        nodes = int(comp.get("expanded_operator_nodes", 10**9))
        depth = int(comp.get("expanded_tree_depth", 10**9))
        if (worst > parent_worst + tol and nodes <= int(meta.get("parent_operator_nodes", 10**9))
                and depth <= int(meta.get("parent_tree_depth", 10**9))):
            return True, "PARETO_WORST_RECALL", delta
    return False, "SIDE_ARCHIVE_ONLY", delta


def final_candidates(archives: Archives, cap: int) -> list[dict[str, Any]]:
    """Records re-evaluated at the end: best formulas, then topologies, then patterns."""
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(rows: Iterable[Mapping[str, Any]], target: int) -> None:
        for r in rows:
            fid = str(r["component_level_formula_id"])
            if fid not in seen:
                seen.add(fid)
                chosen.append(dict(r))
            if len(chosen) >= target:
                break

    add(sorted(archives.performance.values(), key=archives.key), max(1, int(cap * .60)))
    add(sorted(archives.topology.values(), key=archives.key), max(1, int(cap * .82)))
    add(sorted(archives.prediction.values(), key=archives.key), cap)
    return chosen[:cap]
