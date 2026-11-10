"""Components: short expressions over descriptors that skeletons combine.

Components are proposed by random construction, mutation and crossover,
evaluated in worker processes and kept in a registry whose value matrix is a
memory-mapped file shared with the workers.  A component's own BACC is only
metadata; what decides its survival is how much it helps other components
(its context utility, see :func:`.parallel.probe_context`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import grammar as g
from .data import SearchData
from .settings import Settings

VALUE_STORE = "component_values.float64.dat"


class DescriptorSampler:
    """Draws descriptors with the prior weights of the guidance registry.

    Tier A/B/C descriptors are drawn with probability 0.55/0.30/0.15; an
    escape draw uses tier C only.
    """

    def __init__(self, weights: Sequence[float], tiers: Sequence[str],
                 domains: Sequence[str]):
        w = np.asarray(weights, dtype=np.float64).copy()
        w = np.maximum(w, 1e-12)
        w /= w.sum()
        t = np.asarray(tuple(map(str, tiers)))
        self.weights = w
        self.domains = tuple(domains)
        self.tier_a = np.flatnonzero(t == "A").astype(np.int32, copy=False)
        self.tier_b = np.flatnonzero(t == "B").astype(np.int32, copy=False)
        self.tier_c = np.flatnonzero(t == "C").astype(np.int32, copy=False)

    @classmethod
    def from_data(cls, data: SearchData) -> DescriptorSampler:
        return cls(data.prior_weights, data.prior_tiers, data.domains)

    def draw(self, rng: np.random.Generator, *, escape: bool = False,
             exclude: set[int] | None = None) -> int:
        exclude = exclude or set()
        weights, A, B, C = self.weights, self.tier_a, self.tier_b, self.tier_c
        if escape:
            pool = C if len(C) else np.arange(len(weights))
        else:
            u = float(rng.random())
            pool = A if u < .55 and len(A) else (
                B if u < .85 and len(B) else (C if len(C) else np.arange(len(weights))))
        p = weights[pool]
        p = p / p.sum()
        for _ in range(32):
            i = int(rng.choice(pool, p=p))
            if i not in exclude:
                return i
        for i in range(len(weights)):
            if i not in exclude:
                return i
        raise RuntimeError("no descriptor available")


def random_powered_terminal(rng: np.random.Generator, i: int,
                            sampler: DescriptorSampler) -> dict[str, Any]:
    ast = g.terminal(i)
    if rng.random() < .65:
        p = g.POWERS[int(rng.integers(len(g.POWERS)))]
        ast = g.wrap_power(ast, p, sampler.domains[i])
    if rng.random() < .18 and sampler.domains[i] == g.SIGNED:
        ast = {"op": "abs", "arg": ast}
    return ast


def random_component(rng: np.random.Generator, sampler: DescriptorSampler, *,
                     escape: bool = False, max_features: int = 3) -> dict[str, Any]:
    nfeat = 1 if rng.random() < .18 else (2 if max_features < 3 or rng.random() < .80 else 3)
    ids: list[int] = []
    for _ in range(nfeat):
        ids.append(sampler.draw(rng, escape=escape, exclude=set(ids)))
    nodes = [random_powered_terminal(rng, i, sampler) for i in ids]
    if len(nodes) == 1:
        return nodes[0]
    while len(nodes) > 1:
        a = nodes.pop(int(rng.integers(len(nodes))))
        b = nodes.pop(int(rng.integers(len(nodes))))
        op = str(rng.choice(g.COMPONENT_BINARY_OPS,
                            p=np.asarray([.16, .17, .22, .10, .08, .14, .13])))
        if op == "geometric":
            a = {"op": "abs", "arg": a}
            b = {"op": "abs", "arg": b}
        node = {"op": op, "left": a, "right": b}
        if rng.random() < .22:
            coefficient = g.COEFFICIENTS[int(rng.integers(len(g.COEFFICIENTS)))]
            node = {"op": "scale", "coefficient": list(coefficient), "arg": node}
        nodes.append(node)
    return nodes[0]


def initial_components(sampler: DescriptorSampler, settings: Settings,
                       rng: np.random.Generator) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Every descriptor, every descriptor power, then random pairs up to the target."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts = {"RAW": 0, "UNARY": 0, "BINARY": 0}
    domains = sampler.domains

    def add(ast: Mapping[str, Any], source: str) -> None:
        try:
            can = g.validate_component(ast, domains)
        except Exception:
            return
        fid = g.tree_id(can)
        if fid in seen:
            return
        seen.add(fid)
        out.append(can)
        counts[source] += 1

    for i in range(len(domains)):
        add(g.terminal(i), "RAW")
    for i, domain in enumerate(domains):
        for p in g.POWERS:
            add(g.wrap_power(g.terminal(i), p, domain), "UNARY")
        if domain == g.SIGNED:
            add({"op": "abs", "arg": g.terminal(i)}, "UNARY")
    target = settings.initial_component_target
    attempts = 0
    max_attempts = target * settings.max_attempt_factor
    while len(out) < target and attempts < max_attempts:
        attempts += 1
        add(random_component(rng, sampler, escape=(attempts % 10 == 0), max_features=2),
            "BINARY")
    if len(out) < min(target, len(domains) + 100):
        raise RuntimeError(f"only {len(out)} initial components after {attempts} attempts")
    return out[:target], {**counts, "attempts": attempts}


def mutate_component(ast: Mapping[str, Any], rng: np.random.Generator,
                     sampler: DescriptorSampler) -> tuple[dict[str, Any], str]:
    base = g.deep_copy(ast)
    paths = g.node_paths(base)
    op = str(rng.choice(["terminal_replace", "power", "operator", "wrap", "add_feature",
                         "delete_branch"], p=[.28, .18, .18, .14, .14, .08]))
    if op == "terminal_replace":
        tpaths = [p for p in paths if g.node_at(base, p)["op"] == "terminal"]
        if not tpaths:
            return random_component(rng, sampler), "random_immigrant"
        p = tpaths[int(rng.integers(len(tpaths)))]
        old = int(g.node_at(base, p)["feature_index"])
        new = sampler.draw(rng, escape=(rng.random() < .15), exclude={old})
        return g.replace_node(base, p, g.terminal(new)), "mutation"
    if op == "power":
        p = paths[int(rng.integers(len(paths)))]
        node = g.node_at(base, p)
        power = g.POWERS[int(rng.integers(len(g.POWERS)))]
        try:
            domain = g.SIGNED
            if node["op"] == "terminal":
                domain = sampler.domains[int(node["feature_index"])]
            return g.replace_node(base, p, g.wrap_power(node, power, domain)), "mutation"
        except Exception:
            return base, "mutation"
    if op == "operator":
        bpaths = [p for p in paths if g.node_at(base, p)["op"] not in g.UNARY | {"terminal"}]
        if not bpaths:
            return {"op": "abs", "arg": base}, "mutation"
        p = bpaths[int(rng.integers(len(bpaths)))]
        node = g.deep_copy(g.node_at(base, p))
        new_op = str(rng.choice(g.COMPONENT_BINARY_OPS))
        node["op"] = new_op
        if new_op == "geometric":
            node["left"] = {"op": "abs", "arg": node["left"]}
            node["right"] = {"op": "abs", "arg": node["right"]}
        return g.replace_node(base, p, node), "mutation"
    if op == "wrap":
        p = paths[int(rng.integers(len(paths)))]
        node = g.node_at(base, p)
        kind = str(rng.choice(["abs", "power", "scale", "saturating"], p=[.15, .35, .25, .25]))
        if kind == "abs":
            repl = {"op": "abs", "arg": node}
        elif kind == "power":
            repl = {"op": "spower", "power": list(g.POWERS[int(rng.integers(len(g.POWERS)))]),
                    "arg": node}
        elif kind == "scale":
            coefficient = g.COEFFICIENTS[int(rng.integers(len(g.COEFFICIENTS)))]
            repl = {"op": "scale", "coefficient": list(coefficient), "arg": node}
        else:
            i = sampler.draw(rng, escape=(rng.random() < .15),
                             exclude=set(g.feature_indices(base)))
            repl = {"op": "saturating", "left": node,
                    "right": random_powered_terminal(rng, i, sampler)}
        return g.replace_node(base, p, repl), "mutation"
    if op == "add_feature":
        used = set(g.feature_indices(base))
        if len(used) >= 3:
            tpaths = [p for p in paths if g.node_at(base, p)["op"] == "terminal"]
            if not tpaths:
                return base, "mutation"
            p = tpaths[int(rng.integers(len(tpaths)))]
            old = int(g.node_at(base, p)["feature_index"])
            new = sampler.draw(rng, escape=(rng.random() < .20), exclude={old})
            return g.replace_node(base, p, g.terminal(new)), "mutation"
        i = sampler.draw(rng, escape=(rng.random() < .20), exclude=used)
        node = random_powered_terminal(rng, i, sampler)
        bop = str(rng.choice(["add", "sub", "mul", "sgeometric", "saturating"]))
        return {"op": bop, "left": base, "right": node}, "mutation"
    bpaths = [p for p in paths if g.node_at(base, p)["op"] not in g.UNARY | {"terminal"}]
    if not bpaths:
        return base, "mutation"
    p = bpaths[int(rng.integers(len(bpaths)))]
    node = g.node_at(base, p)
    keep = node["left"] if rng.random() < .5 else node["right"]
    return g.replace_node(base, p, keep), "mutation"


def crossover_components(a: Mapping[str, Any], b: Mapping[str, Any],
                         rng: np.random.Generator) -> dict[str, Any]:
    pa = g.node_paths(a)
    pb = g.node_paths(b)
    na = g.node_at(a, pa[int(rng.integers(len(pa)))])
    nb = g.node_at(b, pb[int(rng.integers(len(pb)))])
    op = str(rng.choice(["add", "sub", "mul", "sgeometric", "saturating", "safe_ratio"]))
    return {"op": op, "left": g.deep_copy(na), "right": g.deep_copy(nb)}


def continuous_signature(values: np.ndarray) -> str:
    """Hash of the values rounded on a robust scale; near-identical responses collide."""
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    if not np.any(finite):
        return g.sha256_hex(np.zeros(x.shape, dtype=np.int16).tobytes())
    xf = x[finite]
    med = float(np.median(xf))
    deviation = np.abs(xf - med)
    scale = float(np.quantile(deviation, .75))
    if not math.isfinite(scale) or scale <= 1e-15:
        scale = max(float(np.max(deviation)), 1.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = (x - med) / scale * 256.0
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=32767.0, neginf=-32768.0)
    q = np.clip(np.rint(normalized), -32768, 32767).astype(np.int16)
    return g.sha256_hex(q.tobytes())


def value_id(values: np.ndarray) -> str:
    """Exact-value identity at single precision, or at double precision when
    a finite value lies outside the single-precision range."""
    x = np.asarray(values, dtype=np.float64)
    limit = float(np.finfo(np.float32).max)
    finite = np.isfinite(x)
    if bool(np.all(~finite | (np.abs(x) <= limit))):
        with np.errstate(over="ignore", invalid="ignore"):
            q = np.asarray(x, dtype=np.float32)
        return g.sha256_hex(np.ascontiguousarray(q).tobytes())
    canonical = np.nan_to_num(x, nan=0.0, posinf=np.finfo(np.float64).max,
                              neginf=-np.finfo(np.float64).max)
    payload = WIDE_VALUE_TAG + np.ascontiguousarray(canonical, dtype=np.float64).tobytes()
    return g.sha256_hex(payload)


# Hashed into the identifiers of values beyond single-precision range; kept
# byte for byte so identifiers match the registries of the published runs.
WIDE_VALUE_TAG = b"RC8_FLOAT64_VALUE_ID\0"


class ComponentRegistry:
    """Accepted components: metadata rows plus a memory-mapped value matrix."""

    def __init__(self, root: Path, data: SearchData, settings: Settings, *, create: bool):
        self.root = Path(root)
        self.data = data
        self.capacity = settings.component_capacity
        self.path = self.root / VALUE_STORE
        self.shape = (settings.component_capacity, data.n_samples)
        # Mode w+ creates a sparse zero-filled file, so the full capacity is
        # never written up front.
        self.values = np.memmap(self.path, mode="w+" if create else "r+",
                                dtype=np.float64, shape=self.shape)
        if create:
            self.values.flush()
        self.rows: list[dict[str, Any]] = []
        self.formula_to_id: dict[str, int] = {}
        self.value_to_id: dict[str, int] = {}
        self._asts: list[dict[str, Any] | None] = []
        self._sampling: tuple[int, np.ndarray, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def ast(self, cid: int) -> dict[str, Any]:
        i = int(cid)
        cached = self._asts[i]
        if cached is None:
            cached = json.loads(self.rows[i]["ast_json"])
            self._asts[i] = cached
        return cached

    def ids(self) -> np.ndarray:
        return np.asarray([r["component_id"] for r in self.rows], dtype=np.int32)

    def _sampling_view(self) -> tuple[np.ndarray, np.ndarray]:
        if self._sampling is None or self._sampling[0] != len(self.rows):
            ids = np.asarray([int(r["component_id"]) for r in self.rows], dtype=np.int32)
            if ids.size == 0:
                raise RuntimeError("no component available")
            weights = np.asarray([max(1e-12, float(self.rows[int(i)].get("rf_prior", 0.0)))
                                  for i in ids], dtype=np.float64)
            cdf = np.cumsum(weights)
            cdf /= cdf[-1]
            self._sampling = (len(self.rows), ids, cdf)
        return self._sampling[1], self._sampling[2]

    def sample_component(self, rng: np.random.Generator, mode: str = "rf",
                         exclude: set[int] | None = None) -> int:
        """A component drawn with its prior weight (mean weight of its descriptors)."""
        if mode != "rf":
            raise ValueError(f"the registry samples with mode 'rf' only, not {mode!r}")
        excluded = exclude or set()
        ids, cdf = self._sampling_view()
        for _ in range(64):
            pos = min(len(ids) - 1, int(np.searchsorted(cdf, float(rng.random()), side="right")))
            cid = int(ids[pos])
            if cid not in excluded:
                return cid
        allowed = np.asarray([int(i) for i in ids if int(i) not in excluded], dtype=np.int32)
        if allowed.size == 0:
            raise RuntimeError("no component available after exclusions")
        weights = np.asarray([max(1e-12, float(self.rows[int(i)].get("rf_prior", 0.0)))
                              for i in allowed], dtype=np.float64)
        weights /= weights.sum()
        return int(rng.choice(allowed, p=weights))

    def add(self, result: Mapping[str, Any], *, source: str, generation: int,
            proposal_operator: str, island: int = -1) -> tuple[int | None, str]:
        """Store an evaluated component unless its formula or values are already known."""
        if result.get("status") != "VALID":
            return None, str(result.get("status", "INVALID"))
        ast = dict(result["ast"])
        fid = str(result["formula_id"])
        if fid in self.formula_to_id:
            return self.formula_to_id[fid], "FORMULA_DUPLICATE"
        vals = np.asarray(result["values"], dtype=np.float64)
        vid = str(result["value_id"])
        if vid in self.value_to_id:
            return self.value_to_id[vid], "VALUE_DUPLICATE"
        if len(self.rows) >= self.capacity:
            raise RuntimeError("component registry is full")
        cid = len(self.rows)
        self.values[cid] = vals
        m = dict(result["metrics"])
        row = {
            "component_id": cid, "formula_id": fid, "structure_id": str(result["structure_id"]),
            "value_id": vid, "formula": "", "features": str(result["features"]),
            "feature_names": str(result["feature_names"]), "domain": "",
            "channels": str(result["channels"]), "role_hints": str(result["roles"]),
            "rf_prior": float(result["rf_prior"]), "source": source,
            "created_generation": int(generation), "component_island_id": int(island),
            "proposal_operator": proposal_operator,
            "standalone_bacc": float(m["all_bacc"]),
            "standalone_worst_recall": float(m["all_worst_recall"]),
            "context_utility": 0.0, "context_best_bacc": float(m["all_bacc"]),
            "context_probe_count": 0, "context_anchor_component_id": -1,
            "context_operator": "",
            "continuous_response_id": str(result["continuous_response_id"]),
            "complexity": json.dumps(dict(result["complexity"]), sort_keys=True),
            "ast_json": str(result["ast_json"]), **m,
        }
        self.rows.append(row)
        self._asts.append(ast)
        self.formula_to_id[fid] = cid
        self.value_to_id[vid] = cid
        self._sampling = None
        return cid, "ACCEPTED"

    def flush(self) -> None:
        self.values.flush()

    def state(self) -> dict[str, Any]:
        return {"rows": self.rows, "shape": list(self.shape)}

    def restore(self, state: Mapping[str, Any]) -> None:
        self.rows = [dict(x) for x in state["rows"]]
        self.formula_to_id = {str(r["formula_id"]): int(r["component_id"]) for r in self.rows}
        self.value_to_id = {str(r["value_id"]): int(r["component_id"]) for r in self.rows}
        self._asts = [None] * len(self.rows)
        self._sampling = None

    def table(self) -> pd.DataFrame:
        rows = []
        for r in self.rows:
            x = dict(r)
            ast = json.loads(x["ast_json"])
            x["formula"] = g.to_formula(ast, self.data.names)
            try:
                x["domain"] = g.infer_domain(ast, self.data.domains)
            except Exception:
                x["domain"] = "UNKNOWN"
            rows.append(x)
        return pd.DataFrame(rows)


def ordinary_room(registry: ComponentRegistry, settings: Settings) -> int:
    """Free rows outside the reserve kept for skeleton-feedback components."""
    return max(0, settings.component_capacity - settings.component_feedback_reserve
               - len(registry))


def feedback_room(registry: ComponentRegistry, settings: Settings) -> int:
    return max(0, settings.component_capacity - len(registry))


def choose_component_population(registry: ComponentRegistry, size: int,
                                rng: np.random.Generator) -> list[int]:
    """Reservoir mix: context gain, robust gain, helpfulness, distinct responses,
    structures and channels, prior weight, and a random remainder."""
    ids = registry.ids()
    if ids.size <= size:
        return list(map(int, ids))
    rows = registry.rows
    n = ids.size

    def column(field: str, dtype: type = np.float64) -> np.ndarray:
        cast = int if dtype is np.int64 else float
        return np.fromiter((cast(rows[int(i)].get(field, 0)) for i in ids), dtype=dtype, count=n)

    context = column("context_utility")
    top4 = column("context_top4_mean_gain")
    top8 = column("context_top8_median_gain")
    positive_rate = column("context_positive_rate")
    helpful = column("context_helpful_anchor_count", np.int64)
    rf_prior = column("rf_prior")
    best_gain = ids[np.lexsort((ids, -context))]
    robust = ids[np.lexsort((ids, -top8, -top4))]
    positive = ids[np.lexsort((ids, -helpful, -positive_rate))]
    rf = ids[np.lexsort((ids, -rf_prior))]

    def first_by_key(field: str, default: str = "") -> np.ndarray:
        keys = np.asarray([str(rows[int(i)].get(field, default)) or default for i in ids],
                          dtype=str)
        order = np.lexsort((ids, keys))
        ordered_ids = ids[order]
        ordered_keys = keys[order]
        first = np.empty(len(order), dtype=bool)
        first[0] = True
        first[1:] = ordered_keys[1:] != ordered_keys[:-1]
        return ordered_ids[first]

    structures = first_by_key("structure_id")
    responses = first_by_key("continuous_response_id")
    channels = first_by_key("channels", default="UNCLASSIFIED")
    chosen: list[int] = []
    seen: set[int] = set()

    def add(seq: Iterable[int], target: int) -> None:
        for cid in seq:
            i = int(cid)
            if i not in seen:
                seen.add(i)
                chosen.append(i)
            if len(chosen) >= target:
                break

    add(best_gain, int(size * .25))
    add(robust, int(size * .45))
    add(positive, int(size * .60))
    add(responses, int(size * .75))
    add(structures, int(size * .85))
    add(channels, int(size * .90))
    add(rf, int(size * .95))
    remain = np.asarray([int(i) for i in ids if int(i) not in seen], dtype=np.int32)
    rng.shuffle(remain)
    add(remain, size)
    return chosen[:size]


def select_new_components(results: Sequence[Mapping[str, Any]], registry: ComponentRegistry,
                          limit: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """New valid components to persist; a mix of distinct structures, distinct
    responses, prior weight and standalone BACC when there are too many."""
    valid = []
    formula_seen: set[str] = set()
    value_seen: set[str] = set()
    for r in results:
        if r.get("status") != "VALID":
            continue
        fid = str(r["formula_id"])
        vid = str(r["value_id"])
        if (fid in registry.formula_to_id or vid in registry.value_to_id
                or fid in formula_seen or vid in value_seen):
            continue
        formula_seen.add(fid)
        value_seen.add(vid)
        valid.append(dict(r))
    if len(valid) <= limit:
        return valid
    selected: list[dict[str, Any]] = []
    used: set[int] = set()

    def take(order: Iterable[int], target: int) -> None:
        for i in order:
            if i not in used:
                used.add(i)
                selected.append(valid[i])
            if len(selected) >= target:
                break

    structure_order = _first_of_each(
        sorted(range(len(valid)),
               key=lambda i: (valid[i]["structure_id"], valid[i]["formula_id"])),
        lambda i: valid[i]["structure_id"])
    response_order = _first_of_each(
        sorted(range(len(valid)),
               key=lambda i: (valid[i]["continuous_response_id"], valid[i]["formula_id"])),
        lambda i: valid[i]["continuous_response_id"])
    rf_order = sorted(range(len(valid)), key=lambda i: (-valid[i]["rf_prior"],
                                                        valid[i]["formula_id"]))
    perf_order = sorted(range(len(valid)), key=lambda i: (
        -float(valid[i]["metrics"]["all_bacc"]), valid[i]["formula_id"]))
    take(structure_order, int(limit * .30))
    take(response_order, int(limit * .55))
    take(rf_order, int(limit * .70))
    take(perf_order, int(limit * .85))
    remain = [i for i in range(len(valid)) if i not in used]
    rng.shuffle(remain)
    take(remain, limit)
    return selected[:limit]


def _first_of_each(order: Iterable[int], key: Any) -> list[int]:
    out = []
    seen = set()
    for i in order:
        k = key(i)
        if k not in seen:
            seen.add(k)
            out.append(i)
    return out


def _distinct_by_row_field(registry: ComponentRegistry, ids: Sequence[int], field: str,
                           default: str = "") -> list[int]:
    """First id (in id order) of every distinct value of ``field``."""
    rows = registry.rows
    out = []
    seen = set()
    for i in sorted(ids, key=lambda j: (str(rows[j].get(field, "")), j)):
        key = str(rows[i].get(field, "")) or default
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out


def parse_int_tokens(value: Any) -> set[int]:
    out: set[int] = set()
    for token in str(value or "").split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            out.add(int(token))
        except ValueError:
            continue
    return out


def anchor_panel(registry: ComponentRegistry, pool_ids: Sequence[int], panel_size: int,
                 rng: np.random.Generator) -> tuple[list[int], dict[int, str]]:
    """Anchors for the context scan, drawn from several reservoirs."""
    ids: list[int] = []
    seen: set[int] = set()
    for cid in pool_ids:
        i = int(cid)
        if 0 <= i < len(registry.rows) and i not in seen:
            ids.append(i)
            seen.add(i)
    for row in registry.rows:
        i = int(row["component_id"])
        if i not in seen:
            ids.append(i)
            seen.add(i)
    if not ids:
        return [], {}
    rows = registry.rows
    target = min(int(panel_size), len(ids))
    chosen: list[int] = []
    sources: dict[int, str] = {}
    used: set[int] = set()

    def take(seq: Iterable[int], quota: int, label: str) -> None:
        for cid in seq:
            i = int(cid)
            if i in used:
                continue
            used.add(i)
            chosen.append(i)
            sources[i] = label
            if len([x for x in chosen if sources[x] == label]) >= quota or len(chosen) >= target:
                break

    performance = sorted(ids, key=lambda i: (-float(rows[i].get("standalone_bacc", 0.0)), i))
    context = sorted(ids, key=lambda i: (
        -float(rows[i].get("context_top4_mean_gain", rows[i].get("context_utility", 0.0))),
        -float(rows[i].get("context_positive_rate", 0.0)), i))
    structures = _distinct_by_row_field(registry, ids, "structure_id")
    responses = _distinct_by_row_field(registry, ids, "continuous_response_id")
    channels = _distinct_by_row_field(registry, ids, "channels", "UNCLASSIFIED")
    take(performance, max(1, round(target * .25)), "PERFORMANCE")
    take(context, max(1, round(target * .20)), "CONTEXT")
    take(structures, max(1, round(target * .15)), "STRUCTURE")
    take(responses, max(1, round(target * .15)), "RESPONSE")
    take(channels, max(1, round(target * .10)), "CHANNEL")
    remain = [i for i in ids if i not in used]
    rng.shuffle(remain)
    for i in remain:
        if len(chosen) >= target:
            break
        used.add(i)
        chosen.append(i)
        sources[i] = "RANDOM_ESCAPE"
    return chosen, sources


def screen_anchor_plan(registry: ComponentRegistry, candidate_ids: Sequence[int],
                       anchor_ids: Sequence[int], data: SearchData, screen_count: int,
                       rng: np.random.Generator) -> np.ndarray:
    """For every candidate, the anchors most likely to be complemented by it.

    A cheap screen on 64 rows ranks anchors by low correlation, prediction
    disagreement and disjoint channels; two slots stay random.
    """
    cids = np.asarray(list(map(int, candidate_ids)), dtype=np.int32)
    aids = np.asarray(list(map(int, anchor_ids)), dtype=np.int32)
    if cids.size == 0 or aids.size == 0:
        return np.empty((0, 0), dtype=np.int32)
    rows = registry.rows
    k = min(int(screen_count), len(aids))
    all_idx = np.asarray(data.scoring.all_rows, dtype=np.int32)
    if len(all_idx) > 64:
        sample_pos = np.unique(np.linspace(0, len(all_idx) - 1, 64, dtype=np.int32))
        sample_idx = all_idx[sample_pos]
    else:
        sample_idx = all_idx
    C = np.asarray(registry.values[cids][:, sample_idx], dtype=np.float64)
    A = np.asarray(registry.values[aids][:, sample_idx], dtype=np.float64)

    def zrows(x: np.ndarray) -> np.ndarray:
        y = x - np.mean(x, axis=1, keepdims=True)
        n = np.linalg.norm(y, axis=1, keepdims=True)
        return y / np.where(n > 1e-15, n, 1.0)

    corr = np.abs(zrows(C) @ zrows(A).T)
    cth = np.asarray([float(rows[i].get("threshold", 0.0)) for i in cids])[:, None]
    cor = np.asarray([int(rows[i].get("orientation", 1)) for i in cids])[:, None]
    ath = np.asarray([float(rows[i].get("threshold", 0.0)) for i in aids])[:, None]
    aor = np.asarray([int(rows[i].get("orientation", 1)) for i in aids])[:, None]
    cp = np.where(cor == 1, C >= cth, C <= cth)
    ap = np.where(aor == 1, A >= ath, A <= ath)
    disagreement = np.mean(cp[:, None, :] != ap[None, :, :], axis=2)
    channel_bonus = np.zeros((len(cids), len(aids)), dtype=np.float64)
    overlap = np.zeros_like(channel_bonus)
    anchor_features = [parse_int_tokens(rows[i].get("features", "")) for i in aids]
    anchor_channels = [set(str(rows[i].get("channels", "")).split(";")) - {""} for i in aids]
    for ci, cid in enumerate(cids):
        cf = parse_int_tokens(rows[int(cid)].get("features", ""))
        cc = set(str(rows[int(cid)].get("channels", "")).split(";")) - {""}
        for aj, (af, ac) in enumerate(zip(anchor_features, anchor_channels)):
            channel_bonus[ci, aj] = 1.0 if cc and ac and cc.isdisjoint(ac) else 0.0
            union = len(cf | af)
            overlap[ci, aj] = (len(cf & af) / union) if union else 0.0
    combined = (.45 * (1.0 - corr) + .35 * disagreement + .15 * channel_bonus
                + .05 * (1.0 - overlap))
    plan = np.empty((len(cids), k), dtype=np.int32)
    random_slots = min(2, k)
    q_low = max(1, round((k - random_slots) * .30))
    q_dis = q_low + max(1, round((k - random_slots) * .25))
    q_chan = q_dis + max(1, round((k - random_slots) * .20))
    q_mix = max(k - random_slots, q_chan)
    for ci, cid in enumerate(cids):
        selected: list[int] = []
        selected_set: set[int] = set()

        def add_order(order: np.ndarray, quota: int) -> None:
            for pos in order:
                aid = int(aids[int(pos)])
                if aid == int(cid) or aid in selected_set:
                    continue
                selected.append(aid)
                selected_set.add(aid)
                if len(selected) >= quota:
                    break

        add_order(np.argsort(corr[ci], kind="stable"), q_low)
        add_order(np.argsort(-disagreement[ci], kind="stable"), q_dis)
        add_order(np.argsort(-channel_bonus[ci], kind="stable"), q_chan)
        add_order(np.argsort(-combined[ci], kind="stable"), q_mix)
        available = [int(a) for a in aids if int(a) != int(cid) and int(a) not in selected_set]
        rng.shuffle(available)
        for aid in available:
            if len(selected) >= k:
                break
            selected.append(aid)
            selected_set.add(aid)
        if len(selected) < k:
            fallback = [int(a) for a in aids if int(a) != int(cid)] or [int(aids[0])]
            while len(selected) < k:
                selected.append(fallback[len(selected) % len(fallback)])
        plan[ci] = np.asarray(selected[:k], dtype=np.int32)
    return plan


def select_full_audit(registry: ComponentRegistry, candidate_ids: Sequence[int],
                      screened: Mapping[int, Mapping[str, Any]], limit: int,
                      rng: np.random.Generator) -> list[int]:
    """Candidates re-probed against the whole anchor panel.

    Most of the budget follows the screened gain; a minority is kept for
    response, structure and channel diversity and a random remainder.
    """
    ids = list(map(int, candidate_ids))
    limit = min(max(0, int(limit)), len(ids))
    if limit <= 0:
        return []
    rows = registry.rows
    chosen: list[int] = []
    used: set[int] = set()

    def take(seq: Iterable[int], target: int) -> None:
        for cid in seq:
            i = int(cid)
            if i not in used:
                used.add(i)
                chosen.append(i)
            if len(chosen) >= target:
                break

    by_gain = sorted(ids, key=lambda i: (-float(screened.get(i, {}).get("top4_mean_gain",
                                                                        -np.inf)), i))
    structures = _distinct_by_row_field(registry, ids, "structure_id")
    responses = _distinct_by_row_field(registry, ids, "continuous_response_id")
    channels = _distinct_by_row_field(registry, ids, "channels", "UNCLASSIFIED")
    by_usage = sorted(ids, key=lambda i: (-float(rows[i].get("context_positive_rate", 0.0)),
                                          -float(rows[i].get("context_utility", 0.0)), i))
    take(by_gain, round(limit * .60))
    take(responses, round(limit * .70))
    take(structures, round(limit * .78))
    take(channels, round(limit * .86))
    take(by_usage, round(limit * .92))
    remain = [i for i in ids if i not in used]
    rng.shuffle(remain)
    take(remain, limit)
    return chosen[:limit]


def select_injections(registry: ComponentRegistry, new_ids: Sequence[int], limit: int,
                      rng: np.random.Generator) -> list[int]:
    """Feedback components whose parent skeletons are re-evaluated with them."""
    ids = list(dict.fromkeys(map(int, new_ids)))
    limit = min(max(0, int(limit)), len(ids))
    if limit <= 0:
        return []
    rows = registry.rows
    chosen: list[int] = []
    used: set[int] = set()

    def take(seq: Iterable[int], target: int) -> None:
        for cid in seq:
            i = int(cid)
            if i not in used:
                used.add(i)
                chosen.append(i)
            if len(chosen) >= target:
                break

    robust = sorted(ids, key=lambda i: (-float(rows[i].get("context_top4_mean_gain", 0.0)),
                                        -float(rows[i].get("context_positive_rate", 0.0)), i))
    usage = sorted(ids, key=lambda i: (-float(rows[i].get("context_utility", 0.0)),
                                       -int(rows[i].get("context_helpful_anchor_count", 0)), i))
    structures = _distinct_by_row_field(registry, ids, "structure_id")
    responses = _distinct_by_row_field(registry, ids, "continuous_response_id")
    channels = _distinct_by_row_field(registry, ids, "channels", "UNCLASSIFIED")
    take(robust, round(limit * .25))
    take(usage, round(limit * .45))
    take(responses, round(limit * .65))
    take(structures, round(limit * .80))
    take(channels, round(limit * .90))
    remain = [i for i in ids if i not in used]
    rng.shuffle(remain)
    take(remain, limit)
    return chosen[:limit]
