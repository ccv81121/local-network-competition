"""Batched proposal and evaluation in forked worker processes.

Every batched step is planned in the parent: each attempt gets its own seed
drawn from the parent stream, workers build or evaluate candidates, and the
parent merges results in attempt order.  Results therefore do not depend on
the number of workers or on the batch size.

Candidate values travel through temporary memory-mapped files (in /dev/shm
by default) instead of pickles, and workers read component values directly
from the registry's value file.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import grammar as g
from .components import (ComponentRegistry, DescriptorSampler, anchor_panel,
                         continuous_signature, crossover_components, mutate_component,
                         random_component, screen_anchor_plan, select_full_audit, value_id)
from .data import SearchData
from .metrics import Scoring
from .settings import Settings
from .skeletons import (SnapshotSampler, crossover_skeletons, mutate_skeleton, seed_quotas,
                        seed_template, SEED_CATEGORIES)

UINT64_MAX = np.iinfo(np.uint64).max


@dataclass
class WorkerContext:
    """Everything a worker needs; inherited through fork, never pickled."""

    features: np.ndarray
    epsilon: float
    domains: tuple[str, ...]
    scoring: Scoring
    sampler: DescriptorSampler
    channels: tuple[str, ...]
    primary_roles: tuple[str, ...]
    allowed_roles: tuple[str, ...]
    prior_weights: np.ndarray
    names: tuple[str, ...]
    store_path: str
    store_shape: tuple[int, int]
    component_values: np.memmap | None = None
    candidate_store: tuple[str, tuple[int, int], np.memmap] | None = None
    snapshot: tuple[tuple[str, ...], SnapshotSampler] | None = None


_context: WorkerContext | None = None


def _install(context: WorkerContext) -> None:
    global _context
    _context = context
    context.component_values = np.memmap(context.store_path, mode="r", dtype=np.float64,
                                         shape=context.store_shape)


def _ctx() -> WorkerContext:
    if _context is None:
        raise RuntimeError("worker context not installed")
    return _context


def _candidate_store(path: str, shape: tuple[int, int]) -> np.memmap:
    """One writable mapping per worker and evaluation step."""
    ctx = _ctx()
    if ctx.candidate_store is None or ctx.candidate_store[:2] != (path, shape):
        if ctx.candidate_store is not None:
            old = ctx.candidate_store[2]
            ctx.candidate_store = None
            old.flush()
            del old
        ctx.candidate_store = (path, shape, np.memmap(path, mode="r+", dtype=np.float64,
                                                      shape=shape))
    return ctx.candidate_store[2]


def _snapshot_sampler(paths: tuple[str, ...]) -> SnapshotSampler:
    ctx = _ctx()
    if ctx.snapshot is None or ctx.snapshot[0] != paths:
        arrays = [np.load(p, mmap_mode="r", allow_pickle=False) for p in paths]
        ctx.snapshot = (paths, SnapshotSampler(*arrays))
    return ctx.snapshot[1]


# ------------------------------------------------------------ worker kernels


def _component_metadata(ast: Mapping[str, Any]) -> dict[str, Any]:
    ctx = _ctx()
    feats = sorted(set(g.feature_indices(ast)))
    roles = set()
    for i in feats:
        roles.add(ctx.primary_roles[i])
        for token in ctx.allowed_roles[i].split(";"):
            if token and token != "nan":
                roles.add(token)
    return {
        "formula_id": g.tree_id(ast),
        "structure_id": g.tree_id(g.parameter_free(ast)),
        "features": ";".join(map(str, feats)),
        "feature_names": ";".join(ctx.names[i] for i in feats),
        "channels": ";".join(sorted({ctx.channels[i] for i in feats})),
        "roles": ";".join(sorted(roles)),
        "rf_prior": float(np.mean(ctx.prior_weights[feats])) if feats else 0.0,
        "complexity": dict(g.expression_complexity(ast)),
        "ast_json": json.dumps(ast, sort_keys=True, separators=(",", ":")),
    }


def _component_batch(payload: tuple[str, tuple[int, int], list]) -> dict[str, Any]:
    ctx = _ctx()
    value_path, value_shape, items = payload
    store = _candidate_store(value_path, tuple(value_shape))
    rows: list[dict[str, Any]] = []
    pending = []
    for idx, ast in items:
        try:
            can = g.validate_component(ast, ctx.domains)
            values, diag = g.evaluate_expression(can, ctx.features, epsilon=ctx.epsilon)
            pending.append((int(idx), can, values, diag, _component_metadata(can)))
        except Exception as exc:
            rows.append({"index": int(idx), "status": "INVALID", "error": repr(exc)})
    if pending:
        metrics = ctx.scoring.metrics(np.stack([x[2] for x in pending], axis=0))
        for (idx, can, values, diag, meta), m in zip(pending, metrics):
            ok = bool(m["safe"] and not bool(diag.get("overflow_flag", False)))
            store[idx] = values
            rows.append({"index": idx, "status": "VALID" if ok else "UNSAFE", "ast": can,
                         "metrics": m, **meta, "value_id": value_id(values),
                         "continuous_response_id": continuous_signature(values)})
    return {"rows": rows}


def _skeleton_batch(payload: tuple[str, tuple[int, int], list, dict, tuple]) -> dict[str, Any]:
    ctx = _ctx()
    value_path, value_shape, items, component_asts, domains = payload
    store = _candidate_store(value_path, tuple(value_shape))
    matrix = ctx.component_values

    def component_values(cid: int) -> np.ndarray:
        return matrix[cid]

    cache: dict[bytes, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    pending = []
    for idx, ast in items:
        try:
            can, expanded, complexity = g.validate_skeleton(ast, component_asts.__getitem__,
                                                            domains)
            values = np.asarray(g.evaluate_skeleton(can, component_values, ctx.epsilon, cache),
                                dtype=np.float64)
            pending.append((int(idx), can, expanded, complexity, values))
        except Exception as exc:
            rows.append({"index": int(idx), "status": "INVALID", "error": repr(exc)})
    if pending:
        metrics = ctx.scoring.metrics(np.stack([x[4] for x in pending], axis=0))
        for (idx, can, expanded, complexity, values), m in zip(pending, metrics):
            store[idx] = np.asarray(values, dtype=np.float64)
            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                max_abs_score = float(np.max(np.abs(finite_values)))
                score_span = float(np.max(finite_values) - np.min(finite_values))
            else:
                max_abs_score = score_span = float("inf")
            rows.append({
                "index": idx, "status": "VALID" if m["safe"] else "UNSAFE", "error": "",
                "component_ast": can, "expanded_ast": expanded,
                "formula_id": g.tree_id(expanded),
                "component_level_formula_id": g.skeleton_id(can),
                "topology_id": g.topology_id(can),
                "structure_id": g.tree_id(g.parameter_free(can)),
                "component_ids": ";".join(map(str, sorted(set(g.skeleton_component_ids(can))))),
                "complexity": complexity, "max_abs_score": max_abs_score,
                "score_span": score_span, **m})
    return {"rows": rows}


def _component_proposal_batch(items: list) -> dict[str, Any]:
    ctx = _ctx()
    rows = []
    for attempt, seed, mode, parent_a, parent_b in items:
        rng = np.random.default_rng(np.uint64(seed))
        try:
            if mode == "crossover":
                ast = crossover_components(parent_a or {}, parent_b or {}, rng)
                operator = "crossover"
            elif mode == "mutation":
                ast, operator = mutate_component(parent_a or {}, rng, ctx.sampler)
            else:
                ast = random_component(rng, ctx.sampler, escape=True, max_features=3)
                operator = "random_immigrant"
            can = g.validate_component(ast, ctx.domains)
            rows.append({"attempt": int(attempt), "status": "VALID", "ast": can,
                         "operator": operator})
        except Exception:
            rows.append({"attempt": int(attempt), "status": "INVALID"})
    return {"rows": rows}


def _skeleton_proposal_batch(payload: tuple[tuple[str, ...], bool, list]) -> dict[str, Any]:
    snapshot, joint, items = payload
    sampler = _snapshot_sampler(snapshot)
    rows = []
    for attempt, seed, mode, parent_a, parent_b in items:
        rng = np.random.default_rng(np.uint64(seed))
        try:
            if mode == "crossover":
                candidate = crossover_skeletons(parent_a or {}, parent_b or {}, rng)
                operator = "crossover"
            elif mode == "mutation":
                candidate, operator = mutate_skeleton(parent_a or {}, rng, sampler,
                                                      joint=bool(joint))
            else:
                candidate = seed_template(rng, sampler, "TYPED_RANDOM")
                operator = "de_novo"
            rows.append({"attempt": int(attempt), "status": "VALID",
                         "ast": g.check_skeleton_shape(candidate), "operator": operator})
        except Exception:
            rows.append({"attempt": int(attempt), "status": "INVALID"})
    return {"rows": rows}


def _seed_batch(payload: tuple[tuple[str, ...], list]) -> dict[str, Any]:
    snapshot, items = payload
    sampler = _snapshot_sampler(snapshot)
    rows = []
    for attempt, seed, category in items:
        rng = np.random.default_rng(np.uint64(seed))
        try:
            candidate = seed_template(rng, sampler, str(category))
            rows.append({"attempt": int(attempt), "status": "VALID",
                         "ast": g.check_skeleton_shape(candidate)})
        except Exception:
            rows.append({"attempt": int(attempt), "status": "INVALID"})
    return {"rows": rows}


CONTEXT_OPERATORS = ("add", "sub", "mul", "saturating")


def _context_batch(payload: tuple[list[int], list[list[int]], list[list[str]], str, int,
                                  dict[int, float]]) -> dict[str, Any]:
    """Gain in BACC when candidate components are combined with anchors."""
    ctx = _ctx()
    candidate_ids, anchor_plan, source_plan, audit_level, anchor_block, baseline = payload
    if not candidate_ids:
        return {"rows": [], "expressions": 0}
    matrix = ctx.component_values
    cids = np.asarray(candidate_ids, dtype=np.int32)
    aids = np.asarray(anchor_plan, dtype=np.int32)
    C = np.asarray(matrix[cids], dtype=np.float64)[:, None, :]
    gain_blocks = []
    bacc_blocks = []
    block = max(1, int(anchor_block))
    for a0 in range(0, aids.shape[1], block):
        a1 = min(aids.shape[1], a0 + block)
        subids = aids[:, a0:a1]
        A = np.asarray(matrix[subids], dtype=np.float64)
        with np.errstate(all="ignore"):
            scores = np.stack((A + C, A - C, A * C, A / (1.0 + np.abs(C))), axis=2)
        nc, na, no, ns = scores.shape
        bacc = np.asarray(ctx.scoring.context_bacc(scores.reshape(nc * na * no, ns)))
        bacc = bacc.reshape(nc, na, no)
        base = np.asarray([[float(baseline.get(int(a), 0.0)) for a in row] for row in subids],
                          dtype=np.float64)
        bacc_blocks.append(bacc)
        gain_blocks.append(bacc - base[:, :, None])
    gains = np.concatenate(gain_blocks, axis=1)
    bacc_all = np.concatenate(bacc_blocks, axis=1)
    nc, na, no = gains.shape
    rows = []
    for i, cid in enumerate(cids):
        gi = gains[i]
        order = np.argsort(gi.ravel(), kind="stable")[::-1]
        aj, oj = np.unravel_index(int(order[0]), gi.shape)
        sorted_g = gi.ravel()[order]
        helpful = gi > 0
        helpful_anchor = np.any(helpful, axis=1)
        helpful_sources = {str(source_plan[i][j]) for j in range(na) if helpful_anchor[j]}
        rows.append({
            "component_id": int(cid), "gain": float(gi[aj, oj]),
            "bacc": float(bacc_all[i, aj, oj]),
            "anchor_id": int(aids[i, aj]), "operator": CONTEXT_OPERATORS[int(oj)],
            "probe_count": int(na * no),
            "top4_mean_gain": float(np.mean(sorted_g[:min(4, len(sorted_g))])),
            "top8_median_gain": float(np.median(sorted_g[:min(8, len(sorted_g))])),
            "positive_context_rate": float(np.mean(helpful)),
            "helpful_anchor_count": int(np.sum(helpful_anchor)),
            "helpful_operator_count": int(np.sum(np.any(helpful, axis=0))),
            "anchor_source_diversity": int(len(helpful_sources)), "audit_level": audit_level})
    return {"rows": rows, "expressions": int(nc * na * no)}


# ------------------------------------------------------------- parent side


def effective_task_size(n: int, jobs: int, requested: int) -> int:
    """The requested size, raised so that each worker gets at most four batches."""
    if n <= 0:
        return 1
    workers = max(1, min(int(jobs), int(n)))
    requested = max(1, int(requested))
    four_per_worker = max(1, int(math.ceil(n / max(1, workers * 4))))
    return max(1, min(int(n), max(requested, four_per_worker)))


def batch_slices(n: int, size: int) -> list[tuple[int, int]]:
    return [(i, min(n, i + size)) for i in range(0, n, size)]


class Workers:
    """A fork-based process pool (or in-process execution for one job)."""

    def __init__(self, data: SearchData, settings: Settings, registry: ComponentRegistry,
                 scratch: Path | None = None):
        self.settings = settings
        self.data = data
        self.registry = registry
        self.run_dir = registry.root
        self.scratch = Path(scratch) if scratch else None
        self.tag = hashlib.sha256(str(self.run_dir.resolve()).encode()).hexdigest()[:12]
        context = WorkerContext(
            features=data.features, epsilon=settings.epsilon, domains=data.domains,
            scoring=data.scoring, sampler=DescriptorSampler.from_data(data),
            channels=data.channels, primary_roles=data.primary_roles,
            allowed_roles=data.allowed_roles, prior_weights=data.prior_weights,
            names=data.names, store_path=str(registry.path), store_shape=registry.shape)
        self.remove_stale_files()
        if settings.jobs > 1:
            self.pool = mp.get_context("fork").Pool(settings.jobs, initializer=_install,
                                                    initargs=(context,))
        else:
            self.pool = None
            _install(context)

    def map(self, function: Callable[[Any], dict[str, Any]],
            payloads: Sequence[Any]) -> list[dict[str, Any]]:
        if self.pool is None:
            return [function(p) for p in payloads]
        return list(self.pool.imap_unordered(function, payloads, chunksize=1))

    def slices(self, n: int) -> list[tuple[int, int]]:
        return batch_slices(n, effective_task_size(n, self.settings.jobs,
                                                   self.settings.task_size))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
        self.remove_stale_files()

    def temp_dir(self, required_bytes: int) -> Path:
        preferred = self.scratch or Path("/dev/shm") / f"symbolic_regression_{os.getuid()}"
        for base in (preferred, self.run_dir):
            try:
                base.mkdir(parents=True, exist_ok=True)
                if (os.access(base, os.W_OK)
                        and shutil.disk_usage(base).free > max(2 * required_bytes, 64 << 20)):
                    return base
            except OSError:
                continue
        return self.run_dir

    def temp_file(self, label: str, suffix: str, size: int) -> Path:
        fd, name = tempfile.mkstemp(prefix=f".sr_{self.tag}_{label}.", suffix=suffix,
                                    dir=str(self.temp_dir(size)))
        os.close(fd)
        return Path(name)

    def remove_stale_files(self) -> None:
        for base in {self.run_dir, self.temp_dir(1)}:
            for stale in base.glob(f".sr_{self.tag}_*"):
                try:
                    stale.unlink()
                except OSError:
                    pass

    # ---------------------------------------------------------- evaluation

    def _value_store(self, rows: int, label: str) -> tuple[Path, tuple[int, int]]:
        shape = (int(rows), self.data.n_samples)
        size = shape[0] * shape[1] * 8
        path = self.temp_file(label, ".float64.dat", size)
        with path.open("r+b") as f:
            f.truncate(size)
        return path, shape

    def _collect(self, function: Callable, payloads: list, path: Path,
                 shape: tuple[int, int]) -> list[dict[str, Any]]:
        try:
            mapped = self.map(function, payloads)
            records = [dict(row) for batch in mapped for row in batch["rows"]]
            records.sort(key=lambda r: int(r["index"]))
            store = np.memmap(path, mode="r", dtype=np.float64, shape=shape)
            for row in records:
                if row.get("status") in {"VALID", "UNSAFE"}:
                    row["values"] = store[int(row["index"])]
        finally:
            # The mapping stays readable after the name is removed.
            path.unlink(missing_ok=True)
        return records

    def evaluate_components(self, asts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        n = len(asts)
        if n == 0:
            return []
        path, shape = self._value_store(n, "components")
        payloads = [(str(path), shape, [(i, dict(asts[i])) for i in range(a, b)])
                    for a, b in self.slices(n)]
        return self._collect(_component_batch, payloads, path, shape)

    def evaluate_skeletons(self, asts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        n = len(asts)
        if n == 0:
            return []
        work = [(i, dict(ast)) for i, ast in enumerate(asts)]
        path, shape = self._value_store(n, "skeletons")
        payloads = []
        for a, b in self.slices(n):
            items = work[a:b]
            ids = sorted({cid for _, ast in items for cid in g.skeleton_component_ids(ast)})
            component_asts = {int(cid): self.registry.ast(int(cid)) for cid in ids}
            payloads.append((str(path), shape, items, component_asts, self.data.domains))
        return self._collect(_skeleton_batch, payloads, path, shape)

    # ----------------------------------------------------------- proposals

    def _merge_proposals(self, function: Callable, plans: list, wrap: Callable[[list], Any]
                         ) -> list[dict[str, Any]]:
        payloads = [wrap(plans[a:b]) for a, b in self.slices(len(plans))]
        rows = [r for batch in self.map(function, payloads) for r in batch["rows"]]
        rows.sort(key=lambda r: int(r["attempt"]))
        return rows

    def propose_components(self, population: Sequence[int], requested: int,
                           rng: np.random.Generator
                           ) -> tuple[list[dict[str, Any]], list[str], int]:
        """Mutations (80 %), crossovers (12 %) and random immigrants (8 %)."""
        registry = self.registry
        pop = list(map(int, population))
        proposals: list[dict[str, Any]] = []
        operators: list[str] = []
        attempts = 0
        max_attempts = max(requested * self.settings.max_attempt_factor, 10000)
        while len(proposals) < requested and attempts < max_attempts:
            remaining = requested - len(proposals)
            budget = min(max_attempts - attempts, max(256, int(math.ceil(remaining * 1.24))))
            seeds = rng.integers(0, UINT64_MAX, size=budget, dtype=np.uint64)
            modes = rng.random(budget)
            plans = []
            for local in range(budget):
                u = float(modes[local])
                pa = pb = None
                if u < .12 and len(pop) >= 2:
                    ai = int(rng.integers(len(pop)))
                    bi = int(rng.integers(len(pop) - 1))
                    bi += int(bi >= ai)
                    mode = "crossover"
                    pa = registry.ast(pop[ai])
                    pb = registry.ast(pop[bi])
                elif u < .92 and pop:
                    mode = "mutation"
                    pa = registry.ast(pop[int(rng.integers(len(pop)))])
                else:
                    mode = "random_immigrant"
                plans.append((attempts + local, int(seeds[local]), mode, pa, pb))
            for row in self._merge_proposals(_component_proposal_batch, plans, lambda x: x):
                if row["status"] == "VALID":
                    proposals.append(dict(row["ast"]))
                    operators.append(str(row["operator"]))
                    if len(proposals) >= requested:
                        break
            attempts += budget
        return proposals, operators, attempts

    def propose_feedback_components(self, sources: Sequence[tuple[str, Mapping[str, Any]]],
                                    target: int, rng: np.random.Generator
                                    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        """Mutations of one component of an archived skeleton, remembering the parent."""
        registry = self.registry
        target = int(target)
        max_attempts = target * int(self.settings.max_attempt_factor)
        proposals: list[dict[str, Any]] = []
        parents: list[dict[str, Any]] = []
        attempts = 0
        while len(proposals) < target and attempts < max_attempts:
            remaining = target - len(proposals)
            budget = min(max_attempts - attempts, max(512, int(math.ceil(remaining * 1.25))))
            seeds = rng.integers(0, UINT64_MAX, size=budget, dtype=np.uint64)
            plans = []
            info: dict[int, dict[str, Any]] = {}
            for local in range(budget):
                attempt = attempts + local
                label, rec = sources[int(rng.integers(len(sources)))]
                parent_ast = json.loads(str(rec["component_ast_json"]))
                ids = g.skeleton_component_ids(parent_ast)
                if not ids:
                    continue
                old = int(ids[int(rng.integers(len(ids)))])
                plans.append((attempt, int(seeds[local]), "mutation", registry.ast(old), None))
                info[attempt] = {
                    "parent_formula_id": str(rec["formula_id"]),
                    "parent_bacc": float(rec["all_bacc"]),
                    "parent_worst_recall": float(rec.get("all_worst_recall", -1.0)),
                    "parent_operator_nodes": int(rec.get("operator_nodes", 10**9)),
                    "parent_tree_depth": int(rec.get("tree_depth", 10**9)),
                    "old_component_id": old, "parent_ast": parent_ast,
                    "source_reservoir": label}
            for row in self._merge_proposals(_component_proposal_batch, plans, lambda x: x):
                if row["status"] != "VALID":
                    continue
                parent = info.get(int(row["attempt"]))
                if parent is None:
                    continue
                proposals.append(dict(row["ast"]))
                parents.append({**parent, "proposal_operator": str(row["operator"])})
                if len(proposals) >= target:
                    break
            attempts += budget
        return proposals[:target], parents[:target], attempts

    def write_snapshot(self, ids: np.ndarray, label: str) -> tuple[str, ...]:
        """Sampling arrays over ``ids`` for the proposal workers of one step."""
        rows = self.registry.rows
        ids = np.asarray(ids, dtype=np.int32)
        if ids.size == 0:
            raise RuntimeError("no component to sample from")
        weights = np.asarray([max(1e-12, float(rows[int(i)].get("rf_prior", 0.0))) for i in ids],
                             dtype=np.float64)
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        weights = weights[order]
        weights /= weights.sum()
        cdf = np.cumsum(weights, dtype=np.float64)
        cdf[-1] = 1.0
        channels = [str(rows[int(i)].get("channels", "")) for i in ids]
        codes = np.unique(np.asarray(channels, dtype=object).astype(str),
                          return_inverse=True)[1].astype(np.int64)
        paths = []
        for name, array in (("ids", ids), ("cdf", cdf), ("channels", codes)):
            path = self.temp_file(f"{label}_{name}", ".npy", int(array.nbytes))
            with path.open("wb") as f:
                np.save(f, np.asarray(array), allow_pickle=False)
            paths.append(str(path))
        return tuple(paths)

    def seed_skeletons(self, ids: np.ndarray, rng: np.random.Generator
                       ) -> tuple[list[dict[str, Any]], list[str]]:
        """The initial skeleton population, filled category by category."""
        target = int(self.settings.skeleton_population)
        quotas = seed_quotas(target)
        snapshot = self.write_snapshot(ids, "seeds")
        asts: list[dict[str, Any]] = []
        sources: list[str] = []
        seen: set[str] = set()
        attempts = 0
        max_attempts = max(target * int(self.settings.max_attempt_factor), 10000)

        def fill(category: str, wanted: int, limit: int) -> None:
            nonlocal attempts
            accepted = 0
            while accepted < wanted and attempts < limit:
                remaining = wanted - accepted
                budget = min(limit - attempts, max(512, int(math.ceil(remaining * 1.24))))
                seeds = rng.integers(0, UINT64_MAX, size=budget, dtype=np.uint64)
                plans = [(attempts + i, int(seeds[i]), category) for i in range(budget)]
                for row in self._merge_proposals(_seed_batch, plans, lambda x: (snapshot, x)):
                    if row["status"] != "VALID":
                        continue
                    can = dict(row["ast"])
                    fid = g.skeleton_id(can)
                    if fid in seen:
                        continue
                    seen.add(fid)
                    asts.append(can)
                    sources.append(category)
                    accepted += 1
                    if accepted >= wanted:
                        break
                attempts += budget

        try:
            for category, _ in SEED_CATEGORIES:
                fill(category, quotas[category], max_attempts)
            # Rare category shortfalls are filled with typed random seeds.
            fill("TYPED_RANDOM", target - len(asts), max_attempts * 2)
        finally:
            for path in snapshot:
                Path(path).unlink(missing_ok=True)
        if len(asts) < max(64, target // 2):
            raise RuntimeError(f"only {len(asts)} initial skeletons after {attempts} attempts")
        return asts[:target], sources[:target]

    def propose_skeletons(self, population: Sequence[Mapping[str, Any]], ids: np.ndarray,
                          rng: np.random.Generator, label: str, *, joint: bool
                          ) -> tuple[list[dict[str, Any]], list[str], int]:
        """Mutations (83 %), crossovers (12 %) and new typed skeletons (5 %)."""
        target = int(self.settings.skeleton_population)
        max_attempts = target * int(self.settings.max_attempt_factor)
        proposals: list[dict[str, Any]] = []
        operators: list[str] = []
        attempts = 0
        snapshot = self.write_snapshot(ids, label)
        try:
            pop = [dict(x) for x in population]
            while len(proposals) < target and attempts < max_attempts:
                remaining = target - len(proposals)
                budget = min(max_attempts - attempts, max(512, int(math.ceil(remaining * 1.24))))
                seeds = rng.integers(0, UINT64_MAX, size=budget, dtype=np.uint64)
                modes = rng.random(budget)
                plans = []
                for local in range(budget):
                    u = float(modes[local])
                    pa = pb = None
                    if u < .12 and len(pop) >= 2:
                        ia = int(rng.integers(len(pop)))
                        ib = int(rng.integers(len(pop) - 1))
                        ib += int(ib >= ia)
                        mode = "crossover"
                        pa = pop[ia]
                        pb = pop[ib]
                    elif u < .95 and pop:
                        mode = "mutation"
                        pa = pop[int(rng.integers(len(pop)))]
                    else:
                        mode = "de_novo"
                    plans.append((attempts + local, int(seeds[local]), mode, pa, pb))
                rows = self._merge_proposals(_skeleton_proposal_batch, plans,
                                             lambda x: (snapshot, bool(joint), x))
                for row in rows:
                    if row["status"] == "VALID":
                        proposals.append(dict(row["ast"]))
                        operators.append(str(row["operator"]))
                        if len(proposals) >= target:
                            break
                attempts += budget
            if len(proposals) < max(64, target // 2):
                raise RuntimeError(f"only {len(proposals)} skeleton proposals")
        finally:
            for path in snapshot:
                Path(path).unlink(missing_ok=True)
        return proposals[:target], operators[:target], attempts

    # ------------------------------------------------------------- context

    def probe_context(self, candidate_ids: Sequence[int], anchor_ids: Sequence[int],
                      rng: np.random.Generator) -> dict[str, Any]:
        """Measure how much each candidate component helps a panel of anchors.

        Every candidate is combined with anchors by add, sub, mul and
        saturation; its utility is the best BACC gain over the anchor alone.
        A screen picks a few promising anchors per candidate, and a subset of
        candidates is then probed against the whole panel.  The results are
        written into the registry rows.
        """
        settings = self.settings
        registry = self.registry
        cids = list(dict.fromkeys(map(int, candidate_ids)))[:settings.component_context_candidates]
        if not cids:
            return {"expressions": 0, "best_gain": 0.0}
        panel, sources = anchor_panel(registry, anchor_ids, settings.component_context_anchors,
                                      rng)
        if not panel:
            return {"expressions": 0, "best_gain": 0.0}
        plan = screen_anchor_plan(registry, cids, panel, self.data,
                                  settings.component_context_screen_anchors, rng)
        baseline = {int(i): float(registry.rows[int(i)].get("standalone_bacc", 0.0))
                    for i in panel}

        def payloads(ids: Sequence[int], anchors: np.ndarray, level: str,
                     block: int) -> list[tuple]:
            out = []
            block = max(1, int(block))
            for a in range(0, len(ids), block):
                b = min(len(ids), a + block)
                sub = anchors[a:b]
                labels = [[str(sources.get(int(x), "UNCLASSIFIED")) for x in row] for row in sub]
                out.append((list(map(int, ids[a:b])), sub.astype(int).tolist(), labels, level,
                            int(settings.component_context_anchor_block), baseline))
            return out

        mapped = self.map(_context_batch, payloads(cids, plan, "SCREEN_EXACT",
                                                   settings.component_context_candidate_block))
        screened = {int(r["component_id"]): r for batch in mapped for r in batch["rows"]}
        expressions = sum(int(batch["expressions"]) for batch in mapped)
        final = dict(screened)
        full_ids = select_full_audit(registry, cids, screened,
                                     settings.component_context_full_audit_candidates, rng)
        if full_ids:
            full_plan = np.tile(np.asarray(panel, dtype=np.int32), (len(full_ids), 1))
            full = self.map(_context_batch, payloads(
                full_ids, full_plan, "FULL_AUDIT",
                max(1, min(settings.component_context_candidate_block, 2))))
            expressions += sum(int(batch["expressions"]) for batch in full)
            for batch in full:
                for r in batch["rows"]:
                    final[int(r["component_id"])] = r
        for cid, row in final.items():
            r = registry.rows[cid]
            r["context_utility"] = float(row["gain"])
            r["context_best_bacc"] = float(row["bacc"])
            r["context_probe_count"] = int(row["probe_count"])
            r["context_anchor_component_id"] = int(row["anchor_id"])
            r["context_operator"] = str(row["operator"])
            r["context_top4_mean_gain"] = float(row["top4_mean_gain"])
            r["context_top8_median_gain"] = float(row["top8_median_gain"])
            r["context_positive_rate"] = float(row["positive_context_rate"])
            r["context_helpful_anchor_count"] = int(row["helpful_anchor_count"])
            r["context_helpful_operator_count"] = int(row["helpful_operator_count"])
            r["context_anchor_source_diversity"] = int(row["anchor_source_diversity"])
            r["context_audit_level"] = str(row["audit_level"])
        return {"expressions": expressions,
                "best_gain": max((float(r["gain"]) for r in final.values()), default=0.0)}
