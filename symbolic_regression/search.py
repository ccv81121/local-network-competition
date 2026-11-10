"""Two-level component-skeleton coevolutionary search for one scalar descriptor.

The run has four phases, each ending in a checkpoint:

1. an initial component universe (every descriptor, its powers and random
   pairs), probed for context utility;
2. component evolution;
3. skeleton evolution of a protected baseline lineage, with periodic
   feedback cycles that mutate components inside good skeletons and let
   improved skeletons migrate back;
4. finalization: re-evaluation of the archive leaders, one deletion pass
   for simpler equivalents, and the ranked top list.

Independent random streams drive the phases, so a resumed run continues
exactly where the checkpoint left it.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import __version__
from . import grammar as g
from .components import (ComponentRegistry, DescriptorSampler, choose_component_population,
                         feedback_room, initial_components, ordinary_room,
                         select_injections, select_new_components)
from .data import SearchData, load_data
from .metrics import (GROUPED_OOF, grouped_accuracy, grouped_oof_metrics,
                      natural_bacc_resolution, parameter_key, performance_key, safe_auc)
from .parallel import Workers
from .settings import Settings
from .skeletons import (SEED_CATEGORIES, Archives, archive_record, crossover_skeletons,
                        feedback_gate, final_candidates, mutate_skeleton, parameter_neighbors,
                        select_migrations, select_side_population, select_skeleton_population,
                        simplification_candidates)

RNG_OFFSETS = {"initial_component": 101, "component": 211, "seed_skeleton": 307,
               "skeleton": 401, "feedback": 503, "feedback_side": 557, "beam": 601}
EVALUATED = {"VALID", "UNSAFE"}
CHECKPOINT = "checkpoint.pkl"
RUN_RECORD = "run_record.json"


class StopRequested(Exception):
    """Raised after the checkpoint of the generation the caller asked to stop at."""


def log(message: str) -> None:
    print(message, flush=True)


# ------------------------------------------------------------------- files


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                                allow_nan=False) + "\n")


def write_csv(path: Path, frame: pd.DataFrame | Sequence[Mapping[str, Any]]) -> None:
    frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(list(frame))
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n",
                     float_format="%.17g", quoting=csv.QUOTE_MINIMAL)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def rng_streams(seed: int) -> dict[str, np.random.Generator]:
    return {name: np.random.default_rng(np.random.SeedSequence([seed, offset]))
            for name, offset in RNG_OFFSETS.items()}


def save_checkpoint(outdir: Path, phase: str, settings: Settings, registry: ComponentRegistry,
                    rngs: Mapping[str, np.random.Generator], state: Mapping[str, Any]) -> None:
    payload = {"phase": phase, "settings": settings.scientific(), "registry": registry.state(),
               "rngs": {k: v.bit_generator.state for k, v in rngs.items()}, "state": dict(state)}
    registry.flush()
    fd, tmp = tempfile.mkstemp(prefix=".checkpoint.", dir=str(outdir))
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(payload, f, protocol=5)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, outdir / CHECKPOINT)
    finally:
        Path(tmp).unlink(missing_ok=True)


def package_fingerprint() -> str:
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def software_versions() -> dict[str, str]:
    return {"package": f"symbolic_regression {__version__}",
            "source_sha256": package_fingerprint(),
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "platform": platform.platform()}


# ------------------------------------------------------- component evolution


def evolve_components(registry: ComponentRegistry, workers: Workers, settings: Settings,
                      rng: np.random.Generator, population: Sequence[int],
                      history: list[dict[str, Any]], operator_history: list[dict[str, Any]],
                      checkpoint: Callable[[list[int]], None]) -> list[int]:
    """Evolve components until the best context utility stops improving."""
    pop = list(population)
    stale = int(history[-1].get("early_stop_stale", 0)) if history else 0
    best_context = max((float(registry.rows[i].get("context_utility", 0.0)) for i in pop),
                       default=-np.inf)
    start = int(history[-1]["generation"]) + 1 if history else 1
    requested = settings.component_population
    for gen in range(start, settings.component_max_generations + 1):
        t0 = time.perf_counter()
        proposals, operators, attempts = workers.propose_components(pop, requested, rng)
        if len(proposals) < max(64, requested // 4):
            raise RuntimeError(f"component generation {gen}: only {len(proposals)} proposals")
        results = workers.evaluate_components(proposals)
        limit = min(settings.component_persist_per_generation, ordinary_room(registry, settings))
        selected = select_new_components(results, registry, max(0, limit), rng)

        by_op = {op: _operator_row(gen, op)
                 for op in ("crossover", "mutation", "random_immigrant")}
        formulas: dict[str, set[str]] = {op: set() for op in by_op}
        for ast, op in zip(proposals, operators):
            by_op.setdefault(op, _operator_row(gen, op))
            by_op[op]["proposed"] += 1
            formulas.setdefault(op, set()).add(g.tree_id(ast))
        for op, values in formulas.items():
            by_op[op]["formula_unique"] = len(values)
        for result, op in zip(results, operators):
            by_op[op]["evaluated"] += 1
            by_op[op]["safe_numeric"] += int(result.get("status") == "VALID")
        accepted = []
        status_counts: dict[str, int] = {}
        for x in selected:
            op = operators[int(x["index"])]
            by_op[op]["selected"] += 1
            cid, status = registry.add(x, source="COMPONENT_EVOLUTION", generation=gen,
                                       proposal_operator=op,
                                       island=len(registry) % settings.component_islands)
            status_counts[status] = status_counts.get(status, 0) + 1
            if cid is not None and status == "ACCEPTED":
                accepted.append(cid)
                by_op[op]["persisted"] += 1
        registry.flush()

        anchors = choose_component_population(
            registry, min(settings.component_context_anchors, len(registry)), rng)
        candidates = accepted[:settings.component_context_candidates]
        context = workers.probe_context(candidates, anchors, rng)
        for cid in candidates:
            by_op[str(registry.rows[cid]["proposal_operator"])]["context_probed"] += 1
        pop = choose_component_population(registry, settings.component_population, rng)
        members = set(pop)
        for cid in accepted:
            op = str(registry.rows[cid]["proposal_operator"])
            by_op[op]["survived"] += int(cid in members)
            if float(registry.rows[cid].get("context_utility", 0.0)) > best_context:
                by_op[op]["improved_context"] += 1
        new_best = max((float(registry.rows[i].get("context_utility", 0.0)) for i in pop),
                       default=best_context)
        reset = bool(new_best > best_context + 1e-15)
        best_context = max(best_context, new_best)
        stale = 0 if reset else stale + 1
        history.append({
            "generation": gen, "attempts": attempts, "requested": requested,
            "generated": len(proposals), "evaluated": len(results),
            "new_components": len(accepted), "registry_size": len(registry),
            "best_standalone_bacc": max(float(registry.rows[i]["standalone_bacc"]) for i in pop),
            "best_context_utility": best_context, "early_stop_reset": reset,
            "early_stop_stale": stale, "context_candidates": len(candidates),
            "context_expressions": context["expressions"],
            "context_best_gain": context["best_gain"],
            "distinct_structures": len({registry.rows[i]["structure_id"] for i in pop}),
            "distinct_responses": len({registry.rows[i]["continuous_response_id"] for i in pop}),
            **{f"status_{k.lower()}": v for k, v in status_counts.items()},
            "seconds": time.perf_counter() - t0})
        operator_history.extend(by_op.values())
        log(f"component generation {gen}: {len(results)} evaluated, {len(accepted)} accepted, "
            f"registry {len(registry)}, best context gain {best_context:.6f}")
        checkpoint(pop)
        if gen >= settings.component_min_generations and stale >= settings.component_patience:
            break
    return pop


def _operator_row(gen: int, op: str) -> dict[str, Any]:
    return {"generation": gen, "proposal_operator": op, "proposed": 0, "formula_unique": 0,
            "evaluated": 0, "safe_numeric": 0, "selected": 0, "persisted": 0,
            "context_probed": 0, "survived": 0, "improved_context": 0}


# ------------------------------------------------------------------- feedback


def create_feedback_components(archives: Archives, registry: ComponentRegistry,
                               workers: Workers, settings: Settings, rng: np.random.Generator,
                               generation: int) -> tuple[list[int], list[dict[str, Any]],
                                                         dict[str, Any]]:
    """Mutate components inside archived skeletons and rebuild those skeletons
    with the new components."""
    funnel = {"generation": generation, "proposed": 0, "persisted": 0, "injected": 0}
    sources = []
    for rec in archives.feedback_sources():
        label = ("FEEDBACK_ARCHIVE" if str(rec.get("source", "")) in
                 {"feedback_migrated", "feedback_delayed_migrated"} else "PROTECTED_BASELINE")
        sources.append((label, rec))
        if len(sources) >= max(64, settings.feedback_component_proposals // 4):
            break
    if not sources:
        return [], [], funnel
    proposals, parents, attempts = workers.propose_feedback_components(
        sources, settings.feedback_component_proposals, rng)
    results = workers.evaluate_components(proposals)
    limit = min(feedback_room(registry, settings), settings.feedback_component_persist)
    selected = {int(x["index"]) for x in select_new_components(results, registry, limit, rng)}
    new_ids: list[int] = []
    parent_of: dict[int, dict[str, Any]] = {}
    status_counts: dict[str, int] = {}
    # Persisted in proposal order, not in selection order.
    for idx, (result, parent) in enumerate(zip(results, parents)):
        if idx not in selected:
            continue
        cid, status = registry.add(result, source="SKELETON_FEEDBACK", generation=generation,
                                   proposal_operator=str(parent["proposal_operator"]),
                                   island=len(registry) % settings.component_islands)
        status_counts[status] = status_counts.get(status, 0) + 1
        if cid is not None and status == "ACCEPTED":
            new_ids.append(int(cid))
            parent_of[int(cid)] = dict(parent)
    registry.flush()
    context = {"expressions": 0, "best_gain": 0.0}
    if new_ids:
        anchors = choose_component_population(
            registry, min(settings.component_context_anchors, len(registry)), rng)
        context = workers.probe_context(new_ids[:settings.component_context_candidates],
                                        anchors, rng)
    injected = []
    for cid in select_injections(registry, new_ids, settings.feedback_component_inject, rng):
        info = parent_of.get(int(cid))
        if info is None:
            continue
        parent_ast = dict(info["parent_ast"])
        old = int(info["old_component_id"])
        leaves = [p for p in g.node_paths(parent_ast)
                  if g.node_at(parent_ast, p)["op"] == "component"
                  and int(g.node_at(parent_ast, p)["component_id"]) == old]
        if not leaves:
            continue
        try:
            candidate = g.canonicalize_skeleton(
                g.replace_node(parent_ast, leaves[0], g.component_leaf(cid)))
        except Exception:
            continue
        injected.append({
            "ast": candidate, "source": "feedback_injection",
            "parent_formula_id": str(info["parent_formula_id"]),
            "parent_bacc": float(info["parent_bacc"]),
            "parent_worst_recall": float(info.get("parent_worst_recall", -1.0)),
            "parent_operator_nodes": int(info.get("parent_operator_nodes", 10**9)),
            "parent_tree_depth": int(info.get("parent_tree_depth", 10**9)),
            "old_component_id": old, "new_component_id": int(cid),
            "proposal_operator": str(info["proposal_operator"]),
            "source_reservoir": str(info["source_reservoir"])})
    funnel.update({
        "attempts": int(attempts), "proposed": len(proposals),
        "valid": sum(1 for r in results if r.get("status") == "VALID"),
        "evaluated": len(results), "selected_for_persistence": len(selected),
        "persisted": len(new_ids), "context_exact_expressions": int(context["expressions"]),
        "context_best_gain": float(context["best_gain"]),
        "injection_budget": int(settings.feedback_component_inject), "injected": len(injected),
        "registry_size": len(registry),
        "status_counts": json.dumps(status_counts, sort_keys=True)})
    return new_ids, injected, funnel


def _feedback_record(result: Mapping[str, Any], meta: Mapping[str, Any],
                     registry: ComponentRegistry, source: str, generation: int,
                     delay: int) -> tuple[dict[str, Any], dict[str, Any]]:
    migrate, reason, delta = feedback_gate(result, meta)
    rec = archive_record(result, registry.rows, registry.data.names, source, generation, -1)
    rec.update({"feedback_migrate": bool(migrate), "feedback_gate_reason": reason,
                "feedback_parent_formula_id": str(meta["parent_formula_id"]),
                "feedback_parent_bacc": float(meta["parent_bacc"]),
                "feedback_delta_bacc": float(delta), "feedback_delay_generation": delay,
                "feedback_root_source": str(meta.get("source_reservoir", ""))})
    pair = {"generation": generation, "delay_generation": delay,
            "parent_formula_id": str(meta["parent_formula_id"]),
            "parent_bacc": float(meta["parent_bacc"]), "child_formula_id": str(rec["formula_id"]),
            "child_bacc": float(rec["all_bacc"]), "paired_delta_bacc": float(delta),
            "gate_reason": reason, "migrated_eligible": bool(migrate),
            "old_component_id": int(meta.get("old_component_id", -1)),
            "new_component_id": int(meta.get("new_component_id", -1)),
            "mutation_operator": str(meta.get("proposal_operator", "")),
            "source_reservoir": str(meta.get("source_reservoir", "")),
            "safe": bool(rec.get("safe", False))}
    return rec, pair


def run_feedback_side_branch(injected: Sequence[Mapping[str, Any]], registry: ComponentRegistry,
                             workers: Workers, settings: Settings, rng: np.random.Generator,
                             generation: int, key: Callable
                             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                        dict[str, Any], list[dict[str, Any]]]:
    """Evaluate feedback skeletons apart from the baseline lineage and pick
    the ones allowed to migrate."""
    stats: dict[str, Any] = {"generation": generation, "paired_evaluated": 0, "migrated": 0}
    if not injected:
        return [], [], stats, []
    domains = registry.data.domains
    records: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    meta_of: dict[str, dict[str, Any]] = {}
    direct = workers.evaluate_skeletons([dict(x["ast"]) for x in injected])
    for r, meta0 in zip(direct, injected):
        if r.get("status") not in EVALUATED:
            continue
        meta = dict(meta0)
        rec, pair = _feedback_record(r, meta, registry, "feedback_side_direct", generation, 0)
        records.append(rec)
        pairs.append(pair)
        meta_of[str(rec["formula_id"])] = meta
    population = select_side_population(records, settings.feedback_side_population, rng, key)
    for delay in range(1, settings.feedback_side_generations + 1):
        if not population:
            break
        target = settings.feedback_side_population
        proposals: list[dict[str, Any]] = []
        metas: list[dict[str, Any]] = []
        seen: set[str] = set()
        attempts = 0
        while len(proposals) < target and attempts < target * settings.max_attempt_factor:
            attempts += 1
            if len(population) >= 2 and rng.random() < .15:
                ia, ib = rng.choice(len(population), size=2, replace=False)
                a = population[int(ia)]
                b = population[int(ib)]
                ast = crossover_skeletons(json.loads(a["component_ast_json"]),
                                          json.loads(b["component_ast_json"]), rng)
                root = a if key(a) <= key(b) else b
                op = "feedback_side_crossover"
            else:
                root = population[int(rng.integers(len(population)))]
                ast, mutation = mutate_skeleton(json.loads(root["component_ast_json"]), rng,
                                                registry, joint=True)
                op = f"feedback_side_{mutation}"
            try:
                can, _, _ = g.validate_skeleton(ast, registry.ast, domains)
            except Exception:
                continue
            fid = g.skeleton_id(can)
            if fid in seen:
                continue
            seen.add(fid)
            proposals.append(can)
            root_meta = meta_of.get(str(root["formula_id"]), {
                "parent_formula_id": str(root.get("feedback_parent_formula_id",
                                                  root["formula_id"])),
                "parent_bacc": float(root.get("feedback_parent_bacc", root["all_bacc"])),
                "parent_worst_recall": float(root.get("all_worst_recall", -1.0)),
                "parent_operator_nodes": int(root.get("operator_nodes", 10**9)),
                "parent_tree_depth": int(root.get("tree_depth", 10**9)),
                "source_reservoir": "FEEDBACK_SIDE"})
            metas.append({**root_meta, "proposal_operator": op, "old_component_id": -1,
                          "new_component_id": -1})
        new_records = []
        for r, meta in zip(workers.evaluate_skeletons(proposals), metas):
            if r.get("status") not in EVALUATED:
                continue
            rec, pair = _feedback_record(r, meta, registry, "feedback_side_descendant",
                                         generation, delay)
            new_records.append(rec)
            pairs.append(pair)
            meta_of[str(rec["formula_id"])] = dict(meta)
        records.extend(new_records)
        population = select_side_population(population + new_records,
                                            settings.feedback_side_population, rng, key)
    migrations = select_migrations(records, settings.feedback_component_migrate, rng, key)
    migrated = {str(r["formula_id"]) for r in migrations}
    for pair in pairs:
        pair["migrated"] = str(pair["child_formula_id"]) in migrated
    stats.update({
        "paired_evaluated": len(pairs), "side_records": len(records),
        "eligible": sum(1 for r in records if bool(r.get("feedback_migrate", False))),
        "migrated": len(migrations)})
    return records, migrations, stats, pairs


# -------------------------------------------------------- skeleton evolution


def beam_refine(records: Sequence[Mapping[str, Any]], workers: Workers, settings: Settings,
                rng: np.random.Generator, key: Callable
                ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Try parameter variants of the best structures and keep the best variant of each."""
    safe = [dict(r) for r in records if bool(r.get("safe"))]
    if not safe:
        return [], {"proposed": 0, "formula_unique": 0, "evaluated": 0, "safe": 0, "selected": 0}
    safe.sort(key=key)
    representatives = []
    seen: set[str] = set()
    for r in safe:
        sid = str(r["structure_id"])
        if sid not in seen:
            seen.add(sid)
            representatives.append(r)
        if len(representatives) >= settings.beam_topologies:
            break
    asts = []
    parents = []
    for r in representatives:
        base = r["component_ast"] if "component_ast" in r else json.loads(r["component_ast_json"])
        for ast in parameter_neighbors(base, settings.beam_width, rng):
            asts.append(ast)
            parents.append(r["structure_id"])
    unique = len({g.skeleton_id(ast) for ast in asts})
    evaluated = workers.evaluate_skeletons(asts)
    best: dict[str, dict[str, Any]] = {}
    for r, parent in zip(evaluated, parents):
        if r.get("status") not in EVALUATED:
            continue
        candidate = dict(r)
        candidate["_proposal_source"] = "beam_refine"
        candidate["_parent_structure_id"] = str(parent)
        old = best.get(parent)
        if old is None or parameter_key(candidate) < parameter_key(old):
            best[parent] = candidate
    rows = list(best.values())
    return rows, {"proposed": len(asts), "formula_unique": unique, "evaluated": len(evaluated),
                  "safe": sum(1 for r in evaluated if bool(r.get("safe"))),
                  "selected": len(rows)}


def _island(basis: str, islands: int) -> int:
    return int(hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16], 16) % islands


def evolve_skeletons(registry: ComponentRegistry, workers: Workers, settings: Settings,
                     rngs: Mapping[str, np.random.Generator], state: dict[str, Any],
                     checkpoint: Callable[[], None], stop_after: int | None = None) -> None:
    """Skeleton evolution; ``state`` holds the population, archives and histories."""
    key = partial(performance_key, mode=settings.scoring)
    names = registry.data.names
    rng_skeleton = rngs["skeleton"]
    archives = Archives((settings.performance_archive_cap, settings.topology_archive_cap,
                         settings.prediction_archive_cap), settings.scoring)
    archives.restore(state["archives"])
    history = state["skeleton_history"]
    operator_history = state["skeleton_operator_history"]
    coevolution = state["feedback_history"]
    all_pairs = state["feedback_pairs"]
    start = int(history[-1]["generation"]) + 1 if history else 1
    if start > 1:
        population = [g.canonicalize_skeleton(x) for x in state["skeleton_population"]]
        initial_seed_best = float(history[0].get("initial_seed_best", -1.0))
    else:
        results = workers.evaluate_skeletons(state["skeleton_seeds"])
        initial = [archive_record(r, registry.rows, names, src, 0,
                                  int(r["index"]) % settings.skeleton_islands)
                   for r, src in zip(results, state["skeleton_seed_sources"])
                   if r.get("status") in EVALUATED]
        archives.add_baseline(initial)
        population = select_skeleton_population(initial, settings.skeleton_population,
                                                rng_skeleton, key)
        initial_seed_best = max((float(r["all_bacc"]) for r in initial if bool(r["safe"])),
                                default=-1.0)

    last = history[-1] if history else {}
    base_record = archives.best("baseline_")
    merged_record = archives.best()
    base_best = max(float(base_record["all_bacc"]) if base_record else -1.0,
                    float(last.get("base_best_bacc", -1.0)))
    merged_best = max(float(merged_record["all_bacc"]) if merged_record else -1.0,
                      float(last.get("merged_best_bacc", -1.0)))
    base_worst = max(float(base_record["all_worst_recall"]) if base_record else -1.0,
                     float(last.get("base_best_worst_recall", -1.0)))
    merged_worst = max(float(merged_record["all_worst_recall"]) if merged_record else -1.0,
                       float(last.get("merged_best_worst_recall", -1.0)))
    stale = int(last.get("early_stop_stale", 0))
    near_topologies = max([len(archives.baseline_topology)]
                          + [int(x.get("base_near_optimal_topologies", 0)) for x in history])
    near_predictions = max([len(archives.baseline_prediction)]
                           + [int(x.get("base_near_optimal_predictions", 0)) for x in history])
    saturation_stale = int(last.get("saturation_performance_stale", 0))
    saturation_cycles = int(last.get("saturation_feedback_cycles", 0))
    saturation_trigger = int(last.get("saturation_trigger_generation", -1))
    saturation_end = int(last.get("saturation_consolidation_end_generation", -1))
    # The baseline lineage samples only ordinary components, so feedback
    # never alters it directly.
    baseline_ids = np.asarray([int(r["component_id"]) for r in registry.rows
                               if str(r.get("source", "")) != "SKELETON_FEEDBACK"], dtype=np.int32)
    seed_sources = {name for name, _ in SEED_CATEGORIES}

    for gen in range(start, settings.skeleton_max_generations + 1):
        t0 = time.perf_counter()
        consolidating = bool(settings.saturation_stop and saturation_trigger >= 0
                             and gen > saturation_trigger)
        joint = bool(consolidating or gen > max(settings.skeleton_min_generations,
                                                settings.skeleton_max_generations
                                                - settings.joint_final_generations))
        stage = ("SATURATION_CONSOLIDATION" if consolidating else
                 ("JOINT_FINAL" if joint else
                  ("WARMUP" if gen <= settings.skeleton_warmup_generations else "COEVOLUTION")))

        proposals, operators, attempts = workers.propose_skeletons(
            population, baseline_ids, rng_skeleton, f"g{gen}", joint=joint)
        unique, unique_ops = [], []
        seen: set[str] = set()
        for ast, op in zip(proposals, operators):
            fid = g.skeleton_id(ast)
            if fid in seen:
                continue
            seen.add(fid)
            unique.append(ast)
            unique_ops.append(op)
        evaluated = workers.evaluate_skeletons(unique)
        for r in evaluated:
            r["_proposal_source"] = unique_ops[int(r["index"])]
        beam_rows, beam_stats = beam_refine(evaluated, workers, settings, rngs["beam"], key)
        baseline_records = []
        for r in [r for r in evaluated if r.get("status") in EVALUATED] + beam_rows:
            basis = str(r.get("_parent_structure_id", r.get("component_level_formula_id")))
            baseline_records.append(archive_record(
                r, registry.rows, names, str(r.get("_proposal_source", "beam_refine")), gen,
                _island(basis, settings.skeleton_islands)))
        prior_base = base_best
        prior_merged = merged_best
        prior_merged_worst = merged_worst
        base_entered = archives.add_baseline(baseline_records)
        population = select_skeleton_population(
            baseline_records + list(archives.baseline_performance.values()),
            settings.skeleton_population, rng_skeleton, key)
        base_record = archives.best("baseline_")
        current_base = float(base_record["all_bacc"]) if base_record else base_best
        current_base_worst = float(base_record["all_worst_recall"]) if base_record else base_worst
        topologies = len(archives.baseline_topology)
        predictions = len(archives.baseline_prediction)
        reset = bool(current_base > base_best + 1e-15 or current_base_worst > base_worst + 1e-15
                     or topologies > near_topologies or predictions > near_predictions)
        base_best = max(base_best, current_base)
        base_worst = max(base_worst, current_base_worst)
        near_topologies = max(near_topologies, topologies)
        near_predictions = max(near_predictions, predictions)
        stale = 0 if reset else stale + 1

        feedback_new: list[int] = []
        injected: list[dict[str, Any]] = []
        funnel: dict[str, Any] = {"generation": gen, "proposed": 0, "persisted": 0, "injected": 0}
        side_stats: dict[str, Any] = {"generation": gen, "paired_evaluated": 0, "migrated": 0}
        side_records: list[dict[str, Any]] = []
        migrations: list[dict[str, Any]] = []
        pairs: list[dict[str, Any]] = []
        migrated_entered: set[str] = set()
        feedback_cycle = bool(gen > settings.skeleton_warmup_generations
                              and gen % settings.feedback_interval == 0)
        if feedback_cycle:
            budget_scale = 1.0
            cycle_settings = settings
            recent = coevolution[-2:]
            # Two unproductive cycles in a row halve the next cycle's budget.
            if len(recent) == 2 and all(float(x.get("positive_paired_gain_rate", 1.0)) < .01
                                        and int(x.get("migrated", 1)) == 0 for x in recent):
                budget_scale = .5
                s = settings
                cycle_settings = replace(
                    s, feedback_component_proposals=max(64, s.feedback_component_proposals // 2),
                    feedback_component_persist=max(32, s.feedback_component_persist // 2),
                    feedback_component_inject=max(16, s.feedback_component_inject // 2),
                    feedback_component_migrate=max(8, s.feedback_component_migrate // 2),
                    feedback_side_population=max(32, s.feedback_side_population // 2))
            feedback_new, injected, funnel = create_feedback_components(
                archives, registry, workers, cycle_settings, rngs["feedback"], gen)
            funnel["budget_scale"] = budget_scale
            side_records, migrations, side_stats, pairs = run_feedback_side_branch(
                injected, registry, workers, cycle_settings, rngs["feedback_side"], gen, key)
            archives.add_feedback_side(side_records)
            migrated_entered = archives.add_migrations(migrations)
            # Migrated feedback skeletons may seed the next baseline generation.
            if migrations:
                population = select_skeleton_population(
                    baseline_records + list(migrations)
                    + list(archives.baseline_performance.values())
                    + list(archives.feedback_performance.values()),
                    settings.skeleton_population, rng_skeleton, key)

        merged_record = archives.best()
        current_merged = float(merged_record["all_bacc"]) if merged_record else base_best
        current_merged_worst = (float(merged_record["all_worst_recall"]) if merged_record
                                else base_worst)
        merged_best = max(merged_best, current_merged, base_best)
        merged_worst = max(merged_worst, current_merged_worst, base_worst)

        improved = bool(merged_best > prior_merged + 1e-15
                        or merged_worst > prior_merged_worst + 1e-15)
        saturation_stale = 0 if improved else saturation_stale + 1
        if feedback_cycle:
            saturation_cycles += 1
        cancelled = False
        if consolidating and improved:
            log(f"skeleton generation {gen}: improvement during consolidation, search continues")
            saturation_trigger = saturation_end = -1
            cancelled = True
        triggered = False
        if (settings.saturation_stop and saturation_trigger < 0 and not cancelled
                and gen >= settings.saturation_min_generations
                and saturation_cycles >= settings.saturation_min_feedback_cycles
                and saturation_stale >= settings.saturation_patience):
            saturation_trigger = gen
            saturation_end = min(settings.skeleton_max_generations,
                                 gen + settings.saturation_consolidation_generations)
            triggered = True
            log(f"skeleton generation {gen}: no improvement for {saturation_stale} generations, "
                f"consolidating until generation {saturation_end}")

        by_op = {}
        for op in sorted(set(operators + unique_ops)):
            by_op[op] = {"generation": gen, "stage": stage, "operator": op,
                         "proposed": operators.count(op), "formula_unique": unique_ops.count(op),
                         "evaluated": 0, "safe": 0, "selected_after_beam": 0, "survived": 0,
                         "archive": 0, "improved": 0}
        by_op["beam_refine"] = {"generation": gen, "stage": stage, "operator": "beam_refine",
                                "proposed": beam_stats["proposed"],
                                "formula_unique": beam_stats["formula_unique"],
                                "evaluated": beam_stats["evaluated"], "safe": beam_stats["safe"],
                                "selected_after_beam": beam_stats["selected"], "survived": 0,
                                "archive": 0, "improved": 0}
        members = {g.skeleton_id(x) for x in population}
        for r in baseline_records:
            op = str(r["source"]) if str(r["source"]) in by_op else "beam_refine"
            if op != "beam_refine":
                by_op[op]["evaluated"] += 1
                by_op[op]["safe"] += int(bool(r["safe"]))
            by_op[op]["survived"] += int(r["component_level_formula_id"] in members)
            by_op[op]["archive"] += int(r["formula_id"] in base_entered)
            by_op[op]["improved"] += int(float(r["all_bacc"]) > prior_base + 1e-15)
        operator_history.extend(by_op.values())

        gains = [(int(x["delay_generation"]), float(x["paired_delta_bacc"])) for x in pairs]
        positive = sum(1 for _, delta in gains if delta > 0)
        direct_positive = sum(1 for delay, delta in gains if delay == 0 and delta > 0)
        delayed_positive = sum(1 for delay, delta in gains if delay > 0 and delta > 0)
        feedback_best = archives.best("feedback_")
        history.append({
            "generation": gen, "stage": stage, "attempts": attempts, "evaluated": len(evaluated),
            "new_components": len(feedback_new), "feedback_injected_records": len(injected),
            "component_registry_size": len(registry),
            "distinct_formulas": len({r["formula_id"] for r in baseline_records}),
            "distinct_topologies": len({r["topology_id"] for r in baseline_records}),
            "distinct_predictions": len({r["prediction_id"] for r in baseline_records}),
            "near_optimal_topologies": len(archives.topology),
            "near_optimal_predictions": len(archives.prediction),
            "base_near_optimal_topologies": topologies,
            "base_near_optimal_predictions": predictions,
            "early_stop_reset": reset, "early_stop_stale": stale,
            "best_bacc": merged_best, "best_worst_recall": merged_worst,
            "base_best_bacc": base_best, "base_best_worst_recall": base_worst,
            "merged_best_bacc": merged_best, "merged_best_worst_recall": merged_worst,
            "feedback_side_best_bacc": float(feedback_best["all_bacc"]) if feedback_best else -1.0,
            "baseline_component_count": len(baseline_ids),
            "best_leaves": int(merged_record["component_leaf_count"]) if merged_record else -1,
            "best_raw_features": int(merged_record["raw_feature_count"]) if merged_record else -1,
            "best_expanded_operator_nodes": (int(merged_record["operator_nodes"])
                                             if merged_record else -1),
            "best_expanded_tree_depth": int(merged_record["tree_depth"]) if merged_record else -1,
            "initial_seed_best": initial_seed_best,
            "seed_outside_best": max((float(r["all_bacc"]) for r in baseline_records
                                      if r["source"] not in seed_sources and bool(r["safe"])),
                                     default=-1.0),
            "feedback_proposed": int(funnel.get("proposed", 0)),
            "feedback_persisted": int(funnel.get("persisted", 0)),
            "feedback_paired_evaluated": len(pairs), "feedback_migrated": len(migrations),
            "feedback_direct_positive": direct_positive,
            "feedback_delayed_positive": delayed_positive,
            "feedback_positive_paired_gain": positive,
            "feedback_positive_paired_gain_rate": float(positive / max(1, len(pairs))),
            "feedback_context_exact_expressions": int(funnel.get("context_exact_expressions", 0)),
            "saturation_performance_reset": improved,
            "saturation_performance_stale": saturation_stale,
            "saturation_feedback_cycles": saturation_cycles,
            "saturation_triggered_now": triggered, "saturation_cancelled_now": cancelled,
            "saturation_trigger_generation": saturation_trigger,
            "saturation_consolidation_end_generation": saturation_end,
            "seconds": time.perf_counter() - t0})
        if feedback_new or injected or side_records:
            coevolution.append({
                **funnel, **side_stats, "generation": gen, "new_components": len(feedback_new),
                "feedback_injected_records": len(injected), "paired_evaluated": len(pairs),
                "positive_paired_gain": positive,
                "positive_paired_gain_rate": float(positive / max(1, len(pairs))),
                "direct_positive": direct_positive, "delayed_positive": delayed_positive,
                "migrated": len(migrations), "migrated_archive_entered": len(migrated_entered),
                "base_best_bacc": base_best, "merged_best_bacc": merged_best})
            all_pairs.extend(pairs)
        log(f"skeleton generation {gen} ({stage.lower()}): {len(evaluated)} evaluated, "
            f"{len(feedback_new)} feedback components, {len(migrations)} migrated, "
            f"best BACC {merged_best:.6f}")
        state.update({"skeleton_population": [dict(x) for x in population],
                      "archives": archives.state()})
        checkpoint()
        if stop_after is not None and gen >= stop_after:
            raise StopRequested(f"stopped after skeleton generation {gen}")
        last_generation = settings.final_skeleton_generation
        if last_generation is not None and gen >= last_generation:
            log(f"skeleton generation {gen} is the configured last generation")
            break
        if (settings.saturation_stop and saturation_trigger >= 0 and saturation_end >= 0
                and gen >= saturation_end):
            log(f"skeleton generation {gen}: consolidation finished")
            break
        if gen >= settings.skeleton_min_generations and stale >= settings.skeleton_patience:
            break


# --------------------------------------------------------------- finalizing


def add_group_metrics(record: dict[str, Any], score: np.ndarray, data: SearchData) -> None:
    """Accuracy per curve and per relation component, and the multiconfiguration subset."""
    s = np.asarray(score, dtype=np.float64)
    scoring = data.scoring
    rows = scoring.all_rows
    yy = scoring.y[rows]
    detail = None
    if scoring.mode == GROUPED_OOF:
        detail = grouped_oof_metrics(s[rows][None, :], yy, scoring.folds)
        pred = np.asarray(detail["prediction"][0], dtype=bool)
    else:
        threshold = float(record["threshold"])
        pred = np.where(int(record["orientation"]) == 1, s[rows] >= threshold,
                        s[rows] <= threshold)
    curve = grouped_accuracy(pred, yy, data.curve_codes)
    relation = grouped_accuracy(pred, yy, data.relation_codes)
    record.update({"curve_macro_accuracy": curve["macro_accuracy"],
                   "curve_worst_accuracy": curve["worst_accuracy"],
                   "curve_exact_rate": curve["exact_rate"],
                   "relation_component_macro_accuracy": relation["macro_accuracy"],
                   "relation_component_worst_accuracy": relation["worst_accuracy"],
                   "relation_component_exact_rate": relation["exact_rate"]})
    if detail is None:
        return
    mm = np.asarray(data.multiconfig, dtype=bool)
    out: dict[str, Any] = {"multiconfig_rows": int(mm.sum()), "multiconfig_bacc": None,
                           "multiconfig_acc": None, "multiconfig_auc": None}
    if np.any(mm) and len(np.unique(yy[mm])) == 2:
        p = pred[mm]
        ysub = yy[mm]
        pos = ysub == 1
        neg = ysub == 0
        pr = float(np.mean(p[pos])) if np.any(pos) else float("nan")
        nr = float(np.mean(~p[neg])) if np.any(neg) else float("nan")
        margin = np.asarray(detail["margin"][0])[mm]
        out.update({"multiconfig_bacc": 0.5 * (pr + nr),
                    "multiconfig_acc": float(np.mean(p == ysub)),
                    "multiconfig_auc": safe_auc(ysub, margin, 1)})
    record.update(out)


def oof_tables(top: Sequence[Mapping[str, Any]], scores: Mapping[str, np.ndarray],
               data: SearchData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-sample out-of-fold scores and subset metrics of the top formulas."""
    scoring = data.scoring
    meta = data.rows.reset_index(drop=True)
    folds = scoring.folds
    fold_index = {f: j for j, f in enumerate(sorted(set(map(int, folds))))}
    sample_ids = [str(x) for x in meta["sample_id"].tolist()]
    curve_ids = [str(x) for x in meta["curve_id"].tolist()]
    parents = [str(x) for x in meta["parent_formula"].tolist()]
    multi = [bool(x) for x in meta["is_multiconfig"].tolist()]
    predictions, subsets = [], []
    for r in top:
        cid = str(r["component_level_formula_id"])
        score = np.asarray(scores[cid], dtype=np.float64)[scoring.all_rows]
        detail = grouped_oof_metrics(score[None, :], scoring.y[scoring.all_rows], folds)
        pred = np.asarray(detail["prediction"][0], dtype=bool)
        margin = np.asarray(detail["margin"][0], dtype=float)
        thresholds = np.asarray(detail["thresholds"][0], dtype=float)
        orientations = np.asarray(detail["orientations"][0], dtype=np.int8)
        for i in range(len(meta)):
            f = int(folds[i])
            j = fold_index[f]
            predictions.append({
                "rank": int(r["rank"]), "formula_id": r["formula_id"],
                "component_level_formula_id": cid, "sample_id": sample_ids[i],
                "curve_id": curve_ids[i], "parent_formula": parents[i], "fold_id": f,
                "is_multiconfig": multi[i], "truth": int(scoring.y[i]), "score": float(score[i]),
                "train_only_threshold": float(thresholds[j]),
                "train_only_orientation": int(orientations[j]),
                "oof_margin": float(margin[i]), "oof_prediction": int(pred[i])})
        mm = np.asarray(data.multiconfig, dtype=bool)
        for name, mask in (("ALL", np.ones(len(meta), bool)), ("MULTICONFIG", mm),
                           ("REMAINING", ~mm)):
            row: dict[str, Any] = {"rank": int(r["rank"]), "formula_id": r["formula_id"],
                                   "subset": name, "rows": int(mask.sum()), "bacc": None,
                                   "acc": None, "auc": None}
            if np.any(mask):
                yy = scoring.y[mask]
                pp = pred[mask]
                if len(np.unique(yy)) == 2:
                    row["bacc"] = 0.5 * (float(np.mean(pp[yy == 1]))
                                         + float(np.mean(~pp[yy == 0])))
                    row["auc"] = safe_auc(yy, margin[mask], 1)
                row["acc"] = float(np.mean(pp == yy))
            subsets.append(row)
    return pd.DataFrame(predictions), pd.DataFrame(subsets)


def finalize(outdir: Path, registry: ComponentRegistry, archives: Archives, workers: Workers,
             data: SearchData, settings: Settings) -> dict[str, Any]:
    """Re-evaluate the archive leaders, try deletions, rank and write the results."""
    key = archives.key
    rows, names = registry.rows, data.names
    candidates = final_candidates(archives, settings.final_candidate_cap)
    evaluated = workers.evaluate_skeletons(
        [json.loads(r["component_ast_json"]) for r in candidates])
    records = []
    scores: dict[str, np.ndarray] = {}
    for result, old in zip(evaluated, candidates):
        if result.get("status") not in EVALUATED:
            continue
        rec = archive_record(result, rows, names, "FINAL_REPLAY", int(old.get("generation", -1)),
                             int(old.get("skeleton_island_id", -1)))
        add_group_metrics(rec, result["values"], data)
        records.append(rec)
        scores[rec["component_level_formula_id"]] = np.asarray(result["values"], dtype=np.float64)
    records.sort(key=key)

    resolution = natural_bacc_resolution(data.scoring.y[data.scoring.all_rows])
    deletions, parent_ids = [], []
    for rec in records[:min(64, len(records))]:
        for candidate in simplification_candidates(json.loads(rec["component_ast_json"]))[:16]:
            deletions.append(candidate)
            parent_ids.append(rec["component_level_formula_id"])
    audit = []
    if deletions:
        parents = {r["component_level_formula_id"]: r for r in records}
        for result, parent_id in zip(workers.evaluate_skeletons(deletions), parent_ids):
            if result.get("status") not in EVALUATED or not bool(result["safe"]):
                continue
            parent = parents[parent_id]
            c = result["complexity"]
            simpler = (int(c["expanded_operator_nodes"]) < int(parent["operator_nodes"])
                       or int(c["expanded_tree_depth"]) < int(parent["tree_depth"]))
            audit.append({"parent_formula_id": parent["formula_id"],
                          "candidate_formula_id": result["formula_id"],
                          "parent_bacc": float(parent["all_bacc"]),
                          "candidate_bacc": float(result["all_bacc"]),
                          "delta_bacc": float(result["all_bacc"]) - float(parent["all_bacc"]),
                          "parent_operator_nodes": int(parent["operator_nodes"]),
                          "candidate_operator_nodes": int(c["expanded_operator_nodes"]),
                          "operator_reduction": (int(parent["operator_nodes"])
                                                 - int(c["expanded_operator_nodes"])),
                          "simpler": bool(simpler)})
            # A deletion is kept when it is smaller and loses less than one
            # sample's worth of BACC and worst-class recall.
            if (simpler and float(result["all_bacc"]) + resolution >= float(parent["all_bacc"])
                    and float(result["all_worst_recall"]) + resolution
                    >= float(parent["all_worst_recall"])):
                rec = archive_record(result, rows, names, "FINAL_SIMPLIFICATION",
                                     int(parent["generation"]), int(parent["skeleton_island_id"]))
                add_group_metrics(rec, result["values"], data)
                records.append(rec)
                scores[rec["component_level_formula_id"]] = np.asarray(result["values"],
                                                                       dtype=np.float64)
    best_by_formula: dict[str, dict[str, Any]] = {}
    for r in records:
        old = best_by_formula.get(r["formula_id"])
        if old is None or key(r) < key(old):
            best_by_formula[r["formula_id"]] = r
    records = sorted(best_by_formula.values(), key=key)
    top = records[:settings.final_top_n]
    for rank, r in enumerate(top, 1):
        r["rank"] = rank

    write_csv(outdir / "top30.csv", top)
    write_csv(outdir / "final_candidates.csv", records[:max(settings.final_top_n, 100)])
    write_csv(outdir / "simplification_audit.csv", audit)
    if top:
        write_json(outdir / "best_formula.json", top[0])
        simple = min(top, key=lambda r: (int(r["operator_nodes"]), int(r["tree_depth"]),
                                         -float(r["all_bacc"]), r["formula_id"]))
        write_json(outdir / "best_simple_formula.json", simple)
        topology = next(iter(sorted(archives.topology.values(), key=key)), top[0])
        write_json(outdir / "best_topology_formula.json", topology)
    if data.scoring.mode == GROUPED_OOF and top:
        predictions, subsets = oof_tables(top, scores, data)
        write_csv(outdir / "top30_oof_predictions.csv", predictions)
        write_csv(outdir / "top30_subset_oof.csv", subsets)
    return {"final_candidates": len(records), "top_n": len(top),
            "best_bacc": float(top[0]["all_bacc"]) if top else None,
            "best_formula_id": top[0]["formula_id"] if top else None,
            "best_formula": top[0]["formula"] if top else None}


def write_progress(outdir: Path, registry: ComponentRegistry, state: Mapping[str, Any],
                   archives: Archives | None, data: SearchData) -> None:
    """Components, histories and archives as they stand."""
    write_csv(outdir / "components.csv", registry.table())
    for name, field in (("component_history", "component_history"),
                        ("component_operator_stats", "component_operator_history"),
                        ("skeleton_history", "skeleton_history"),
                        ("skeleton_operator_stats", "skeleton_operator_history"),
                        ("feedback_history", "feedback_history"),
                        ("feedback_pairs", "feedback_pairs")):
        write_csv(outdir / f"{name}.csv", state.get(field, []))
    write_csv(outdir / "feature_scales.csv", pd.DataFrame({
        "feature_index": np.arange(len(data.names), dtype=np.int32),
        "feature_name": list(data.names), "scale": data.scales,
        "scale_method": list(data.scale_methods)}))
    if archives is None:
        return
    key = archives.key
    performance = sorted(archives.performance.values(), key=key)
    write_csv(outdir / "archive_performance.csv", performance)
    write_csv(outdir / "archive_topology.csv", sorted(archives.topology.values(), key=key))
    write_csv(outdir / "archive_prediction.csv", sorted(archives.prediction.values(), key=key))
    if performance:
        best = float(performance[0]["all_bacc"])
        band = natural_bacc_resolution(data.scoring.y[data.scoring.all_rows])
        simple = sorted([r for r in performance if float(r["all_bacc"]) + band >= best],
                        key=lambda r: (int(r["operator_nodes"]), int(r["tree_depth"]),
                                       -float(r["all_bacc"]), str(r["formula_id"])))
        write_csv(outdir / "archive_simple_near_optimal.csv", simple[:512])


def seed_table(seeds: Sequence[Mapping[str, Any]], sources: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame([{
        "seed_index": i, "source": src, "component_level_formula_id": g.skeleton_id(ast),
        "topology_id": g.topology_id(ast),
        "component_ids": ";".join(map(str, sorted(set(g.skeleton_component_ids(ast))))),
        "component_ast_json": json.dumps(ast, sort_keys=True, separators=(",", ":"))}
        for i, (ast, src) in enumerate(zip(seeds, sources))])


# ---------------------------------------------------------------------- run


def run(settings: Settings, protocol: Path, space: Path, outdir: Path, *, resume: bool = False,
        overwrite: bool = False, scratch: Path | None = None,
        stop_after_skeleton_generation: int | None = None) -> dict[str, Any]:
    """Run, resume or finish a search; returns the run record."""
    outdir = Path(outdir).resolve()
    data = load_data(protocol, space, settings.scoring)
    rngs = rng_streams(settings.seed)
    if resume:
        with (outdir / CHECKPOINT).open("rb") as f:
            payload = pickle.load(f)
        if payload["settings"] != settings.scientific():
            changed = sorted(k for k, v in settings.scientific().items()
                             if payload["settings"].get(k) != v)
            raise ValueError(f"settings differ from the checkpoint: {', '.join(changed)}")
        registry = ComponentRegistry(outdir, data, settings, create=False)
        registry.restore(payload["registry"])
        for name, rng in rngs.items():
            rng.bit_generator.state = payload["rngs"][name]
        phase, state = payload["phase"], dict(payload["state"])
        log(f"resuming at phase {phase} with {len(registry)} components")
        record = json.loads((outdir / RUN_RECORD).read_text(encoding="utf-8"))
        record.setdefault("resumed", []).append(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    else:
        if outdir.exists() and any(outdir.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{outdir} is not empty; use --overwrite or --resume")
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        registry = ComponentRegistry(outdir, data, settings, create=True)
        phase, state = "PREPARE", {}
        record = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "settings": asdict(settings),
                  "inputs": {"protocol": str(Path(protocol).resolve()),
                             "space": str(Path(space).resolve()), "sha256": data.checksums}}
    record.update({"command": sys.argv, "software": software_versions(), "status": "running"})
    write_json(outdir / RUN_RECORD, record)

    def save(new_phase: str) -> None:
        nonlocal phase
        phase = new_phase
        save_checkpoint(outdir, phase, settings, registry, rngs, state)

    archives = None
    workers = Workers(data, settings, registry, scratch)
    try:
        if phase == "PREPARE":
            asts, counts = initial_components(DescriptorSampler.from_data(data), settings,
                                              rngs["initial_component"])
            rejections: dict[str, int] = {}
            for result in workers.evaluate_components(asts):
                status = str(result.get("status", "INVALID"))
                if status == "VALID":
                    ast = result["ast"]
                    if ast["op"] == "terminal":
                        source = "RAW_TERMINAL"
                    elif int(g.expression_complexity(ast)["n_base_features"]) == 1:
                        source = "UNARY_ENUMERATION"
                    else:
                        source = "INITIAL_COMPONENT_UNIVERSE"
                    _, status = registry.add(result, source=source, generation=0,
                                             proposal_operator="initial")
                rejections[status] = rejections.get(status, 0) + 1
            registry.flush()
            population = choose_component_population(registry, settings.component_population,
                                                     rngs["component"])
            anchors = population[:min(settings.component_context_anchors, len(population))]
            workers.probe_context(population[:settings.component_context_candidates], anchors,
                                  rngs["component"])
            state = {"component_population": population, "component_history": [],
                     "component_operator_history": [], "initial_component_counts": counts,
                     "initial_component_statuses": rejections}
            log(f"initial components: {len(asts)} proposed, {len(registry)} accepted")
            save("INITIAL_COMPONENTS")

        if phase in {"INITIAL_COMPONENTS", "COMPONENT_EVOLUTION"}:
            def component_checkpoint(population: list[int]) -> None:
                state["component_population"] = list(map(int, population))
                save("COMPONENT_EVOLUTION")

            state["component_population"] = evolve_components(
                registry, workers, settings, rngs["component"],
                list(map(int, state["component_population"])), state["component_history"],
                state["component_operator_history"], component_checkpoint)
            save("COMPONENT_COMPLETE")

        if phase == "COMPONENT_COMPLETE":
            seeds, sources = workers.seed_skeletons(registry.ids(), rngs["seed_skeleton"])
            write_csv(outdir / "initial_skeleton_seeds.csv", seed_table(seeds, sources))
            empty = Archives((settings.performance_archive_cap, settings.topology_archive_cap,
                              settings.prediction_archive_cap), settings.scoring)
            state.update({"skeleton_seeds": seeds, "skeleton_seed_sources": sources,
                          "skeleton_population": seeds, "archives": empty.state(),
                          "skeleton_history": [], "skeleton_operator_history": [],
                          "feedback_history": [], "feedback_pairs": []})
            save("SKELETON_READY")

        if phase in {"SKELETON_READY", "SKELETON_EVOLUTION"}:
            evolve_skeletons(registry, workers, settings, rngs, state,
                             lambda: save("SKELETON_EVOLUTION"), stop_after_skeleton_generation)
            save("SKELETON_COMPLETE")

        if phase == "SKELETON_COMPLETE":
            archives = Archives((settings.performance_archive_cap, settings.topology_archive_cap,
                                 settings.prediction_archive_cap), settings.scoring)
            archives.restore(state["archives"])
            write_progress(outdir, registry, state, archives, data)
            summary = finalize(outdir, registry, archives, workers, data, settings)
            state["summary"] = summary
            save("FINALIZED")
            log(f"finalized: best BACC {summary['best_bacc']}")
        record.update({"status": "finalized", "result": state.get("summary")})
    except StopRequested as stop:
        write_progress(outdir, registry, state, None, data)
        record.update({"status": "stopped", "phase": phase, "message": str(stop)})
        log(str(stop))
    finally:
        workers.close()
        record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(outdir / RUN_RECORD, record)
    return record
