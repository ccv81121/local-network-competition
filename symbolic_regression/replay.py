"""Re-evaluating saved formulas.

A saved formula is a skeleton over components (``component_ast_json``) and
needs the expressions of those components, taken from a ``components.csv``
of the run.  Replay rebuilds each component from its expression on the
scaled descriptors, evaluates the skeleton with the arithmetic of the search,
refits the threshold and compares the result with every stored field.

Components carried over from a single-precision registry are marked with
``value_dtype = float32`` in the components file; their values are rounded
to single precision before use, as they were in that registry.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import grammar as g
from .data import SearchData
from .metrics import GROUPED_OOF
from .search import add_group_metrics, oof_tables
from .settings import Settings

COMPARED_FIELDS = (
    "formula_id", "component_level_formula_id", "topology_id", "raw_feature_count",
    "operator_nodes", "tree_depth", "orientation", "threshold", "threshold_margin",
    "all_bacc", "all_worst_recall", "all_auc", "all_acc", "all_f1", "all_mcc",
    "hard_bacc", "hard_worst_recall", "soft_bacc", "soft_auc", "prediction_id",
    "curve_macro_accuracy", "relation_component_macro_accuracy",
)
OOF_FIELDS = ("oof_bacc", "oof_auc", "oof_worst_recall", "oof_orientation_agreement",
              "oof_threshold_std_scaled", "full_fit_bacc", "multiconfig_bacc", "multiconfig_auc")


def load_formulas(path: Path) -> list[dict[str, Any]]:
    """Formula records from a JSON object, a JSON list or a CSV table."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, float_precision="round_trip").to_dict("records")
    value = json.loads(path.read_text(encoding="utf-8"))
    return [value] if isinstance(value, dict) else list(value)


def load_components(path: Path, ids: set[int]) -> dict[int, tuple[dict[str, Any], str]]:
    """Expressions (and stored precision) of the components in ``ids``."""
    found: dict[int, tuple[dict[str, Any], str]] = {}
    header = pd.read_csv(path, nrows=0).columns
    columns = ["component_id", "ast_json"] + (["value_dtype"] if "value_dtype" in header else [])
    for chunk in pd.read_csv(path, usecols=columns, chunksize=200000):
        chunk = chunk[chunk["component_id"].isin(ids)]
        for row in chunk.itertuples(index=False):
            dtype = str(getattr(row, "value_dtype", "float64"))
            found[int(row.component_id)] = (json.loads(row.ast_json), dtype)
    missing = sorted(ids - set(found))
    if missing:
        raise KeyError(f"components missing from {path}: {missing[:10]}")
    return found


def component_values(ast: Mapping[str, Any], dtype: str, data: SearchData,
                     epsilon: float) -> np.ndarray:
    values, _ = g.evaluate_expression(ast, data.features, epsilon=epsilon)
    if dtype == "float32":
        return values.astype(np.float32).astype(np.float64)
    return np.asarray(values, dtype=np.float64)


def evaluate_formula(record: Mapping[str, Any],
                     components: Mapping[int, tuple[dict[str, Any], str]],
                     data: SearchData, settings: Settings) -> tuple[dict[str, Any], np.ndarray]:
    """Metric record and per-sample scores of one saved formula."""
    skeleton = json.loads(record["component_ast_json"])
    asts = {cid: ast for cid, (ast, _) in components.items()}
    can, expanded, complexity = g.validate_skeleton(skeleton, asts.__getitem__, data.domains)
    cache: dict[int, np.ndarray] = {}

    def values_of(cid: int) -> np.ndarray:
        if cid not in cache:
            ast, dtype = components[cid]
            cache[cid] = component_values(ast, dtype, data, settings.epsilon)
        return cache[cid]

    scores = g.evaluate_skeleton(can, values_of, settings.epsilon, {})
    replayed = dict(data.scoring.metrics(scores[None, :])[0])
    replayed.update({
        "formula_id": g.tree_id(expanded), "component_level_formula_id": g.skeleton_id(can),
        "topology_id": g.topology_id(can), "raw_feature_count": complexity["raw_feature_count"],
        "operator_nodes": complexity["expanded_operator_nodes"],
        "tree_depth": complexity["expanded_tree_depth"],
        "formula": g.to_formula(expanded, data.names)})
    add_group_metrics(replayed, scores, data)
    return replayed, scores


def oof_predictions(records: Sequence[Mapping[str, Any]], reports: Sequence[Mapping[str, Any]],
                    scores: pd.DataFrame, data: SearchData) -> pd.DataFrame:
    """Per-sample out-of-fold predictions of replayed formulas (grouped scoring)."""
    if data.scoring.mode != GROUPED_OOF:
        raise ValueError("out-of-fold predictions need grouped scoring")
    n = data.n_samples
    top, by_id = [], {}
    for i, (record, report) in enumerate(zip(records, reports)):
        key = f"{i}:{record.get('component_level_formula_id', '')}"
        top.append({"rank": report["rank"], "formula_id": record.get("formula_id"),
                    "component_level_formula_id": key})
        by_id[key] = scores["score"].to_numpy()[i * n:(i + 1) * n]
    table, _ = oof_tables(top, by_id, data)
    table["component_level_formula_id"] = [
        k.split(":", 1)[1] for k in table["component_level_formula_id"]]
    return table


def _same(stored: Any, replayed: Any) -> bool:
    if stored is None or (isinstance(stored, float) and math.isnan(stored)):
        return replayed is None or (isinstance(replayed, float) and math.isnan(replayed))
    if isinstance(replayed, (bool, np.bool_)) or isinstance(stored, str):
        return str(stored) == str(replayed)
    if isinstance(replayed, (int, float, np.integer, np.floating)):
        return float(stored) == float(replayed)
    return str(stored) == str(replayed)


def replay(records: Sequence[Mapping[str, Any]], components_path: Path, data: SearchData,
           settings: Settings) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Compare stored and replayed fields; also return per-sample scores."""
    ids = {cid for r in records
           for cid in g.skeleton_component_ids(json.loads(r["component_ast_json"]))}
    components = load_components(components_path, ids)
    fields = COMPARED_FIELDS + (OOF_FIELDS if settings.scoring == GROUPED_OOF else ())
    reports, score_rows = [], []
    sample_ids = [str(x) for x in data.rows["sample_id"].tolist()]
    for position, record in enumerate(records, 1):
        replayed, scores = evaluate_formula(record, components, data, settings)
        checks = {f: {"stored": record[f], "replayed": replayed.get(f),
                      "equal": _same(record[f], replayed.get(f))}
                  for f in fields if f in record}
        rank = record.get("rank", position)
        reports.append({"rank": rank, "formula_id": record.get("formula_id"),
                        "formula": replayed["formula"],
                        "all_equal": all(c["equal"] for c in checks.values()), "checks": checks})
        score_rows.extend({"rank": rank, "formula_id": replayed["formula_id"],
                           "sample_id": sample_ids[i], "score": float(scores[i])}
                          for i in range(len(scores)))
    return reports, pd.DataFrame(score_rows)
