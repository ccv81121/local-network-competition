"""Building the protocol and space directories of a search from descriptor tables.

``prepare(family, raw, out)`` reads the tables of one family from ``raw``
(``data/descriptor_tables/<family>/``) and writes ``out/protocol`` and
``out/space``, the two directories ``run`` and ``replay`` take.

Raw tables
    ``descriptors.csv``  one row per structure: ``sample_id`` and the
                         descriptor values;
    ``samples.csv``      labels and grouping of the same rows;
    spinel only:         ``descriptor_definitions.csv`` (sector, unit, role
                         hint and range over the 1003-structure database);
    DP only:             ``numerical_guard.csv`` (descriptors kept for the
                         search after the numerical-resolution check).

Spinel rows are split into hard-core and soft rows by the random-forest
margin scan (``random_forest.py``), which also selects the tier-A
descriptors of the sampling prior. Perovskite rows are all hard-core and get
five parent-grouped folds; their prior is uniform.

``edge`` (DP only) replaces the label by the band-edge label of the valence
or conduction edge (1 = extremum at the momentum of the lowest direct gap),
read from ``edge_labels`` (columns ``A_V_exact`` / ``A_C_exact``).  The
folds stay those of the direct-gap label, as in the published edge runs.

``spinel-edges`` writes the inputs of the spinel two-edge search
(``edges run``) instead: ``protocol/`` with the descriptor matrix of all 681
retained states, their ids and the descriptor names, and next to it
``samples.csv`` (the 440 discovery states in search order),
``band_edges.csv`` (k points and Γ-versus-non-Γ margins per state) and
``prior_valence.csv`` / ``prior_conduction.csv`` (random-forest importances).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import perovskite, random_forest, registry

FAMILIES = ("spinel", "sp", "dp", "spinel-edges")

SPINEL_ROW_COLUMNS = (
    "source_global_row_index", "sample_id", "strict_label", "y", "ordering_margin_eV",
    "ordering_margin_kind", "is_metal_excluded", "is_all_primary", "is_hard_core",
    "is_soft_all", "is_margin_unavailable", "curve_id", "mp_id", "k",
    "relation_component_id", "challenge_target", "challenge_guard", "challenge_5d_soc",
    "challenge_small_gap", "challenge_structure_occupancy", "challenge_rare_chemistry",
    "challenge_multi_factor", "is_development")


def _read(path: Path) -> pd.DataFrame:
    # round_trip parsing returns every value exactly as written
    return pd.read_csv(path, float_precision="round_trip")


def _descriptors(raw: Path, samples: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    table = _read(raw / "descriptors.csv")
    if not table.sample_id.astype(str).equals(samples.sample_id.astype(str)):
        raise ValueError("descriptors.csv and samples.csv list different rows")
    names = [str(c) for c in table.columns[1:]]
    X = table[names].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("descriptor table has non-finite values")
    return names, np.ascontiguousarray(X)


def _spinel(raw: Path) -> dict:
    samples = _read(raw / "samples.csv")
    names, X = _descriptors(raw, samples)
    # Read with the default parser, as the published registry was.
    definitions = pd.read_csv(raw / "descriptor_definitions.csv")
    if definitions.name.astype(str).tolist() != names:
        raise ValueError("descriptor_definitions.csv does not follow descriptors.csv")
    rows = random_forest.Rows(
        X=X, y=samples.y.to_numpy(np.int8), margin=samples.ordering_margin_eV.to_numpy(float),
        curves=samples.curve_id.astype(str).to_numpy(),
        components=samples.relation_component_id.astype(str).to_numpy())
    split = random_forest.split_and_select(rows)
    table = samples.assign(is_all_primary=True, is_hard_core=split.hard, is_soft_all=~split.hard,
                           challenge_target=False, challenge_guard=False, is_development=True)
    return {"names": names, "X": X, "y": rows.y, "hard": split.hard,
            "rows": table[list(SPINEL_ROW_COLUMNS)], "row_float_format": "%.17g",
            "definitions": definitions, "rf_selected": set(split.selected.tolist()),
            "summary": {"tau_positive": split.tau_positive, "tau_negative": split.tau_negative,
                        "hard_rows": int(split.hard.sum()),
                        "rf_selected": [names[i] for i in split.selected]}}


def _perovskite(raw: Path, family: str, edge: str | None = None,
                edge_labels: Path | None = None) -> dict:
    samples = _read(raw / "samples.csv")
    names, X = _descriptors(raw, samples)
    round_trips = 1
    if family == "dp":
        guard = _read(raw / "numerical_guard.csv").set_index("descriptor")
        keep = [i for i, n in enumerate(names) if bool(guard.loc[n, "allow_for_search"])]
        names, X = [names[i] for i in keep], np.ascontiguousarray(X[:, keep])
        round_trips = 2   # the published DP definitions passed through one more CSV step
    y = samples.target.to_numpy(np.int8)
    folds = perovskite.grouped_folds(samples.target.astype(int).to_numpy(),
                                     samples.parent_formula.astype(str).to_numpy())
    rows = perovskite.protocol_rows(samples, folds)
    if edge is not None:
        labels = _read(edge_labels).set_index("sample_id")
        column = {"valence": "A_V_exact", "conduction": "A_C_exact"}[edge]
        y = labels.loc[samples.sample_id.astype(str), column].to_numpy(np.int8)
        rows["edge_label"] = y
    return {"names": names, "X": X, "y": y, "hard": np.ones(len(y), dtype=bool),
            "folds": folds, "multiconfig": rows.is_multiconfig.to_numpy(bool),
            "rows": rows, "row_float_format": None,
            "definitions": perovskite.definitions(names, X, round_trips), "rf_selected": None,
            # the published SP matrix was stored column-major (same values)
            "column_major": family == "sp",
            "summary": {"hard_rows": len(y), "folds": int(folds.max()) + 1}}


def _write_csv(frame: pd.DataFrame, path: Path, float_format: str | None = "%.17g") -> None:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n",
                 float_format=float_format)


def _spinel_edges(raw: Path, out: Path) -> dict:
    """Inputs of ``edges run`` for the spinel valence and conduction coordinates."""
    table = _read(raw / "descriptors.csv")
    names = [str(c) for c in table.columns[1:]]
    X = np.ascontiguousarray(table[names].to_numpy(dtype=np.float64))
    if not np.isfinite(X).all():
        raise ValueError("descriptor table has non-finite values")
    sample_ids = table.sample_id.astype(str).to_numpy()
    arrays, features = out / "protocol" / "arrays", out / "protocol" / "features"
    for d in (arrays, features):
        d.mkdir(parents=True, exist_ok=True)
    np.save(arrays / "X_full_clean_float64.npy", X, allow_pickle=False)
    np.save(arrays / "sample_id_unicode.npy", np.asarray(sample_ids.tolist()), allow_pickle=False)
    (features / "formula_full_feature_names.json").write_text(
        json.dumps(names, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    # The other tables are copied as they are (their cells are the source values).
    shutil.copyfile(raw / "discovery_samples.csv", out / "samples.csv")
    shutil.copyfile(raw / "band_edges.csv", out / "band_edges.csv")
    # Importances stay text, so the prior reads exactly the source values.
    rf = pd.read_csv(raw / "random_forest_importance.csv", dtype=str, keep_default_na=False)
    for edge in ("valence", "conduction"):
        rf[["feature", edge]].rename(columns={edge: "importance"}).to_csv(
            out / f"prior_{edge}.csv", index=False, lineterminator="\n")
    discovery = pd.read_csv(out / "samples.csv").sample_id.astype(str)
    if not discovery.isin(sample_ids).all():
        raise ValueError("discovery_samples.csv lists states missing from descriptors.csv")
    return {"family": "spinel-edges", "rows": int(X.shape[0]), "descriptors": len(names),
            "discovery_rows": int(len(discovery)), "protocol": str(out / "protocol"),
            "samples": str(out / "samples.csv"), "kpoints": str(out / "band_edges.csv"),
            "prior_valence": str(out / "prior_valence.csv"),
            "prior_conduction": str(out / "prior_conduction.csv")}


def prepare(family: str, raw: Path, out: Path, edge: str | None = None,
            edge_labels: Path | None = None) -> dict:
    """Write ``out/protocol`` and ``out/space`` for one family; return a summary."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if edge is not None and (family != "dp" or edge_labels is None):
        raise ValueError("edge labels are available for the DP family only (with edge_labels)")
    raw, out = Path(raw), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    if family == "spinel-edges":
        return _spinel_edges(raw, out)
    built = _spinel(raw) if family == "spinel" else _perovskite(raw, family, edge, edge_labels)
    names, X, hard = built["names"], built["X"], built["hard"]

    protocol, space = out / "protocol", out / "space"
    arrays = protocol / "RF_PROTOCOL_BUNDLE" / "arrays"
    features = protocol / "RF_PROTOCOL_BUNDLE" / "features"
    for d in (arrays, features, space):
        d.mkdir(parents=True, exist_ok=True)

    np.save(arrays / "X_full_clean_float64.npy",
            np.asfortranarray(X) if built.get("column_major") else X, allow_pickle=False)
    np.save(arrays / "y_int8.npy", built["y"].astype(np.int8), allow_pickle=False)
    np.save(arrays / "mask_all_primary_bool.npy", np.ones(len(hard), dtype=bool),
            allow_pickle=False)
    np.save(arrays / "mask_hard_core_bool.npy", hard.astype(bool), allow_pickle=False)
    np.save(arrays / "mask_soft_all_bool.npy", ~hard.astype(bool), allow_pickle=False)
    if "folds" in built:
        np.save(arrays / "fold_id_int16.npy", built["folds"].astype(np.int16), allow_pickle=False)
        np.save(arrays / "mask_multiconfig_bool.npy", built["multiconfig"], allow_pickle=False)
    (features / "formula_full_feature_names.json").write_text(
        json.dumps(names, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    _write_csv(built["rows"], protocol / "protocol_rows.csv", built["row_float_format"])

    terminals = registry.terminal_registry(built["definitions"], X)
    roles = registry.role_registry(built["definitions"], terminals)
    prior = registry.sampling_prior(roles, built["rf_selected"])
    _write_csv(terminals, space / "terminal_registry.csv")
    _write_csv(roles, space / "PHYSICAL_CHANNEL_ROLE_REGISTRY.csv")
    _write_csv(prior, space / "RF_SEARCH_GUIDANCE_REGISTRY.csv")
    return {"family": family, "edge": edge, "positive": int(built["y"].sum()),
            "rows": int(X.shape[0]), "descriptors": len(names),
            "active_descriptors": int(terminals.active_for_enumeration.sum()),
            **built["summary"], "protocol": str(protocol), "space": str(space)}
