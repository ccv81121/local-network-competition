"""Loading the prepared inputs of a search.

A search reads two directories written by ``prepare`` (``inputs/``):

protocol directory
    ``RF_PROTOCOL_BUNDLE/arrays/*.npy`` (descriptor matrix, labels, row
    masks and, for grouped scoring, folds and the multiconfiguration mask),
    ``RF_PROTOCOL_BUNDLE/features/formula_full_feature_names.json`` and
    ``protocol_rows.csv``;
space directory
    ``terminal_registry.csv`` (descriptor domains),
    ``PHYSICAL_CHANNEL_ROLE_REGISTRY.csv`` (channel and role labels) and
    ``RF_SEARCH_GUIDANCE_REGISTRY.csv`` (descriptor sampling prior).

Descriptors are divided by a robust scale (IQR, else MAD, else SD) computed
on the hard-core rows, so every terminal of the grammar is dimensionless.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .metrics import GROUPED_OOF, Scoring


@dataclass(frozen=True)
class SearchData:
    features: np.ndarray          # scaled descriptors, shape (descriptor, sample)
    scales: np.ndarray
    scale_methods: tuple[str, ...]
    names: tuple[str, ...]
    domains: tuple[str, ...]
    channels: tuple[str, ...]
    primary_roles: tuple[str, ...]
    allowed_roles: tuple[str, ...]
    prior_weights: np.ndarray
    prior_tiers: tuple[str, ...]
    scoring: Scoring
    curve_codes: np.ndarray
    relation_codes: np.ndarray
    multiconfig: np.ndarray | None
    rows: pd.DataFrame
    checksums: dict[str, str]

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[1])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def robust_scale(values: np.ndarray) -> tuple[float, str]:
    x = np.asarray(values, dtype=np.float64)
    q25, q75 = np.quantile(x, [.25, .75])
    iqr = float(q75 - q25)
    if math.isfinite(iqr) and iqr > 1e-12:
        return iqr, "IQR"
    mad = float(np.median(np.abs(x - np.median(x)))) * 1.4826
    if math.isfinite(mad) and mad > 1e-12:
        return mad, "MAD"
    std = float(np.std(x))
    if math.isfinite(std) and std > 1e-12:
        return std, "STD"
    return 1.0, "UNIT"


def _codes(values: Sequence[str]) -> np.ndarray:
    labels = sorted(set(map(str, values)))
    index = {v: i for i, v in enumerate(labels)}
    return np.asarray([index[str(v)] for v in values], dtype=np.int32)


def load_data(protocol: Path, space: Path, scoring_mode: str) -> SearchData:
    protocol = Path(protocol).resolve()
    space = Path(space).resolve()
    arrays = protocol / "RF_PROTOCOL_BUNDLE" / "arrays"
    files = {
        "descriptors": arrays / "X_full_clean_float64.npy",
        "labels": arrays / "y_int8.npy",
        "mask_all": arrays / "mask_all_primary_bool.npy",
        "mask_hard": arrays / "mask_hard_core_bool.npy",
        "mask_soft": arrays / "mask_soft_all_bool.npy",
        "names": protocol / "RF_PROTOCOL_BUNDLE" / "features" / "formula_full_feature_names.json",
        "rows": protocol / "protocol_rows.csv",
        "terminals": space / "terminal_registry.csv",
        "roles": space / "PHYSICAL_CHANNEL_ROLE_REGISTRY.csv",
        "prior": space / "RF_SEARCH_GUIDANCE_REGISTRY.csv",
    }
    if scoring_mode == GROUPED_OOF:
        files["folds"] = arrays / "fold_id_int16.npy"
        files["multiconfig"] = arrays / "mask_multiconfig_bool.npy"
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing input files: " + ", ".join(missing))

    X = np.load(files["descriptors"], allow_pickle=False)
    y = np.load(files["labels"], allow_pickle=False).astype(np.int8, copy=False)
    mask_all, mask_hard, mask_soft = (
        np.load(files[key], allow_pickle=False).astype(bool, copy=False)
        for key in ("mask_all", "mask_hard", "mask_soft"))
    names = tuple(json.loads(files["names"].read_text(encoding="utf-8")))
    rows = pd.read_csv(files["rows"])
    terminals = pd.read_csv(files["terminals"])
    roles = pd.read_csv(files["roles"])
    prior = pd.read_csv(files["prior"])
    if X.shape != (len(y), len(names)):
        raise ValueError(f"descriptor matrix {X.shape} does not match "
                         f"{len(y)} labels and {len(names)} names")
    if not (len(terminals) == len(roles) == len(prior) == len(names)):
        raise ValueError("space registries and descriptor names differ in length")
    if len(rows) != len(y):
        raise ValueError("protocol_rows.csv and labels differ in length")
    if np.any(mask_hard & mask_soft) or not np.array_equal(mask_hard | mask_soft, mask_all):
        raise ValueError("hard-core and soft rows must partition the scored rows")

    hard_rows = np.flatnonzero(mask_hard)
    all_rows = np.flatnonzero(mask_all)
    soft_rows = np.flatnonzero(mask_soft)
    scales = np.ones(X.shape[1], dtype=np.float64)
    methods = []
    for i in range(X.shape[1]):
        scales[i], method = robust_scale(X[hard_rows, i])
        methods.append(method)
    features = np.ascontiguousarray((X / scales[None, :]).T, dtype=np.float64)
    features.flags.writeable = False

    folds = multiconfig = None
    if scoring_mode == GROUPED_OOF:
        if not np.array_equal(all_rows, np.arange(len(y))):
            raise ValueError("grouped out-of-fold scoring needs every row scored")
        folds = np.load(files["folds"], allow_pickle=False).astype(np.int16, copy=False)
        multiconfig = np.load(files["multiconfig"], allow_pickle=False).astype(bool, copy=False)
        if len(folds) != len(y) or len(multiconfig) != len(y):
            raise ValueError("fold or multiconfiguration array has the wrong length")
        folds = folds[mask_all]
        multiconfig = multiconfig[mask_all]

    scored = rows.loc[mask_all].reset_index(drop=True)
    return SearchData(
        features=features, scales=scales, scale_methods=tuple(methods), names=names,
        domains=tuple(terminals.input_domain.astype(str)),
        channels=tuple(roles.primary_channel.astype(str)),
        primary_roles=tuple(roles.primary_role.astype(str)),
        allowed_roles=tuple(roles.allowed_roles.astype(str)),
        prior_weights=np.asarray(prior.sampling_weight, dtype=np.float64),
        prior_tiers=tuple(prior.search_tier.astype(str)),
        scoring=Scoring(scoring_mode, y, hard_rows, all_rows, soft_rows, folds),
        curve_codes=_codes(scored.curve_id.astype(str)),
        relation_codes=_codes(scored.relation_component_id.astype(str)),
        multiconfig=multiconfig, rows=rows,
        checksums={str(path): sha256_file(path) for path in files.values()},
    )
