"""Descriptor registries of the space directory.

``terminal_registry.csv``
    domain (POSITIVE, NONNEGATIVE or SIGNED, from the minimum in the
    descriptor definitions) and exact-duplicate / constant flags;
``PHYSICAL_CHANNEL_ROLE_REGISTRY.csv``
    physical channel and component role inferred from the sector, role hint
    and name tokens of each descriptor;
``RF_SEARCH_GUIDANCE_REGISTRY.csv``
    the sampling prior: tier (A/B/C) and weight of each descriptor.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

POSITIVE, NONNEGATIVE, SIGNED = "POSITIVE", "NONNEGATIVE", "SIGNED"

ATOMIC = "ATOMIC_ONSITE"
LOCAL_BX = "LOCAL_BX_HYBRIDIZATION"
EXTENDED_XX = "EXTENDED_XX_NETWORK"
GEOMETRY = "OCTAHEDRAL_GEOMETRY"
ENVIRONMENT = "GLOBAL_ENVIRONMENT"
UNKNOWN = "MIXED_OR_UNKNOWN"

LOCAL = "LOCAL_COHERENT"
NETWORK = "NETWORK_COHERENT"
DEPHASING = "DEPHASING"
MODULATOR = "ENVIRONMENT_MODULATOR"
ESCAPE = "GLOBAL_ESCAPE"

SPREAD_STATISTICS = ("cv", "std", "variance", "anisotropy", "disorder", "distortion",
                     "mismatch", "heterogeneity", "range", "spread", "dephasing")
LEVEL_STATISTICS = ("mean", "median", "minimum", "min", "maximum", "max")


def terminal_registry(definitions: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    """Domains from the definition minimum; duplicates and constants from the values."""
    first_with_values: dict[bytes, int] = {}
    rows = []
    for i, d in enumerate(definitions.itertuples(index=False)):
        column = np.ascontiguousarray(X[:, i], dtype=np.float64)
        keeper = first_with_values.setdefault(column.tobytes(), i)
        duplicate = i != keeper
        constant = bool(np.ptp(column) == 0.0)
        lo, hi = float(d.minimum), float(d.maximum)
        rows.append({
            "feature_index": i, "feature_name": str(d.name),
            "canonical_feature_index": keeper,
            "canonical_feature_name": str(definitions.name.iloc[keeper]),
            "active_for_enumeration": not duplicate and not constant,
            "is_exact_duplicate_alias": duplicate, "is_constant": constant,
            "input_domain": POSITIVE if lo > 0 else NONNEGATIVE if lo >= 0 else SIGNED,
            "contains_zero": lo <= 0.0 <= hi,
            "minimum": lo, "maximum": hi, "median": float(d.median),
            "physical_sector": str(d.physical_sector), "role_hint": str(d.role_hint),
            "redundancy_group": str(d.redundancy_group), "unit": str(d.unit),
            "dimension_signature": str(d.dimension_signature),
            "allowed_transforms_upstream": str(d.allowed_transforms),
        })
    return pd.DataFrame(rows)


def _tokens(text: Any) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t}


def statistic_type(name: str, definition: str) -> str:
    tokens = _tokens(name) | _tokens(definition)
    for key in SPREAD_STATISTICS + LEVEL_STATISTICS:
        if key in tokens:
            return key.upper()
    return "RAW_OR_SCALE"


def channel_and_role(d: Mapping[str, Any]) -> dict[str, Any]:
    """Physical channel and component role of one descriptor definition."""
    name = str(d["name"])
    n = name.lower()
    sector = str(d["physical_sector"]).lower()
    hint = str(d["role_hint"]).lower()
    stat = statistic_type(name, str(d["definition"]))
    regularized = n.startswith("g_") or "regular" in n or hint == "coherence"

    # Name-specific pathways take priority over the broad sector.
    if n.startswith("xx_") or "t_pp" in n or sector in {"anion_network", "octahedral_network"}:
        channel = EXTENDED_XX
    elif sector == "local_octahedral_geometry":
        channel = GEOMETRY
    elif n.startswith("bx_") or "t_pd" in n or sector == "relative_hybridization":
        channel = LOCAL_BX
    elif sector in {"elemental_composition", "elemental_compatibility", "onsite_orbital_proxy",
                    "orbital_occupancy_proxy", "orbital_geometry_proxy"}:
        channel = ATOMIC
    elif sector in {"global_structure", "cage_environment", "alloy_disorder",
                    "alloy_configuration"}:
        channel = ENVIRONMENT
    else:
        channel = UNKNOWN

    spread = stat in {s.upper() for s in SPREAD_STATISTICS}
    if hint == "dephasing" or (spread and not regularized):
        role, allowed = DEPHASING, [DEPHASING]
    elif channel == EXTENDED_XX and hint in {"network", "coherence", "scale", "geometry"}:
        role, allowed = NETWORK, [NETWORK]
    elif channel in {LOCAL_BX, GEOMETRY} and hint in {"onsite", "coherence", "occupancy",
                                                      "geometry", "scale"}:
        role, allowed = LOCAL, [LOCAL] + ([NETWORK] if channel == GEOMETRY else [])
    elif channel == ATOMIC and hint in {"onsite", "coherence", "occupancy", "geometry"}:
        role = LOCAL
        allowed = [LOCAL] + ([MODULATOR] if "elem_a" in n or "a_site" in n else [])
    elif channel == ENVIRONMENT or hint in {"scale", "gate_unknown", "signed_correction",
                                            "configuration"}:
        role, allowed = MODULATOR, [MODULATOR]
    elif channel == EXTENDED_XX:
        role, allowed = NETWORK, [NETWORK]
    elif channel in {LOCAL_BX, GEOMETRY, ATOMIC}:
        role, allowed = LOCAL, [LOCAL]
    else:
        role, allowed = ESCAPE, [ESCAPE]

    if "elem_a" in n or "a_site" in n or n.startswith("ax_"):
        site = "A"
    elif n.startswith("bx_") or "t_pd" in n:
        site = "BX"
    elif n.startswith("xx_") or "t_pp" in n:
        site = "XX"
    else:
        site = "GLOBAL" if channel == ENVIRONMENT else "MIXED"
    return {
        "primary_channel": channel, "secondary_channel": "", "primary_role": role,
        "allowed_roles": ";".join(allowed), "site_tag": site, "statistic_type": stat,
        "mapping_rule": "ledger_sector+role_hint+auditable_name_tokens",
        "mapping_confidence": "HIGH" if channel != UNKNOWN and role != ESCAPE else "LOW",
    }


def role_registry(definitions: pd.DataFrame, terminals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(definitions.to_dict(orient="records")):
        t = terminals.iloc[i]
        rows.append({
            "feature_index": i, "feature_name": str(d["name"]), **channel_and_role(d),
            "input_domain": str(t.input_domain),
            "active_for_enumeration": bool(t.active_for_enumeration),
            "canonical_feature_index": int(t.canonical_feature_index),
            "access_via_canonical_feature": int(t.canonical_feature_index),
            "redundancy_group": str(d["redundancy_group"]),
        })
    return pd.DataFrame(rows)


def sampling_prior(roles: pd.DataFrame, rf_selected: set[int] | None) -> pd.DataFrame:
    """Tiers and weights the descriptor sampler draws from.

    Perovskites (``rf_selected`` is None): every active descriptor is tier B
    with weight 1. Spinel: the random-forest-selected descriptors are tier A;
    of the others, the first 96 in registry order or the first 24 of their
    role are tier B, the rest tier C; every active descriptor has weight 0.15.
    Inactive descriptors (duplicates, constants) are tier C with weight 0.
    """
    active = roles.active_for_enumeration.astype(bool).to_numpy()
    n = len(roles)
    order = np.arange(1, n + 1)
    role_order = np.empty(n, dtype=np.int64)
    for ids in roles.groupby("primary_role").indices.values():
        role_order[np.sort(ids)] = np.arange(1, len(ids) + 1)
    if rf_selected is None:
        tiers = np.where(active, "B", "C")
        weights = np.where(active, 1.0, 0.0)
        role_order = order
        promoted = np.zeros(n, dtype=bool)
        scope = "NO_RF_FRESH_TOPOLOGY_NATIVE_UNIFORM_ACCESS"
    else:
        selected = np.isin(np.arange(n), sorted(rf_selected))
        tiers = np.where(active & selected, "A",
                         np.where(active & ((order <= 96) | (role_order <= 24)), "B", "C"))
        weights = np.where(active, 0.15, 0.0)
        promoted = active & ~selected & (role_order <= 4)
        scope = "CURRENT_LOCKED_RF_RUN; HARD_IMPURITY_PRIMARY; SOFT_PERMUTATION_SECONDARY"
    return pd.DataFrame({
        "feature_index": np.arange(n), "feature_name": roles.feature_name.to_numpy(),
        "primary_role": roles.primary_role.to_numpy(),
        "rf_selected": np.zeros(n, dtype=bool) if rf_selected is None else selected,
        "impurity_importance_hard": 0.0, "permutation_importance_soft_all": 0.0,
        "rf_guidance_importance": 0.0, "global_rf_rank": order, "within_role_rank": role_order,
        "search_tier": tiers, "sampling_weight": weights, "direct_access": active,
        "access_via_canonical_feature": roles.canonical_feature_index.to_numpy(),
        "importance_scope": scope, "role_promoted": promoted,
    })
