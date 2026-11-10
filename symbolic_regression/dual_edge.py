"""Two-edge symbolic search: one continuous coordinate per band edge.

For every development structure the valence (V) and conduction (C) band
extrema are either at Γ or away from it.  The search looks for two
independent expressions, Π_V(X) for the valence edge and Π_C(X) for the
conduction edge, each of which orders the structures by that edge (its AUC
for "extremum at Γ"), and agrees with the explicit Γ-versus-non-Γ energy
difference of the edge where it is known (Spearman correlation).  It produced
the spinel edge-resolved coordinates; the published Π_V and Π_C are simplified
and coefficient-rounded versions of formulas from this search.

How it works

* Descriptors of the discovery rows are divided by a robust scale (IQR, else
  MAD, else SD).  The fixed single-coordinate relation Π_S, rescaled by its
  IQR, is offered as one more terminal.
* A shared library of short random expressions ("components") over the
  descriptors is built first, with descriptors drawn from a soft prior (the
  random-forest importances of the two edges).  Every few generations the
  library grows by expressions drawn from the descriptors that the best
  formulas of *both* edges use.
* V and C each evolve their own population of skeletons, i.e. expressions
  over components.  Two families evolve side by side: FREE skeletons never
  contain Π_S, PI_S_ROOTED skeletons always do.
* Parents are chosen from the current Pareto front (AUC, Spearman agreement,
  simplicity of the fully expanded expression, numerical concentration) and
  from a hall of fame that keeps the best formulas ever seen.  The trajectory
  AUC along substitution paths only acts as a gate (≥ 0.65); transitions,
  balanced accuracy and valley separation are reported but never rank.
* The search stops after the minimum number of generations once neither edge
  improves for ``patience`` generations (or later at the nominal length).
  The final top lists are chosen independently for V and C; their pairing is
  audited afterwards and never changes them.

There is no cross-validation inside the search; all discovery rows are used.
Results do not depend on the number of worker processes.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import platform
import re
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import __version__
from . import grammar as g

EDGES = ("V", "C")
EDGE_NAMES = {"V": "valence", "C": "conduction"}
FAMILIES = ("FREE", "PI_S_ROOTED")
TRAJECTORY_GATE = 0.65
RATIO_EPSILON = 1e-9   # protected ratio a/(|b| + RATIO_EPSILON * Q75|b|) of the search
CHECKPOINT = "checkpoint.pkl"
RUN_RECORD = "run_record.json"

# Descriptor names containing these tokens describe band energies, labels or
# margins and never enter a formula.
LABEL_TOKENS = (
    "target", "label", "strict_label", "directness", "direct_label", "indirect_label",
    "is_direct", "is_indirect", "y_true", "y_pred", "prediction", "probability",
    "ordering_margin", "edge_stability_margin", "delta_e_directness", "canonical_gap_ev",
    "band_gap", "gap_ev", "fundamental_gap", "canonical_gap", "vbm_energy", "cbm_energy",
    "fermi", "classification", "lambda_v", "lambda_c", "valence_margin",
    "conduction_margin", "valley_margin", "edge_margin", "vbm", "cbm",
)

# Descriptor families, used only to describe formulas (first matching family).
FAMILY_PATTERNS: dict[str, tuple[str, ...]] = {
    "elemental_context": (
        r"^elem_", r"ionization", r"polariz", r"mendeleev", r"pauling", r"electronegat",
    ),
    "d_onsite_filling": (r"onsite", r"d_filling", r"d_electron", r"valence_filling"),
    "crystal_field_symmetry": (r"crystal_field", r"symmetry", r"octahedral_shape"),
    "bx_hybridization": (r"t_pd", r"BX_", r"radial_overlap", r"hybrid"),
    "bb_network": (r"^BB_", r"BB_network", r"B_Bprime"),
    "xx_network": (r"^XX_", r"XX_network", r"t_pp"),
    "local_geometry": (r"BX6_", r"BXB_angle", r"XBX_angle", r"distortion", r"offcenter",
                       r"radius_ratio", r"cage_"),
    "network_topology": (r"path", r"edge_fraction", r"corner_fraction", r"connectivity",
                         r"graph_"),
    "configurational_disorder": (r"entropy", r"configuration", r"composition_disorder",
                                 r"variance"),
    "variability_dephasing": (r"_std$", r"_cv$", r"_skew$", r"anisotropy", r"dephasing"),
    "chemical_mismatch": (r"mismatch", r"compatibility", r"signed_drift"),
}

# Grammar.  Leaves are descriptors ("feature"), components and the Π_S terminal;
# its op name "rc8" is kept because formula ids hash the tree.
COEFFICIENTS = (-3.0, -2.0, -1.0, -2.0 / 3.0, -0.5, 0.5, 2.0 / 3.0, 1.0, 2.0, 3.0)
UNARY_OPS = ("abs", "ssqrt", "square")
BINARY_OPS = ("add", "sub", "mul", "ratio", "sgeom")
LEAF_OPS = {"feature", "component", "rc8"}
SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "ratio": "/p", "sgeom": "sgeom"}


class StopRequested(Exception):
    """Raised after the checkpoint of the generation the caller asked to stop at."""


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


# ------------------------------------------------------------------ settings


@dataclass(frozen=True)
class EdgeSettings:
    seed: int
    initial_components: int     # size of the first shared component library
    feedback_components: int    # components added at every feedback step
    population: int             # skeletons per edge and generation
    min_generations: int
    nominal_generations: int
    max_generations: int
    patience: int               # generations without improvement before stopping
    feedback_interval: int
    parent_pool: int
    pareto_preselect: int
    pareto_size: int
    top_n: int
    component_depth: int
    skeleton_depth: int
    skeleton_leaves: int
    pi_s_epsilon: float         # protected-ratio regularizer inside Π_S
    jobs: int
    task_size: int

    def scientific(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("jobs")
        out.pop("task_size")
        return out


EDGE_PROFILES: dict[str, dict[str, Any]] = {
    "standard": {"initial_components": 12000, "feedback_components": 2000,
                 "population": 24000, "min_generations": 40, "nominal_generations": 60,
                 "max_generations": 80, "patience": 15, "feedback_interval": 5,
                 "parent_pool": 5000, "pareto_preselect": 4000, "pareto_size": 1600,
                 "top_n": 30, "component_depth": 3, "skeleton_depth": 5,
                 "skeleton_leaves": 6, "task_size": 128},
    "small": {"initial_components": 256, "feedback_components": 32, "population": 96,
              "min_generations": 2, "nominal_generations": 3, "max_generations": 4,
              "patience": 2, "feedback_interval": 2, "parent_pool": 64,
              "pareto_preselect": 96, "pareto_size": 64, "top_n": 12,
              "component_depth": 2, "skeleton_depth": 3, "skeleton_leaves": 4,
              "task_size": 32},
}


def load_edge_settings(path: Path, **overrides: Any) -> EdgeSettings:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    values.pop("description", None)
    profile = values.pop("profile", "standard")
    merged = {"seed": 56, "pi_s_epsilon": 1e-6, "jobs": 1, **EDGE_PROFILES[profile], **values}
    merged.update({k: v for k, v in overrides.items() if v is not None})
    unknown = set(merged) - {f.name for f in fields(EdgeSettings)}
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    return EdgeSettings(**merged)


# ---------------------------------------------------------------- utilities


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    def clean(x: Any) -> Any:
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")


def write_pickle(path: Path, value: Any) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    tmp = Path(name)
    try:
        with tmp.open("wb") as f:
            pickle.dump(value, f, protocol=5)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def finite(r: Mapping[str, Any], key: str, default: float) -> float:
    """A record field as a finite float, else ``default``."""
    try:
        x = float(r.get(key, default))
    except Exception:
        return default
    return x if math.isfinite(x) else default


def feature_family(name: str) -> str:
    for family, patterns in FAMILY_PATTERNS.items():
        if any(re.search(p, name, flags=re.I) for p in patterns):
            return family
    return "other"


def feature_allowed(name: str) -> bool:
    low = name.lower()
    return not any(t.lower() in low for t in LABEL_TOKENS)


def robust_scale_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divide every column by its IQR (else 1.4826 MAD, else SD, else 1); no centring."""
    q25 = np.nanquantile(X, 0.25, axis=0)
    q75 = np.nanquantile(X, 0.75, axis=0)
    s = q75 - q25
    mad = np.nanmedian(np.abs(X - np.nanmedian(X, axis=0)), axis=0) * 1.4826
    std = np.nanstd(X, axis=0)
    s = np.where(np.isfinite(s) & (s > 1e-12), s, mad)
    s = np.where(np.isfinite(s) & (s > 1e-12), s, std)
    s = np.where(np.isfinite(s) & (s > 1e-12), s, 1.0)
    Xs = X / s[None, :]
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.ascontiguousarray(Xs, dtype=np.float64), s.astype(np.float64)


# ------------------------------------------------------------- edge targets


def trajectory_of(sample_id: str) -> str:
    return re.sub(r"_k\d+$", "", str(sample_id))


def k_index_of(sample_id: str, row: Mapping[str, Any]) -> int:
    for key in ("k", "k_index", "concentration_index"):
        if key in row:
            try:
                return int(float(row[key]))
            except Exception:
                pass
    m = re.search(r"_k(\d+)$", str(sample_id))
    return int(m.group(1)) if m else -1


def parse_kpoint(v: Any) -> tuple[float, float, float] | None:
    """Fractional k point reduced to [0, 1), or None."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 3:
        arr = np.asarray(v[:3], dtype=float)
    else:
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        try:
            x = ast.literal_eval(s)
            if not (isinstance(x, (list, tuple, np.ndarray)) and len(x) >= 3):
                return None
            arr = np.asarray(x[:3], dtype=float)
        except Exception:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            if len(nums) < 3:
                return None
            arr = np.asarray(list(map(float, nums[:3])), dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    return tuple(float(x % 1.0) for x in arr)


def exact_k_key(k: tuple[float, float, float] | None, ndigits: int = 6) -> str:
    if k is None:
        return "MISSING"
    vals = [round((x % 1.0), ndigits) for x in k]
    vals = [0.0 if abs(x - 1.0) < 10**(-ndigits) or abs(x) < 10**(-ndigits) else x
            for x in vals]
    return "[" + ",".join(f"{x:.{ndigits}f}" for x in vals) + "]"


def star_k_key(k: tuple[float, float, float] | None, ndigits: int = 6) -> str:
    """Key invariant under inversion and coordinate permutation (the cubic star)."""
    if k is None:
        return "MISSING"
    q = [min(x % 1.0, 1.0 - (x % 1.0)) for x in k]
    q = sorted(round(x, ndigits) for x in q)
    q = [0.0 if abs(x) < 10**(-ndigits) else x for x in q]
    return "STAR[" + ",".join(f"{x:.{ndigits}f}" for x in q) + "]"


def is_gamma(k: tuple[float, float, float] | None, tol: float = 1e-6) -> bool:
    if k is None:
        return False
    return all(min(abs(x % 1.0), abs(1.0 - (x % 1.0))) <= tol for x in k)


def _sample_key(columns: Sequence[str]) -> str | None:
    lookup = {str(c).lower(): str(c) for c in columns}
    for x in ("sample_id", "task directory", "task_directory", "task_id"):
        if x in lookup:
            return lookup[x]
    return None


def _column(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def kpoint_table(path: Path, sample_ids: Sequence[str]) -> pd.DataFrame:
    """Rows of the band-edge table for ``sample_ids``, in that order."""
    df = pd.read_csv(path, low_memory=False)
    key = _sample_key(df.columns)
    if key is None:
        raise ValueError(f"{path} has no sample key column")
    if key != "sample_id":
        df = df.rename(columns={key: "sample_id"})
    df["sample_id"] = df["sample_id"].astype(str)
    if df["sample_id"].duplicated().any():
        raise ValueError(f"{path} lists a sample more than once")
    out = pd.DataFrame({"sample_id": list(sample_ids)}).merge(df, on="sample_id", how="left",
                                                              validate="one_to_one")
    empty = out.drop(columns=["sample_id"]).isna().all(axis=1)
    if empty.any():
        raise ValueError(f"{path} misses samples: {out.sample_id[empty].tolist()[:5]}")
    return out


@dataclass
class EdgeTarget:
    edge: str
    sample_ids: np.ndarray
    trajectories: np.ndarray
    k_index: np.ndarray
    exact_key: np.ndarray
    star_key: np.ndarray
    is_gamma: np.ndarray
    available: np.ndarray      # k point known
    y: np.ndarray              # 1 = extremum at Γ, -1 = unknown
    margin: np.ndarray         # Γ-versus-non-Γ energy difference, positive = Γ wins
    has_margin: np.ndarray
    is_endpoint: np.ndarray


def edge_target(table: pd.DataFrame, edge: str) -> EdgeTarget:
    """Γ / non-Γ label and energy margin of one edge.

    Path endpoints (k = 0 or 16, or flagged) take the k point of the
    primitive-cell calculation, other states the k point of the path
    calculation.  The explicit margins S_V = E_V(Γ) - E_V(non-Γ) and
    S_C = E_C(non-Γ) - E_C(Γ) are positive when Γ wins; endpoints have none.
    """
    E = edge.upper()
    prim = _column(table, [f"primitive_{E}BM_kpoint_frac", f"primitive_{E}BM_kpoint",
                           f"primitive_{E}BM_k_point"])
    path_k = _column(table, [f"{E}BM_kpoint_frac", f"{E}BM_kpoint", f"{E}BM_k_point"])
    flag = _column(table, ["endpoint_canonicalized", "v2_is_endpoint", "is_endpoint"])
    margin_col = _column(table, ["S_V_Gamma_minus_nonGamma_eV" if E == "V"
                                 else "S_C_nonGamma_minus_Gamma_eV"])
    n = len(table)
    points: list[tuple[float, float, float] | None] = []
    endpoint = np.zeros(n, bool)
    kidx = np.full(n, -1, int)
    for i, row in table.iterrows():
        sid = str(row["sample_id"])
        kidx[i] = k_index_of(sid, row)
        flagged = False
        if flag is not None:
            v = row[flag]
            flagged = (bool(v) if isinstance(v, (bool, np.bool_))
                       else str(v).strip().lower() in {"1", "true", "yes", "y"})
        endpoint[i] = flagged or kidx[i] in {0, 16}
        column = prim if endpoint[i] else path_k
        points.append(parse_kpoint(row[column]) if column else None)
    raw = (pd.to_numeric(table[margin_col], errors="coerce").to_numpy(dtype=float)
           if margin_col is not None else np.full(n, np.nan))
    margin = np.full(n, np.nan)
    if margin_col is not None:
        for i, x in enumerate(raw):
            if not endpoint[i] and math.isfinite(float(x)):
                margin[i] = float(x)
    sid = table["sample_id"].astype(str).to_numpy()
    star = np.asarray([star_k_key(k) for k in points], dtype=object)
    gam = np.asarray([is_gamma(k) for k in points], dtype=bool)
    available = star != "MISSING"
    return EdgeTarget(
        edge=edge, sample_ids=sid,
        trajectories=np.asarray([trajectory_of(s) for s in sid], dtype=object),
        k_index=kidx, exact_key=np.asarray([exact_k_key(k) for k in points], dtype=object),
        star_key=star, is_gamma=gam, available=available,
        y=np.where(available, gam.astype(np.int8), -1).astype(np.int8),
        margin=margin, has_margin=np.isfinite(margin), is_endpoint=endpoint)


def edge_states(V: EdgeTarget, C: EdgeTarget) -> np.ndarray:
    """GG, GN, NG, NN_same or NN_diff from the exact k points of both edges."""
    gamma = exact_k_key((0, 0, 0))
    ve, ce = V.exact_key, C.exact_key
    return np.where((ve == "MISSING") | (ce == "MISSING"), "UNKNOWN",
                    np.where(ve == ce, np.where(ve == gamma, "GG", "NN_same"),
                             np.where(ve == gamma, "GN", np.where(ce == gamma, "NG", "NN_diff"))))


def target_table(V: EdgeTarget, C: EdgeTarget, states: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": V.sample_ids, "trajectory_id": V.trajectories, "k_index": V.k_index,
        "is_endpoint": V.is_endpoint,
        "V_kpoint": V.exact_key, "V_star": V.star_key, "V_gamma": V.y, "V_margin_eV": V.margin,
        "C_kpoint": C.exact_key, "C_star": C.star_key, "C_gamma": C.y, "C_margin_eV": C.margin,
        "edge_state": states})


def target_summary(t: EdgeTarget) -> dict[str, Any]:
    return {"rows": int(len(t.y)), "with_kpoint": int(t.available.sum()),
            "gamma": int(np.sum(t.available & t.is_gamma)),
            "non_gamma": int(np.sum(t.available & ~t.is_gamma)),
            "valley_stars": int(len(set(t.star_key[t.available].tolist()))),
            "with_margin": int(t.has_margin.sum())}


# ---------------------------------------------------------------- inputs


@dataclass
class EdgeData:
    X: np.ndarray               # scaled discovery descriptors, (sample, descriptor)
    scales: np.ndarray
    names: list[str]
    source_index: np.ndarray    # column of each discovery descriptor in the full matrix
    sample_ids: np.ndarray
    targets: dict[str, EdgeTarget]
    states: np.ndarray
    pi_s: np.ndarray | None     # Π_S on the discovery rows, divided by its IQR
    prior: np.ndarray
    checksums: dict[str, str]


def load_matrix(protocol: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Descriptor matrix, sample ids and descriptor names of a protocol directory."""
    protocol = Path(protocol)
    X = np.asarray(np.load(protocol / "arrays" / "X_full_clean_float64.npy", mmap_mode="r",
                           allow_pickle=False), dtype=np.float64)
    sid = np.load(protocol / "arrays" / "sample_id_unicode.npy", allow_pickle=False).astype(str)
    names = list(map(str, json.loads((protocol / "features" / "formula_full_feature_names.json")
                                     .read_text(encoding="utf-8"))))
    if X.shape != (len(sid), len(names)):
        raise ValueError(f"descriptor matrix {X.shape} does not match ids and names")
    return X, sid, names


def pi_s_values(X: np.ndarray, formula: Path, scales: Path, epsilon: float,
                rows: np.ndarray, fit_on_rows: bool = False) -> np.ndarray:
    """Π_S on ``rows`` of ``X`` (with its own descriptor scales).

    The protected-ratio regularizer uses the denominator quartile over all
    rows of ``X``, or only over ``rows`` when ``fit_on_rows``.
    """
    record = json.loads(Path(formula).read_text(encoding="utf-8"))
    tree = record.get("ast_json") or record.get("expanded_ast_json") or record.get("ast")
    tree = json.loads(tree) if isinstance(tree, str) else tree
    s = np.load(scales, allow_pickle=False).astype(float)
    if len(s) != X.shape[1]:
        raise ValueError("Π_S scales do not match the descriptor matrix")
    Xfs = X / s[None, :]
    if fit_on_rows:
        values, _ = g.evaluate_expression(tree, Xfs[rows].T, epsilon=epsilon)
        return np.asarray(values, dtype=np.float64)
    values, _ = g.evaluate_expression(tree, Xfs.T, epsilon=epsilon)
    return np.asarray(values[rows], dtype=np.float64)


def iqr_scale(values: np.ndarray) -> float:
    s = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    return s if math.isfinite(s) and s > 1e-12 else float(np.std(values) or 1.0)


def soft_prior(names: Sequence[str], path: Path | None) -> np.ndarray:
    """Sampling weight 1 + 2 * importance / max importance from a random-forest table."""
    w = np.ones(len(names), float)
    if path is None or not Path(path).is_file():
        return w / w.sum()
    df = pd.read_csv(path)
    imp = {str(a): float(b) for a, b in zip(df["feature"],
                                            pd.to_numeric(df["importance"], errors="coerce"))
           if math.isfinite(float(b))}
    if imp:
        mx = max(imp.values()) or 1.0
        for i, n in enumerate(names):
            w[i] = 1.0 + 2.0 * max(0.0, imp.get(n, 0.0)) / mx
    return w / w.sum()


def load_edge_data(protocol: Path, samples: Path, kpoints: Path, settings: EdgeSettings, *,
                   pi_s_formula: Path | None = None, pi_s_scales: Path | None = None,
                   prior_valence: Path | None = None, prior_conduction: Path | None = None,
                   pi_s_fit_on_samples: bool = False) -> EdgeData:
    """Discovery rows, targets, the Π_S terminal and the sampling prior.

    ``pi_s_fit_on_samples`` fits the protected-ratio regularizer of Π_S on
    the discovery rows instead of all rows of the matrix (the convention of
    the published edge coordinates; the search used all rows).
    """
    X, sid, names = load_matrix(protocol)
    ids = pd.read_csv(samples)["sample_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample ids in the sample list")
    index = {s: i for i, s in enumerate(sid)}
    missing = [s for s in ids if s not in index]
    if missing:
        raise ValueError(f"samples missing from the protocol: {missing[:5]}")
    rows = np.asarray([index[s] for s in ids], dtype=int)
    keep = np.flatnonzero([feature_allowed(n) for n in names])
    Xs, scales = robust_scale_columns(X[rows][:, keep])
    table = kpoint_table(kpoints, sid[rows].tolist())
    V, C = edge_target(table, "V"), edge_target(table, "C")
    pi_s = None
    if pi_s_formula is not None:
        raw = pi_s_values(X, pi_s_formula, pi_s_scales, settings.pi_s_epsilon, rows,
                          pi_s_fit_on_samples)
        # Only the ordering of Π_S is used, not its threshold.
        pi_s = np.asarray(raw / iqr_scale(raw), dtype=np.float64)
    prior = (soft_prior([names[i] for i in keep], prior_valence)
             + soft_prior([names[i] for i in keep], prior_conduction)) / 2
    prior /= prior.sum()
    files = {"protocol_matrix": Path(protocol) / "arrays" / "X_full_clean_float64.npy",
             "samples": samples, "kpoints": kpoints, "pi_s_formula": pi_s_formula,
             "pi_s_scales": pi_s_scales, "prior_valence": prior_valence,
             "prior_conduction": prior_conduction}
    return EdgeData(X=Xs, scales=scales, names=[names[i] for i in keep], source_index=keep,
                    sample_ids=sid[rows], targets={"V": V, "C": C}, states=edge_states(V, C),
                    pi_s=pi_s, prior=prior,
                    checksums={k: sha256_file(Path(p)) for k, p in files.items()
                               if p is not None and Path(p).is_file()})


# ------------------------------------------------------------------ grammar


def tree_key(node: Mapping[str, Any]) -> str:
    return json.dumps(node, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tree_hash(node: Mapping[str, Any]) -> str:
    return hashlib.sha1(tree_key(node).encode("utf-8")).hexdigest()


def tree_depth(node: Mapping[str, Any]) -> int:
    op = node["op"]
    if op in LEAF_OPS:
        return 1
    if op in UNARY_OPS or op == "scale":
        return 1 + tree_depth(node["arg"])
    return 1 + max(tree_depth(node["left"]), tree_depth(node["right"]))


def tree_leaves(node: Mapping[str, Any]) -> int:
    op = node["op"]
    if op in LEAF_OPS:
        return 1
    if op in UNARY_OPS or op == "scale":
        return tree_leaves(node["arg"])
    return tree_leaves(node["left"]) + tree_leaves(node["right"])


def tree_ops(node: Mapping[str, Any]) -> int:
    op = node["op"]
    if op in LEAF_OPS:
        return 0
    if op in UNARY_OPS or op == "scale":
        return 1 + tree_ops(node["arg"])
    return 1 + tree_ops(node["left"]) + tree_ops(node["right"])


def component_ids(node: Mapping[str, Any]) -> list[int]:
    op = node["op"]
    if op == "component":
        return [int(node["id"])]
    if op in {"feature", "rc8"}:
        return []
    if op in UNARY_OPS or op == "scale":
        return component_ids(node["arg"])
    return component_ids(node["left"]) + component_ids(node["right"])


def feature_ids(node: Mapping[str, Any], components: Sequence[Mapping[str, Any]]) -> list[int]:
    op = node["op"]
    if op == "feature":
        return [int(node["id"])]
    if op == "component":
        return feature_ids(components[int(node["id"])], components)
    if op == "rc8":
        return []
    if op in UNARY_OPS or op == "scale":
        return feature_ids(node["arg"], components)
    return feature_ids(node["left"], components) + feature_ids(node["right"], components)


def expanded_stats(node: Mapping[str, Any],
                   components: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Operators, leaves, ratios and depth with components expanded (Π_S is one leaf)."""
    op = node["op"]
    if op in {"feature", "rc8"}:
        return {"ops": 0.0, "leaves": 1.0, "ratios": 0.0, "depth": 1.0}
    if op == "component":
        return expanded_stats(components[int(node["id"])], components)
    if op in UNARY_OPS or op == "scale":
        q = expanded_stats(node["arg"], components)
        return {"ops": q["ops"] + 1.0, "leaves": q["leaves"], "ratios": q["ratios"],
                "depth": q["depth"] + 1.0}
    a = expanded_stats(node["left"], components)
    b = expanded_stats(node["right"], components)
    return {"ops": a["ops"] + b["ops"] + 1.0, "leaves": a["leaves"] + b["leaves"],
            "ratios": a["ratios"] + b["ratios"] + (1.0 if op == "ratio" else 0.0),
            "depth": max(a["depth"], b["depth"]) + 1.0}


def expanded_complexity(node: Mapping[str, Any],
                        components: Sequence[Mapping[str, Any]]) -> tuple[float, dict[str, float]]:
    q = expanded_stats(node, components)
    return float(q["ops"] + 0.35 * q["leaves"] + 0.80 * q["ratios"]), q


def contains_pi_s(node: Mapping[str, Any], components: Sequence[Mapping[str, Any]]) -> bool:
    op = node["op"]
    if op == "rc8":
        return True
    if op == "component":
        return contains_pi_s(components[int(node["id"])], components)
    if op == "feature":
        return False
    if op in UNARY_OPS or op == "scale":
        return contains_pi_s(node["arg"], components)
    return contains_pi_s(node["left"], components) or contains_pi_s(node["right"], components)


def protected_ratio(a: np.ndarray, b: np.ndarray, epsilon: float = RATIO_EPSILON) -> np.ndarray:
    den = np.abs(b)
    q75 = float(np.quantile(den, 0.75))
    eps = epsilon * (q75 if q75 > 0 and math.isfinite(q75) else 1.0)
    return a / (den + eps)


def evaluate(node: Mapping[str, Any], X: np.ndarray, components: np.ndarray | None,
             pi_s: np.ndarray | None, ratio_epsilon: float = RATIO_EPSILON) -> np.ndarray:
    """Values of a tree; X is (sample, descriptor), components (component, sample)."""
    op = node["op"]
    if op == "feature":
        return X[:, int(node["id"])]
    if op == "component":
        if components is None:
            raise ValueError("component values unavailable")
        return components[int(node["id"])]
    if op == "rc8":
        if pi_s is None:
            raise ValueError("the Π_S terminal is unavailable")
        return pi_s
    if op == "scale":
        return float(node["c"]) * evaluate(node["arg"], X, components, pi_s, ratio_epsilon)
    if op in UNARY_OPS:
        a = evaluate(node["arg"], X, components, pi_s, ratio_epsilon)
        with np.errstate(all="ignore"):
            if op == "abs":
                return np.abs(a)
            if op == "ssqrt":
                return np.sign(a) * np.sqrt(np.abs(a))
            # signed square, base clipped at 1e6
            return np.sign(a) * np.minimum(np.abs(a), 1e6) ** 2
    a = evaluate(node["left"], X, components, pi_s, ratio_epsilon)
    b = evaluate(node["right"], X, components, pi_s, ratio_epsilon)
    with np.errstate(all="ignore"):
        if op == "add":
            return a + b
        if op == "sub":
            return a - b
        if op == "mul":
            return a * b
        if op == "ratio":
            return protected_ratio(a, b, ratio_epsilon)
        if op == "sgeom":
            p = a * b
            return np.sign(p) * np.sqrt(np.abs(p))
    raise ValueError(f"unknown operator {op}")


def formula_text(node: Mapping[str, Any], names: Sequence[str],
                 components: Sequence[Mapping[str, Any]], expand: bool = True) -> str:
    op = node["op"]
    if op == "feature":
        return names[int(node["id"])]
    if op == "rc8":
        return "Pi_S"
    if op == "component":
        cid = int(node["id"])
        return formula_text(components[cid], names, components, True) if expand else f"C[{cid}]"
    if op == "scale":
        return f"({node['c']:.6g})*({formula_text(node['arg'], names, components, expand)})"
    if op in UNARY_OPS:
        return f"{op}({formula_text(node['arg'], names, components, expand)})"
    a = formula_text(node["left"], names, components, expand)
    b = formula_text(node["right"], names, components, expand)
    return f"({a} {SYMBOLS[op]} {b})"


def node_paths(node: Mapping[str, Any], path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """All node paths in pre-order, root first."""
    out = [path]
    op = node["op"]
    if op in LEAF_OPS:
        return out
    if op in UNARY_OPS or op == "scale":
        return out + node_paths(node["arg"], path + ("arg",))
    return out + node_paths(node["left"], path + ("left",)) + node_paths(node["right"],
                                                                        path + ("right",))


def replace_at(node: Mapping[str, Any], path: Sequence[str],
               new: Mapping[str, Any]) -> dict[str, Any]:
    if not path:
        return dict(new)
    out = dict(node)
    out[path[0]] = replace_at(out[path[0]], path[1:], new)
    return out


# --------------------------------------------------------------- components


def random_component(rng: np.random.Generator, nfeat: int, weights: np.ndarray,
                     depth: int) -> dict[str, Any]:
    if depth <= 1 or rng.random() < 0.42:
        node = {"op": "feature", "id": int(rng.choice(nfeat, p=weights))}
        if rng.random() < 0.25:
            node = {"op": rng.choice(UNARY_OPS).item(), "arg": node}
        return node
    if rng.random() < 0.22:
        return {"op": rng.choice(UNARY_OPS).item(),
                "arg": random_component(rng, nfeat, weights, depth - 1)}
    return {"op": rng.choice(BINARY_OPS).item(),
            "left": random_component(rng, nfeat, weights, depth - 1),
            "right": random_component(rng, nfeat, weights, depth - 1)}


def component_table(components: Sequence[Mapping[str, Any]], names: Sequence[str]) -> pd.DataFrame:
    rows = []
    for cid, a in enumerate(components):
        fids = sorted(set(feature_ids(a, components)))
        rows.append({"component_id": cid, "ast_json": tree_key(a),
                     "formula": formula_text(a, names, components, True),
                     "operator_nodes": tree_ops(a), "tree_depth": tree_depth(a),
                     "raw_feature_count": len(fids),
                     "raw_features": ";".join(names[i] for i in fids),
                     "families": ";".join(sorted(set(feature_family(names[i]) for i in fids))),
                     "contains_pi_s": contains_pi_s(a, components)})
    return pd.DataFrame(rows)


def grow_components(X: np.ndarray, settings: EdgeSettings, rng: np.random.Generator,
                    weights: np.ndarray, pi_s: np.ndarray | None,
                    components: list[dict[str, Any]] | None = None,
                    values: np.ndarray | None = None) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Build the shared library, or add ``feedback_components`` to an existing one.

    A new library starts with every descriptor and the Π_S terminal.  Random
    expressions are accepted when finite, not near-constant, not too large
    (99th percentile of |value| <= 1e10), new as trees and, as standardized
    value patterns, new among the last 512 accepted.  Values are stored in
    single precision.
    """
    nfeat = X.shape[1]
    w = np.maximum(np.asarray(weights, dtype=float).copy(), 1e-12)
    w /= w.sum()
    asts = [] if components is None else list(components)
    vals = [] if values is None else [row.copy() for row in values]
    seen = {tree_hash(a) for a in asts}
    target = (settings.initial_components if components is None
              else len(asts) + settings.feedback_components)
    if components is None:
        for i in range(nfeat):
            a = {"op": "feature", "id": i}
            asts.append(a)
            vals.append(X[:, i].astype(np.float32))
            seen.add(tree_hash(a))
        if pi_s is not None:
            a = {"op": "rc8"}
            asts.append(a)
            vals.append(np.asarray(pi_s, dtype=np.float32))
            seen.add(tree_hash(a))
    recent: list[str | None] = [None] * len(asts)   # value fingerprints of new components
    attempts, max_attempts = 0, max(1000, (target - len(asts)) * 30)
    while len(asts) < target and attempts < max_attempts:
        attempts += 1
        a = random_component(rng, nfeat, w, settings.component_depth)
        h = tree_hash(a)
        if h in seen:
            continue
        try:
            v = evaluate(a, X, None, pi_s)
        except Exception:
            continue
        if not np.all(np.isfinite(v)):
            continue
        q = np.quantile(np.abs(v), [0.5, 0.99])
        if q[1] > 1e10 or np.std(v) < 1e-10:
            continue
        sig = np.round((v - np.mean(v)) / (np.std(v) + 1e-12), 4)
        vh = hashlib.blake2b(sig.tobytes(), digest_size=8).hexdigest()
        if vh in recent[-512:]:
            continue
        asts.append(a)
        recent.append(vh)
        vals.append(np.asarray(v, dtype=np.float32))
        seen.add(h)
    return asts, np.asarray(vals, dtype=np.float32)


# ---------------------------------------------------------------- skeletons


def random_leaf(rng: np.random.Generator, ncomp: int, pi_s_id: int | None) -> dict[str, Any]:
    """A component leaf other than the Π_S component."""
    if pi_s_id is None:
        cid = int(rng.integers(0, ncomp))
    else:
        cid = int(rng.integers(0, ncomp - 1))
        if cid >= int(pi_s_id):
            cid += 1
    return {"op": "component", "id": cid}


def random_skeleton(rng: np.random.Generator, ncomp: int, settings: EdgeSettings,
                    pi_s_id: int | None, family: str, depth: int | None = None) -> dict[str, Any]:
    def grow(z: int) -> dict[str, Any]:
        if z <= 1 or rng.random() < 0.38:
            leaf = random_leaf(rng, ncomp, pi_s_id)
            if rng.random() < 0.22:
                leaf = {"op": "scale", "c": float(rng.choice(COEFFICIENTS)), "arg": leaf}
            return leaf
        return {"op": rng.choice(BINARY_OPS).item(), "left": grow(z - 1), "right": grow(z - 1)}
    a = grow(settings.skeleton_depth if depth is None else depth)
    if family == "PI_S_ROOTED":
        root: dict[str, Any] = {"op": "component", "id": int(pi_s_id)}
        if rng.random() < 0.8:
            root = {"op": "scale", "c": float(rng.choice((0.5, 2 / 3, 1.0, 1.5, 2.0))),
                    "arg": root}
        a = {"op": rng.choice(("add", "sub")).item(), "left": root, "right": a}
    return a


def with_pi_s(a: dict[str, Any], pi_s_id: int) -> dict[str, Any]:
    return {"op": "add", "left": {"op": "component", "id": int(pi_s_id)}, "right": a}


def mutate(parent: Mapping[str, Any], rng: np.random.Generator, ncomp: int,
           settings: EdgeSettings, pi_s_id: int | None, family: str,
           components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Replace a random subtree by a small new one (65%) or rescale it."""
    a = json.loads(json.dumps(parent))
    paths = node_paths(a)
    p = paths[int(rng.integers(0, len(paths)))]
    if rng.random() < 0.65:
        new = random_skeleton(rng, ncomp, settings, pi_s_id, "FREE",
                              depth=int(rng.integers(1, 3)))
    else:
        new = {"op": "scale", "c": float(rng.choice(COEFFICIENTS)),
               "arg": dict(g.node_at(a, p))}
    a = replace_at(a, p, new)
    if family == "PI_S_ROOTED" and not contains_pi_s(a, components):
        a = with_pi_s(a, pi_s_id)
    if family == "FREE" and contains_pi_s(a, components):
        return random_skeleton(rng, ncomp, settings, pi_s_id, "FREE")
    return a


def crossover(a: Mapping[str, Any], b: Mapping[str, Any],
              rng: np.random.Generator) -> dict[str, Any]:
    aa = json.loads(json.dumps(a))
    pb, pa = node_paths(b), node_paths(aa)
    xa = pa[int(rng.integers(0, len(pa)))]
    xb = pb[int(rng.integers(0, len(pb)))]
    return replace_at(aa, xa, g.node_at(b, xb))


def valid_skeleton(a: Mapping[str, Any], settings: EdgeSettings,
                   components: Sequence[Mapping[str, Any]], family: str) -> bool:
    if tree_depth(a) > settings.skeleton_depth + 2 or tree_leaves(a) > settings.skeleton_leaves:
        return False
    return contains_pi_s(a, components) == (family == "PI_S_ROOTED")


def initial_population(rng: np.random.Generator, ncomp: int, settings: EdgeSettings,
                       pi_s_id: int | None, components: Sequence[Mapping[str, Any]]
                       ) -> list[tuple[str, dict[str, Any]]]:
    pop: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    families = ["FREE"] if pi_s_id is None else list(FAMILIES)
    attempts = 0
    while len(pop) < settings.population and attempts < settings.population * 30:
        attempts += 1
        fam = families[len(pop) % len(families)]
        a = random_skeleton(rng, ncomp, settings, pi_s_id, fam)
        if not valid_skeleton(a, settings, components, fam):
            continue
        h = tree_hash(a)
        if h in seen:
            continue
        seen.add(h)
        pop.append((fam, a))
    if len(pop) < settings.population:
        raise RuntimeError(f"initial population incomplete: {len(pop)}/{settings.population}")
    return pop


def next_population(rng: np.random.Generator, records: Sequence[Mapping[str, Any]],
                    components: Sequence[Mapping[str, Any]], ncomp: int,
                    settings: EdgeSettings, pi_s_id: int | None
                    ) -> list[tuple[str, dict[str, Any]]]:
    """Children: 16% new random skeletons, 18% crossovers, the rest mutations."""
    safe = [r for r in records if r.get("safe")]
    if not safe:
        raise RuntimeError("no safe parents")
    parents = parent_pool(safe, settings.parent_pool)
    by_family = {f: [r for r in parents if r["family"] == f] for f in FAMILIES}
    families = ["FREE"] if pi_s_id is None else list(FAMILIES)
    pop: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    attempts = 0
    while len(pop) < settings.population and attempts < settings.population * 40:
        attempts += 1
        fam = families[len(pop) % len(families)]
        pool = by_family.get(fam) or parents
        p = pool[int(rng.integers(0, len(pool)))]
        a = json.loads(p["ast_json"])
        u = rng.random()
        if u < 0.16:
            child = random_skeleton(rng, ncomp, settings, pi_s_id, fam)
        elif u < 0.34 and len(pool) > 1:
            q = pool[int(rng.integers(0, len(pool)))]
            child = crossover(a, json.loads(q["ast_json"]), rng)
            if fam == "PI_S_ROOTED" and not contains_pi_s(child, components):
                child = with_pi_s(child, pi_s_id)
            if fam == "FREE" and contains_pi_s(child, components):
                continue
        else:
            child = mutate(a, rng, ncomp, settings, pi_s_id, fam, components)
        if not valid_skeleton(child, settings, components, fam):
            continue
        h = tree_hash(child)
        if h in seen:
            continue
        seen.add(h)
        pop.append((fam, child))
    if len(pop) < settings.population:
        raise RuntimeError(f"next population incomplete: {len(pop)}/{settings.population}")
    return pop


# ------------------------------------------------------------------ metrics


def rank_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC of ``s`` for y = 1 with average ranks for ties."""
    y = np.asarray(y, dtype=np.int8)
    s = np.asarray(s, dtype=float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i + 1
        while j < len(ss) and ss[j] == ss[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    u = ranks[pos].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation with ordinal (stable-sort) ranks."""
    if len(a) < 4:
        return float("nan")

    def ranks(x: np.ndarray) -> np.ndarray:
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), float)
        r[o] = np.arange(len(x), dtype=float)
        return r
    ra, rb = ranks(a), ranks(b)
    return (float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0
            else float("nan"))


def best_threshold(y: np.ndarray, s: np.ndarray, orientation: int) -> tuple[float, float, float]:
    """Threshold of best balanced accuracy, that accuracy, and the plateau width.

    The plateau is the spread of thresholds within 0.01 of the best balanced
    accuracy, divided by the IQR of the score and capped at 10.
    """
    z = s if orientation == 1 else -s
    order = np.argsort(z, kind="mergesort")
    zz, yy = z[order], y[order]
    P = max(1, int((yy == 1).sum()))
    N = max(1, int((yy == 0).sum()))
    tp = P - np.cumsum(yy == 1)
    tn = np.cumsum(yy == 0)
    cand = np.concatenate(([0.5], 0.5 * (tp / P + tn / N)))   # [0] = all positive
    k = int(np.argmax(cand))
    best = float(cand[k])
    if k == 0:
        th = float(zz[0] - 1e-12)
    elif k >= len(zz):
        th = float(zz[-1] + 1e-12)
    else:
        j = k - 1
        th = float(zz[j] + 1e-12) if j == len(zz) - 1 else float(0.5 * (zz[j] + zz[j + 1]))
    good = np.flatnonzero(cand >= best - 0.01)
    if len(good) >= 2:
        ts = []
        for kk in good:
            if kk == 0:
                ts.append(zz[0])
            elif kk >= len(zz):
                ts.append(zz[-1])
            else:
                j = kk - 1
                ts.append(zz[j] if j == len(zz) - 1 else 0.5 * (zz[j] + zz[j + 1]))
        width = float(max(ts) - min(ts))
    else:
        width = 0.0
    iqr = float(np.quantile(z, 0.75) - np.quantile(z, 0.25))
    plateau = width / (abs(iqr) + 1e-12)
    return (th if orientation == 1 else -th), best, float(min(plateau, 10.0))


def trajectory_auc(y: np.ndarray, s: np.ndarray, traj: np.ndarray, mask: np.ndarray,
                   orientation: int) -> tuple[float, int]:
    """Mean AUC within substitution paths that contain both classes."""
    vals = []
    for t in np.unique(traj[mask]):
        ii = np.flatnonzero(mask & (traj == t))
        if len(ii) < 2 or len(np.unique(y[ii])) < 2:
            continue
        a = rank_auc(y[ii], s[ii])
        vals.append(a if orientation == 1 else 1.0 - a)
    return (float(np.mean(vals)) if vals else float("nan"), len(vals))


def transition_score(y: np.ndarray, s: np.ndarray, traj: np.ndarray, kidx: np.ndarray,
                     mask: np.ndarray, threshold: float, orientation: int) -> float:
    """Agreement of predicted and true Γ / non-Γ switches along each path."""
    pred = (s >= threshold).astype(np.int8) if orientation == 1 else (s <= threshold).astype(np.int8)
    scores = []
    for t in np.unique(traj[mask]):
        ii = np.flatnonzero(mask & (traj == t))
        ii = ii[np.argsort(kidx[ii], kind="stable")]
        if len(ii) < 2:
            continue
        yt, yp = y[ii], pred[ii]
        at = np.flatnonzero(yt[1:] != yt[:-1])
        ap = np.flatnonzero(yp[1:] != yp[:-1])
        if len(at) == 0 and len(ap) == 0:
            scores.append(1.0)
            continue
        count_pen = abs(len(at) - len(ap)) / max(1, len(at) + len(ap))
        if len(at) and len(ap):
            loc = np.mean([min(abs(int(a) - int(p)) for p in ap) for a in at]) / max(1, len(ii) - 1)
        else:
            loc = 1.0
        scores.append(max(0.0, 1.0 - 0.55 * count_pen - 0.45 * loc))
    return float(np.mean(scores)) if scores else float("nan")


def star_eta2(score: np.ndarray, target: EdgeTarget, orientation: int) -> tuple[float, int]:
    """Share of the oriented-score variance among non-Γ states explained by their valley star."""
    mask = target.available & (~target.is_gamma) & (target.star_key != "MISSING")
    if mask.sum() < 8:
        return float("nan"), 0
    z = np.asarray(score[mask], float) * int(orientation)
    grp = target.star_key[mask]
    labels = [x for x in np.unique(grp) if np.sum(grp == x) >= 2]
    use = np.isin(grp, labels)
    if len(labels) < 2 or use.sum() < 8:
        return float("nan"), len(labels)
    z, grp = z[use], grp[use]
    mu = float(np.mean(z))
    total = float(np.sum((z - mu) ** 2))
    if total <= 1e-15:
        return 0.0, len(labels)
    between = 0.0
    for lab in labels:
        zz = z[grp == lab]
        between += len(zz) * (float(np.mean(zz)) - mu) ** 2
    return float(max(0.0, min(1.0, between / total))), len(labels)


UNSAFE = {"safe": False, "auc": 0.5, "bacc": 0.5, "margin_spearman": 0.0,
          "trajectory_auc": 0.5, "transition_score": 0.0, "star_eta2": 0.0, "star_group_n": 0,
          "plateau": 0.0}


def candidate_metrics(score: np.ndarray, target: EdgeTarget, complexity: float) -> dict[str, Any]:
    if not np.all(np.isfinite(score)) or np.std(score) < 1e-12:
        return {**UNSAFE, "complexity": complexity}
    m = target.available
    yy, ss = target.y[m], score[m]
    if len(np.unique(yy)) < 2:
        return {**UNSAFE, "complexity": complexity}
    raw_auc = rank_auc(yy, ss)
    orientation = 1 if raw_auc >= 0.5 else -1
    auc = max(raw_auc, 1.0 - raw_auc)
    th, bacc, plateau = best_threshold(yy, ss, orientation)
    mm = target.has_margin & target.available
    rho = spearman(score[mm] * orientation, target.margin[mm]) if mm.sum() >= 8 else float("nan")
    ta, ntraj = trajectory_auc(target.y, score, target.trajectories, target.available,
                               orientation)
    ts = transition_score(target.y, score, target.trajectories, target.k_index,
                          target.available, th, orientation)
    eta2, star_n = star_eta2(score, target, orientation)
    # Numerical concentration: a few samples must not dominate the score.
    q50 = float(np.quantile(np.abs(score), 0.50))
    q99 = float(np.quantile(np.abs(score), 0.99))
    concentration = q99 / (q50 + 1e-12)
    penalty = 0.0
    if concentration > 1e4:
        penalty = min(1.0, math.log10(concentration / 1e4 + 1) / 3)
    if concentration > 1e8:
        penalty = 1.0
    nan = float("nan")
    return {"safe": True, "auc": float(auc), "raw_auc": float(raw_auc),
            "orientation": orientation, "threshold": float(th), "bacc": float(bacc),
            "plateau": float(plateau), "margin_spearman": float(rho) if math.isfinite(rho) else nan,
            "trajectory_auc": float(ta) if math.isfinite(ta) else nan,
            "trajectory_auc_n": int(ntraj),
            "transition_score": float(ts) if math.isfinite(ts) else nan,
            "star_eta2": float(eta2) if math.isfinite(eta2) else nan,
            "star_group_n": int(star_n), "complexity": float(complexity),
            "q99_q50": float(concentration), "physical_penalty": float(penalty)}


def _simplicity(r: Mapping[str, Any], default: float) -> float:
    return 1.0 / (1.0 + 0.035 * max(0.0, finite(r, "complexity", default)))


def _margin(r: Mapping[str, Any]) -> float:
    """Margin agreement mapped from [-1, 1] to [0, 1]."""
    return (finite(r, "margin_spearman", 0.0) + 1.0) / 2.0


def selection_score(m: Mapping[str, Any]) -> float:
    """Parent-selection score: AUC, margin agreement and simplicity; path AUC only as a gate."""
    traj = finite(m, "trajectory_auc", 0.5)
    ntraj = int(m.get("trajectory_auc_n", 0) or 0)
    deficit = max(0.0, TRAJECTORY_GATE - traj) if ntraj >= 3 else 0.0
    return (0.68 * finite(m, "auc", 0.5) + 0.22 * _margin(m) + 0.10 * _simplicity(m, 10.0)
            - 0.15 * finite(m, "physical_penalty", 0.0) - 0.30 * deficit)


def objective_vector(r: Mapping[str, Any]) -> tuple[float, ...]:
    return (finite(r, "auc", 0.5), _margin(r), _simplicity(r, 100.0),
            -finite(r, "physical_penalty", 1.0))


def recommendation_score(r: Mapping[str, Any]) -> float:
    return (0.67 * finite(r, "auc", 0.5) + 0.23 * _margin(r) + 0.10 * _simplicity(r, 10.0)
            - 0.15 * finite(r, "physical_penalty", 0.0))


# --------------------------------------------------------------- evaluation

_WORKER: dict[str, Any] = {}


def _init_worker(values: np.ndarray, X: np.ndarray, targets: Mapping[str, EdgeTarget],
                 components: list[dict[str, Any]]) -> None:
    _WORKER.update(values=values, X=X, targets=targets, components=components)


def _evaluate_batch(payload: tuple[str, list[tuple[int, str, dict[str, Any]]]]
                    ) -> list[dict[str, Any]]:
    edge, batch = payload
    target, components = _WORKER["targets"][edge], _WORKER["components"]
    out = []
    for idx, fam, a in batch:
        skeleton = (tree_ops(a) + 0.5 * tree_leaves(a)
                    + 0.7 * sum(1 for p in node_paths(a) if g.node_at(a, p).get("op") == "ratio"))
        comp, est = expanded_complexity(a, components)
        try:
            s = evaluate(a, _WORKER["X"], _WORKER["values"], None)
            m = candidate_metrics(np.asarray(s, dtype=float), target, comp)
        except Exception:
            m = {**UNSAFE, "complexity": comp}
        m.update({"skeleton_complexity": float(skeleton),
                  "expanded_operator_nodes": int(est["ops"]),
                  "expanded_leaf_count": int(est["leaves"]),
                  "expanded_ratio_nodes": int(est["ratios"]),
                  "expanded_depth": int(est["depth"]),
                  "candidate_index": idx, "family": fam, "ast_json": tree_key(a),
                  "formula_id": tree_hash(a)})
        m["selection_score"] = selection_score(m)
        out.append(m)
    return out


class Evaluator:
    """Fork pool over skeleton batches; rebuilt whenever the component library grows."""

    def __init__(self, values: np.ndarray, X: np.ndarray, targets: Mapping[str, EdgeTarget],
                 components: list[dict[str, Any]], jobs: int, task_size: int):
        self.task_size = int(task_size)
        self.pool = None
        if jobs <= 1:
            _init_worker(values, X, targets, components)
        else:
            self.pool = mp.get_context("fork").Pool(
                processes=int(jobs), initializer=_init_worker,
                initargs=(values, X, targets, components))

    def evaluate(self, edge: str, pop: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        payload = [(i, f, a) for i, (f, a) in enumerate(pop)]
        batches = [(edge, payload[i:i + self.task_size])
                   for i in range(0, len(payload), self.task_size)]
        chunks = (self.pool.map(_evaluate_batch, batches, chunksize=1) if self.pool
                  else [_evaluate_batch(b) for b in batches])
        return [x for c in chunks for x in c]

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None


# ----------------------------------------------------------------- archives


def dedup(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Safe records, one per formula (highest selection score; widest generation range)."""
    out: dict[str, dict[str, Any]] = {}
    for rr in records:
        if not bool(rr.get("safe", False)):
            continue
        r = dict(rr)
        fid = str(r.get("formula_id", tree_hash(json.loads(r["ast_json"]))))
        old = out.get(fid)
        if old is None or finite(r, "selection_score", -1e99) > finite(old, "selection_score",
                                                                        -1e99):
            out[fid] = r
        else:
            if "first_seen_generation" in r:
                old["first_seen_generation"] = min(
                    int(old.get("first_seen_generation", r["first_seen_generation"])),
                    int(r["first_seen_generation"]))
            if "last_seen_generation" in r:
                old["last_seen_generation"] = max(
                    int(old.get("last_seen_generation", r["last_seen_generation"])),
                    int(r["last_seen_generation"]))
    return list(out.values())


def _ranked(pool: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(pool, key=lambda r: (finite(r, "selection_score", 0), finite(r, "auc", 0.5),
                                       -finite(r, "complexity", 100)), reverse=True)


def _top(pool: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return sorted(pool, key=lambda r: finite(r, key, -1e99), reverse=True)


def diverse_select(records: Sequence[Mapping[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Bounded archive that keeps the leaders of every primary objective and family."""
    pool = dedup(records)
    if len(pool) <= limit:
        return _ranked(pool)
    chosen: dict[str, dict[str, Any]] = {}

    def add(seq: Sequence[dict[str, Any]], n: int) -> None:
        for r in list(seq)[:n]:
            chosen[str(r["formula_id"])] = r
    per = max(48, min(160, limit // 6))
    for k in ("auc", "margin_spearman", "selection_score"):
        add(_top(pool, k), per)
    add(sorted(pool, key=lambda r: (finite(r, "complexity", 1e99),
                                    finite(r, "physical_penalty", 1.0))), per)
    for fam in FAMILIES:
        q = [r for r in pool if r.get("family") == fam]
        add(_top(q, "selection_score"), max(24, per // 3))
        add(_top(q, "auc"), max(24, per // 3))
    best_auc = max(finite(r, "auc", 0.5) for r in pool)
    near = [r for r in pool if finite(r, "auc", 0.5) >= best_auc - 0.010]
    add(sorted(near, key=lambda r: (finite(r, "complexity", 1e9), finite(r, "physical_penalty", 1.0),
                                    -finite(r, "margin_spearman", 0.0))), max(64, per))
    add(_top(pool, "trajectory_auc"), max(8, per // 8))
    if len(chosen) < limit:
        add(_ranked(pool), limit - len(chosen))
    return _ranked(list(chosen.values()))[:limit]


def pareto_front(records: Sequence[Mapping[str, Any]], preselect: int,
                 limit: int) -> list[dict[str, Any]]:
    safe = [dict(r) for r in records if bool(r.get("safe", False))]
    if not safe:
        return []
    chosen = {}
    per = max(80, preselect // 5)
    for k, larger in (("auc", True), ("margin_spearman", True), ("complexity", False),
                      ("physical_penalty", False), ("selection_score", True)):
        vals = (sorted(safe, key=lambda r: finite(r, k, -1e99), reverse=True) if larger
                else sorted(safe, key=lambda r: finite(r, k, 1e99)))
        for r in vals[:per]:
            chosen[r["formula_id"]] = r
    pool = list(chosen.values())
    V = np.asarray([objective_vector(r) for r in pool], dtype=np.float64)
    n = len(pool)
    dominated = np.zeros(n, dtype=bool)
    for st in range(0, n, 256):
        A = V[st:min(n, st + 256)]
        ge = np.all(V[:, None, :] >= A[None, :, :] - 1e-12, axis=2)
        gt = np.any(V[:, None, :] > A[None, :, :] + 1e-12, axis=2)
        dominated[st:st + len(A)] = np.any(ge & gt, axis=0)
    return diverse_select([pool[i] for i in np.flatnonzero(~dominated)], limit)


def hall_of_fame(previous: Sequence[Mapping[str, Any]], current: Sequence[Mapping[str, Any]],
                 generation: int, limit: int) -> list[dict[str, Any]]:
    stamped = []
    for rr in current:
        r = dict(rr)
        r.setdefault("first_seen_generation", generation)
        r["last_seen_generation"] = generation
        stamped.append(r)
    return diverse_select(dedup(list(previous) + stamped), limit)


def parent_pool(records: Sequence[Mapping[str, Any]], limit: int) -> list[dict[str, Any]]:
    pool = dedup(records)
    if len(pool) <= limit:
        return pool
    chosen: dict[str, dict[str, Any]] = {}

    def add(seq: Sequence[dict[str, Any]], n: int) -> None:
        for r in list(seq)[:n]:
            chosen[str(r["formula_id"])] = r
    add(_top(pool, "selection_score"), max(96, limit // 2))
    add(_top(pool, "auc"), max(48, limit // 8))
    add(_top(pool, "margin_spearman"), max(48, limit // 8))
    best_auc = max(finite(r, "auc", 0.5) for r in pool)
    near = [r for r in pool if finite(r, "auc", 0.5) >= best_auc - 0.012]
    add(sorted(near, key=lambda r: (finite(r, "complexity", 1e9),
                                    finite(r, "physical_penalty", 1.0))), max(48, limit // 8))
    if len(chosen) < limit:
        add(_top(pool, "selection_score"), limit - len(chosen))
    return list(chosen.values())[:limit]


def best_values(front: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    def mx(k: str, default: float) -> float:
        vals = []
        for r in front:
            try:
                x = float(r.get(k, float("nan")))
            except Exception:
                continue
            if math.isfinite(x):
                vals.append(x)
        return max(vals) if vals else default
    return {"auc": mx("auc", 0.5), "trajectory_auc": mx("trajectory_auc", 0.5),
            "transition_score": mx("transition_score", 0.0),
            "margin_spearman": mx("margin_spearman", 0.0), "star_eta2": mx("star_eta2", 0.0),
            "bacc": mx("bacc", 0.5), "selection_score": mx("selection_score", 0.0)}


def improved(current: Mapping[str, float], best: Mapping[str, float] | None,
             tol: float = 1e-5) -> tuple[bool, dict[str, float]]:
    """Did AUC, margin agreement or selection score improve; and the running best."""
    if best is None:
        return True, dict(current)
    up = any(float(current.get(k, -1e99)) > float(best.get(k, -1e99)) + tol
             for k in ("auc", "margin_spearman", "selection_score"))
    return up, {k: max(float(best.get(k, -1e99)), float(current.get(k, -1e99))) for k in current}


def finalists(hof: Sequence[Mapping[str, Any]], live: Sequence[Mapping[str, Any]],
              limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Top list, objective champions and simple near-optimal formulas of one edge.

    Eligible formulas are within 0.015 AUC of the best (at least 0.70), pass
    the trajectory gate and are not numerically concentrated; the top list
    ranks them by the recommendation score, followed by the AUC and margin
    leaders, simple near-optimal formulas and the leaders of each family.
    """
    pool = dedup(list(hof) + list(live))
    if not pool:
        return [], [], []
    for r in pool:
        r["final_recommendation_score"] = recommendation_score(r)
    best_auc = max(finite(r, "auc", 0.5) for r in pool)
    auc_gate = max(0.70, best_auc - 0.015)
    near = [r for r in pool if finite(r, "auc", 0.5) >= auc_gate]

    def passes(r: Mapping[str, Any], gate: float) -> bool:
        return ((int(r.get("trajectory_auc_n", 0) or 0) < 3
                 or finite(r, "trajectory_auc", 0.5) >= gate)
                and finite(r, "physical_penalty", 0.0) <= 0.85)
    traj_gate, relaxed = TRAJECTORY_GATE, False
    eligible = [r for r in near if passes(r, traj_gate)]
    if not eligible:   # relax the path gate to 0.60 only if nothing passes
        relaxed, traj_gate = True, 0.60
        eligible = [r for r in near if passes(r, traj_gate)]
    if not eligible:
        eligible = near if near else pool
    eligible_ids = {str(r["formula_id"]) for r in eligible}
    for r in pool:
        r["recommendation_eligible"] = str(r["formula_id"]) in eligible_ids
        r["recommendation_auc_gate"] = float(auc_gate)
        r["recommendation_trajectory_gate"] = float(traj_gate)
        r["recommendation_trajectory_gate_relaxed"] = bool(relaxed)
    simple = sorted(near, key=lambda r: (finite(r, "complexity", 1e9),
                                         finite(r, "physical_penalty", 1.0),
                                         -finite(r, "margin_spearman", 0.0),
                                         -finite(r, "auc", 0.5)))[:max(20, limit)]
    champions, seen = [], set()
    for k, role in (("auc", "primary"), ("margin_spearman", "primary"),
                    ("final_recommendation_score", "primary"), ("complexity", "primary_min"),
                    ("trajectory_auc", "audit_only"), ("transition_score", "audit_only"),
                    ("star_eta2", "audit_only"), ("bacc", "audit_only"),
                    ("plateau", "audit_only")):
        r = (min(pool, key=lambda x: finite(x, k, 1e99)) if k == "complexity"
             else max(pool, key=lambda x: finite(x, k, -1e99)))
        if str(r["formula_id"]) not in seen:
            champions.append({**r, "champion_objective": k, "champion_role": role})
            seen.add(str(r["formula_id"]))
    chosen: dict[str, dict[str, Any]] = {}

    def add(seq: Sequence[dict[str, Any]], n: int, reason: str) -> None:
        for r0 in list(seq)[:n]:
            r = dict(r0)
            fid = str(r["formula_id"])
            prev = chosen.get(fid)
            if prev is None:
                r["finalist_reason"] = reason
                chosen[fid] = r
            elif reason not in str(prev.get("finalist_reason", "")):
                prev["finalist_reason"] = str(prev.get("finalist_reason", "")) + ";" + reason
    add(sorted(eligible, key=lambda r: (finite(r, "final_recommendation_score", -1e99),
                                        finite(r, "auc", 0.5)), reverse=True),
        max(12, limit // 2), "independent_edge_recommendation")
    add(_top(pool, "auc"), 5, "auc_champion")
    add(_top(pool, "margin_spearman"), 4, "margin_champion")
    add(simple, 10, "simple_near_optimal")
    for fam in FAMILIES:
        add(_top([r for r in eligible if r.get("family") == fam], "final_recommendation_score"),
            3, f"{fam.lower()}_eligible_leader")
    add(_top(pool, "trajectory_auc"), 2, "trajectory_audit_champion")
    if len(chosen) < limit:
        add(_top(pool, "final_recommendation_score"), limit - len(chosen), "fill")

    def order(r: Mapping[str, Any]) -> tuple[Any, ...]:
        return (bool(r.get("recommendation_eligible", False)),
                finite(r, "final_recommendation_score", -1e99), finite(r, "auc", 0.5),
                -finite(r, "complexity", 100))
    champion_ids = {str(r["formula_id"]) for r in champions}
    mandatory = [r for fid, r in chosen.items() if fid in champion_ids]
    others = sorted([r for fid, r in chosen.items() if fid not in champion_ids], key=order,
                    reverse=True)
    kept = others[:max(0, limit - len(mandatory[:limit]))] + mandatory[:limit]
    return sorted(kept, key=order, reverse=True), champions, simple


def feedback_weights(archive: Sequence[Mapping[str, Any]],
                     components: Sequence[Mapping[str, Any]], nfeat: int) -> np.ndarray:
    """1 + 2.5 x (rank-weighted use of each descriptor among the 300 best formulas)."""
    w = np.ones(nfeat, float)
    if not archive:
        return w
    counts = np.zeros(nfeat, float)
    for rank, r in enumerate(archive[:300]):
        try:
            a = json.loads(r["ast_json"])
        except Exception:
            continue
        gain = 1.0 / (1.0 + 0.02 * rank)
        for i in set(feature_ids(a, components)):
            counts[i] += gain
    if counts.max() > 0:
        w *= 1.0 + 2.5 * counts / counts.max()
    return w


def formula_table(records: Sequence[Mapping[str, Any]], components: Sequence[Mapping[str, Any]],
                  names: Sequence[str], edge: str) -> pd.DataFrame:
    rows = []
    for rank, r in enumerate(records, 1):
        a = json.loads(r["ast_json"])
        fids = sorted(set(feature_ids(a, components)))
        rows.append({**r, "rank": rank, "edge": edge,
                     "formula": formula_text(a, names, components, True),
                     "component_level_formula": formula_text(a, names, components, False),
                     "raw_features": ";".join(names[i] for i in fids),
                     "raw_feature_count": len(fids),
                     "feature_families": ";".join(sorted(set(feature_family(names[i])
                                                             for i in fids))),
                     "component_ids": ";".join(map(str, sorted(set(component_ids(a)))))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- search


def save_library(outdir: Path, components: list[dict[str, Any]], values: np.ndarray,
                 names: Sequence[str]) -> None:
    (outdir / "components.json").write_text(
        json.dumps(components, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    np.save(outdir / "component_values.float32.npy", np.asarray(values, dtype=np.float32))
    component_table(components, names).to_csv(outdir / "components.csv", index=False)


def search(data: EdgeData, settings: EdgeSettings, outdir: Path, components: list[dict[str, Any]],
           values: np.ndarray, resume: bool, stop_after: int | None) -> dict[str, Any]:
    """Coevolve the V and C skeleton populations over the shared component library."""
    seed = settings.seed
    rng = {"V": np.random.default_rng(seed + 101), "C": np.random.default_rng(seed + 1000103)}
    rng_components = np.random.default_rng(seed + 2000203)
    archive: dict[str, list] = {"V": [], "C": []}
    hof: dict[str, list] = {"V": [], "C": []}
    history: dict[str, list] = {"V": [], "C": []}
    best: dict[str, Any] = {"V": None, "C": None}
    feedback: list[dict[str, Any]] = []
    stale, start = 0, 1
    if resume:
        st = pickle.loads((outdir / CHECKPOINT).read_bytes())
        components = json.loads((outdir / "components.json").read_text(encoding="utf-8"))
        values = np.load(outdir / "component_values.float32.npy", allow_pickle=False)
        pop = {e: st["population"][e] for e in EDGES}
        archive, hof, history, best = st["archive"], st["hof"], st["history"], st["best"]
        feedback, stale, start = st["feedback"], int(st["stale"]), int(st["generation"]) + 1
        for e in EDGES:
            rng[e].bit_generator.state = st["rng"][e]
        rng_components.bit_generator.state = st["rng_components"]
        log(f"resuming after generation {start - 1} with {len(components)} components")
    pi_s_id = next((i for i, a in enumerate(components) if a.get("op") == "rc8"), None)
    if not resume:
        pop = {e: initial_population(rng[e], len(components), settings, pi_s_id, components)
               for e in EDGES}
    names = data.names
    evaluator = Evaluator(values, data.X, data.targets, components, settings.jobs,
                          settings.task_size)
    try:
        for gen in range(start, settings.max_generations + 1):
            rec: dict[str, list] = {}
            front: dict[str, list] = {}
            current: dict[str, dict] = {}
            up = {}
            for e in EDGES:
                t0 = time.time()
                rec[e] = evaluator.evaluate(e, pop[e])
                front[e] = pareto_front(rec[e] + archive[e], settings.pareto_preselect,
                                        settings.pareto_size)
                seconds = time.time() - t0
                hof[e] = hall_of_fame(hof[e], rec[e] + front[e], gen,
                                      max(2400, settings.pareto_size))
                current[e] = best_values(front[e])
                up[e], best[e] = improved(best_values(hof[e]), best[e])
                families = pd.Series([r["family"] for r in front[e]]).value_counts().to_dict()
                history[e].append({
                    "generation": gen, **{f"current_{k}": v for k, v in current[e].items()},
                    **{f"global_best_{k}": v for k, v in best_values(hof[e]).items()},
                    "pareto_n": len(front[e]), "historical_elite_n": len(hof[e]),
                    "safe_n": sum(bool(r.get("safe")) for r in rec[e]),
                    "free_pareto": int(families.get("FREE", 0)),
                    "rooted_pareto": int(families.get("PI_S_ROOTED", 0)),
                    "joint_stale": None, "seconds": seconds})
            stale = 0 if (up["V"] or up["C"]) else stale + 1
            for e in EDGES:
                history[e][-1]["joint_stale"] = stale
                archive[e] = front[e]
                pd.DataFrame(history[e]).to_csv(outdir / f"history_{EDGE_NAMES[e]}.csv",
                                                index=False)
            log(f"generation {gen}: V auc {current['V']['auc']:.4f} (best "
                f"{best_values(hof['V'])['auc']:.4f}), C auc {current['C']['auc']:.4f} (best "
                f"{best_values(hof['C'])['auc']:.4f}), stale {stale}")
            if gen % settings.feedback_interval == 0 and gen < settings.max_generations:
                # Grow the library from descriptors used by good formulas of both edges.
                fb = np.sqrt(feedback_weights(hof["V"], components, len(names))
                             * feedback_weights(hof["C"], components, len(names)))
                weights = data.prior * np.maximum(fb, 1e-12)
                weights /= weights.sum()
                before = len(components)
                components, values = grow_components(data.X, settings, rng_components, weights,
                                                     data.pi_s, components, values)
                feedback.append({"generation": gen, "components_before": before,
                                 "components_after": len(components),
                                 "components_added": len(components) - before,
                                 "V_pareto_n": len(archive["V"]), "C_pareto_n": len(archive["C"])})
                pd.DataFrame(feedback).to_csv(outdir / "feedback_history.csv", index=False)
                save_library(outdir, components, values, names)
                evaluator.close()
                evaluator = Evaluator(values, data.X, data.targets, components, settings.jobs,
                                      settings.task_size)
                log(f"component feedback: {len(components) - before} added, {len(components)} total")
            stop = ((gen >= settings.min_generations and stale >= settings.patience)
                    or (gen >= settings.nominal_generations and stale >= max(8, settings.patience // 2))
                    or gen >= settings.max_generations)
            # The next populations are drawn before the checkpoint, so a resumed
            # run continues with exactly the same random streams.
            if not stop:
                pop = {e: next_population(rng[e], rec[e] + archive[e] + hof[e], components,
                                          len(components), settings, pi_s_id) for e in EDGES}
            write_pickle(outdir / CHECKPOINT, {
                "settings": settings.scientific(), "generation": gen, "population": pop,
                "archive": archive, "hof": hof, "history": history, "best": best,
                "feedback": feedback, "stale": stale,
                "rng": {e: rng[e].bit_generator.state for e in EDGES},
                "rng_components": rng_components.bit_generator.state})
            if stop_after is not None and gen >= stop_after:
                raise StopRequested(f"stopped after generation {gen}")
            if stop:
                if gen < settings.max_generations:
                    log(f"stopping after generation {gen} (stale {stale})")
                break
    finally:
        evaluator.close()

    top = {}
    for e in EDGES:
        name = EDGE_NAMES[e]
        final, champions, simple = finalists(hof[e], archive[e], settings.top_n)
        top[e] = formula_table(final, components, names, e)
        top[e]["final_rank"] = np.arange(1, len(top[e]) + 1)
        top[e].to_csv(outdir / f"top30_{name}.csv", index=False)
        formula_table(champions, components, names, e).to_csv(outdir / f"champions_{name}.csv",
                                                               index=False)
        formula_table(simple, components, names, e).to_csv(
            outdir / f"simple_near_optimal_{name}.csv", index=False)
        formula_table(hof[e], components, names, e).to_csv(outdir / f"elite_{name}.csv",
                                                           index=False)
        formula_table(archive[e], components, names, e).to_csv(outdir / f"pareto_{name}.csv",
                                                               index=False)
    return {"top": top, "components": components, "values": values,
            "best": {EDGE_NAMES[e]: best_values(archive[e]) for e in EDGES}}


# --------------------------------------------------------------- pair audit


def score_of(record: Mapping[str, Any], X: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.asarray(evaluate(json.loads(record["ast_json"]), X, values, None), dtype=float)


def predict(score: np.ndarray, record: Mapping[str, Any]) -> np.ndarray:
    th = float(record["threshold"])
    return ((score >= th) if int(record["orientation"]) == 1 else (score <= th)).astype(int)


def oriented_coordinate(score: np.ndarray, orientation: int) -> np.ndarray:
    """(oriented score - median) / IQR."""
    z = np.asarray(score, float) * int(orientation)
    iqr = float(np.quantile(z, 0.75) - np.quantile(z, 0.25))
    if not math.isfinite(iqr) or iqr < 1e-12:
        iqr = float(np.std(z) or 1.0)
    return (z - float(np.median(z))) / iqr


def coarse_state(pv: np.ndarray, pc: np.ndarray) -> np.ndarray:
    return np.where((pv == 1) & (pc == 1), "GG", np.where((pv == 1) & (pc == 0), "GN",
                    np.where((pv == 0) & (pc == 1), "NG", "NN")))


def pair_audit(top: Mapping[str, pd.DataFrame], data: EdgeData, values: np.ndarray
               ) -> pd.DataFrame:
    """Joint edge states of every pairing of the first 60 V and C formulas.

    Diagnostic only: the V and C formulas are selected independently.  For
    NN states the distance between the two oriented coordinates is read as
    "same non-Γ valley" versus "different valleys".
    """
    truth = data.states
    coarse_truth = np.where(np.isin(truth, ["NN_same", "NN_diff"]), "NN", truth)
    nn = np.isin(truth, ["NN_same", "NN_diff"])
    nn_y = (truth[nn] == "NN_same").astype(np.int8)
    rows_v, rows_c = top["V"].head(60).to_dict("records"), top["C"].head(60).to_dict("records")
    sv_cache = {r["formula_id"]: score_of(r, data.X, values) for r in rows_v}
    sc_cache = {r["formula_id"]: score_of(r, data.X, values) for r in rows_c}
    rows = []
    for vr in rows_v:
        sv = sv_cache[vr["formula_id"]]
        pv, zv = predict(sv, vr), oriented_coordinate(sv, int(vr["orientation"]))
        for cr in rows_c:
            sc = sc_cache[cr["formula_id"]]
            pc, zc = predict(sc, cr), oriented_coordinate(sc, int(cr["orientation"]))
            pred = coarse_state(pv, pc)
            if nn.sum() >= 4 and len(np.unique(nn_y)) == 2:
                d = -np.abs(zv[nn] - zc[nn])
                raw = rank_auc(nn_y, d)
                nn_auc, nn_orient = max(raw, 1 - raw), 1 if raw >= 0.5 else -1
                nn_th, nn_bacc, _ = best_threshold(nn_y, d, nn_orient)
                nn_pred = (d >= nn_th).astype(int) if nn_orient == 1 else (d <= nn_th).astype(int)
                exact = pred.copy()
                exact[nn] = np.where(nn_pred == 1, "NN_same", "NN_diff")
                exact5 = float(np.mean(exact == truth))
            else:
                nn_auc = nn_bacc = exact5 = float("nan")
            rows.append({
                "V_formula_id": vr["formula_id"], "C_formula_id": cr["formula_id"],
                "V_rank": int(vr["rank"]), "C_rank": int(cr["rank"]),
                "V_auc": float(vr["auc"]), "C_auc": float(cr["auc"]),
                "V_transition": float(vr["transition_score"]),
                "C_transition": float(cr["transition_score"]),
                "coarse_edge_state_accuracy_GG_GN_NG_NN": float(np.mean(pred == coarse_truth)),
                "NN_identity_auc_same_vs_diff": nn_auc, "NN_identity_bacc_discovery": nn_bacc,
                "exact_5state_accuracy_with_NN_identity_readout": exact5,
                "pair_complexity": float(vr["complexity"]) + float(cr["complexity"]),
                **{f"{s}_truth_n": int(np.sum(truth == s))
                   for s in ("NN_same", "NN_diff", "GG", "GN", "NG")}})
    df = pd.DataFrame(rows)
    nnterm = df["NN_identity_auc_same_vs_diff"].fillna(0.5)
    vgate = max(0.70, float(df["V_auc"].max()) - 0.020)
    cgate = max(0.70, float(df["C_auc"].max()) - 0.020)
    df["pair_edge_quality_eligible"] = (df["V_auc"] >= vgate) & (df["C_auc"] >= cgate)
    df["V_auc_gate"], df["C_auc_gate"] = vgate, cgate
    df["pair_audit_score"] = (0.40 * df["coarse_edge_state_accuracy_GG_GN_NG_NN"]
                              + 0.20 * (df["V_auc"] + df["C_auc"]) / 2
                              + 0.18 * (df["V_transition"] + df["C_transition"]) / 2
                              + 0.10 * nnterm + 0.12 / (1 + 0.03 * df["pair_complexity"]))
    rec_v, rec_c = str(top["V"].iloc[0]["formula_id"]), str(top["C"].iloc[0]["formula_id"])
    df["is_independently_recommended_formula_pair"] = (
        (df["V_formula_id"].astype(str) == rec_v) & (df["C_formula_id"].astype(str) == rec_c))
    df = df.sort_values(["pair_edge_quality_eligible", "pair_audit_score",
                         "coarse_edge_state_accuracy_GG_GN_NG_NN",
                         "NN_identity_auc_same_vs_diff", "pair_complexity"],
                        ascending=[False, False, False, False, True],
                        kind="mergesort").reset_index(drop=True)
    df.insert(0, "pair_rank", np.arange(1, len(df) + 1))
    return df


def recommended_exports(top: Mapping[str, pd.DataFrame], pairs: pd.DataFrame, data: EdgeData,
                        values: np.ndarray, outdir: Path) -> None:
    """Per-sample scores and path crossings of the two recommended formulas."""
    vr, cr = top["V"].iloc[0], top["C"].iloc[0]
    V, C = data.targets["V"], data.targets["C"]
    sv, sc = score_of(vr, data.X, values), score_of(cr, data.X, values)
    pv, pc = predict(sv, vr), predict(sc, cr)
    zv = oriented_coordinate(sv, int(vr["orientation"]))
    zc = oriented_coordinate(sc, int(cr["orientation"]))
    tab = pd.DataFrame({
        "sample_id": V.sample_ids, "trajectory_id": V.trajectories, "k_index": V.k_index,
        "Pi_V_raw": sv, "Pi_V_oriented_robust": zv, "tau_V_raw": float(vr["threshold"]),
        "V_pred_Gamma": pv, "V_true_Gamma": V.y,
        "Pi_C_raw": sc, "Pi_C_oriented_robust": zc, "tau_C_raw": float(cr["threshold"]),
        "C_pred_Gamma": pc, "C_true_Gamma": C.y,
        "predicted_coarse_state": coarse_state(pv, pc), "true_exact_state": data.states,
        "V_exact_k": V.exact_key, "C_exact_k": C.exact_key, "V_star": V.star_key,
        "C_star": C.star_key, "V_signed_margin_eV": V.margin, "C_signed_margin_eV": C.margin})
    tab["NN_coordinate_distance_abs"] = np.abs(zv - zc)
    tab.to_csv(outdir / "recommended_pair_scores.csv", index=False)
    rows = []
    for t, grp in tab.groupby("trajectory_id", sort=True):
        grp = grp.sort_values("k_index")

        def changes(a: Any) -> str:
            a = np.asarray(a)
            ii = np.flatnonzero(a[1:] != a[:-1])
            return ";".join(f"{int(grp.k_index.iloc[i])}->{int(grp.k_index.iloc[i + 1])}"
                            for i in ii)
        rows.append({"trajectory_id": t, "n": len(grp), "k_min": int(grp.k_index.min()),
                     "k_max": int(grp.k_index.max()),
                     "V_true_crossings": changes(grp.V_true_Gamma),
                     "V_pred_crossings": changes(grp.V_pred_Gamma),
                     "C_true_crossings": changes(grp.C_true_Gamma),
                     "C_pred_crossings": changes(grp.C_pred_Gamma),
                     "true_exact_state_sequence": ";".join(grp.true_exact_state.astype(str)),
                     "predicted_coarse_state_sequence":
                         ";".join(grp.predicted_coarse_state.astype(str))})
    pd.DataFrame(rows).to_csv(outdir / "recommended_pair_trajectories.csv", index=False)
    hit = pairs[pairs["is_independently_recommended_formula_pair"]]
    write_json(outdir / "recommended_formulas.json",
               {"V": vr.to_dict(), "C": cr.to_dict(),
                "pair_audit": (hit.iloc[0] if len(hit) else pairs.iloc[0]).to_dict()})


# ---------------------------------------------------------------------- run


def software_versions() -> dict[str, str]:
    return {"package": f"symbolic_regression {__version__}",
            "source_sha256": sha256_file(Path(__file__)), "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()}


def run(settings: EdgeSettings, inputs: Mapping[str, Path | None], outdir: Path, *,
        resume: bool = False, overwrite: bool = False,
        stop_after_generation: int | None = None) -> dict[str, Any]:
    """Run or resume a two-edge search; returns the run record."""
    outdir = Path(outdir).resolve()
    if resume:
        state = pickle.loads((outdir / CHECKPOINT).read_bytes())
        if state["settings"] != settings.scientific():
            changed = sorted(k for k, v in settings.scientific().items()
                             if state["settings"].get(k) != v)
            raise ValueError(f"settings differ from the checkpoint: {', '.join(changed)}")
        record = json.loads((outdir / RUN_RECORD).read_text(encoding="utf-8"))
        record.setdefault("resumed", []).append(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    else:
        if outdir.exists() and any(outdir.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{outdir} is not empty; use --overwrite or --resume")
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        record = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "settings": asdict(settings)}
    data = load_edge_data(inputs["protocol"], inputs["samples"], inputs["kpoints"], settings,
                          pi_s_formula=inputs.get("pi_s_formula"),
                          pi_s_scales=inputs.get("pi_s_scales"),
                          prior_valence=inputs.get("prior_valence"),
                          prior_conduction=inputs.get("prior_conduction"))
    record.update({"inputs": {k: str(v) for k, v in inputs.items() if v is not None},
                   "input_sha256": data.checksums, "command": sys.argv,
                   "software": software_versions(), "status": "running",
                   "discovery_rows": int(len(data.sample_ids)), "descriptors": len(data.names),
                   "targets": {EDGE_NAMES[e]: target_summary(t) for e, t in data.targets.items()},
                   "edge_states": pd.Series(data.states).value_counts().to_dict()})
    write_json(outdir / RUN_RECORD, record)
    log(f"{len(data.sample_ids)} discovery rows, {len(data.names)} descriptors; "
        + ", ".join(f"{EDGE_NAMES[e]}: {s['gamma']} at Gamma / {s['non_gamma']} not"
                    for e, s in ((e, target_summary(t)) for e, t in data.targets.items())))
    if not resume:
        np.save(outdir / "discovery_scales.npy", data.scales)
        pd.DataFrame({"feature": data.names, "source_feature_index": data.source_index,
                      "family": [feature_family(n) for n in data.names]}).to_csv(
            outdir / "discovery_features.csv", index=False)
        target_table(data.targets["V"], data.targets["C"], data.states).to_csv(
            outdir / "edge_targets.csv", index=False)
        if data.pi_s is not None:
            baseline = [{**candidate_metrics(data.pi_s, data.targets[e], complexity=0.0),
                         "edge": e, "formula": "Pi_S"} for e in EDGES]
            pd.DataFrame(baseline).to_csv(outdir / "pi_s_baseline.csv", index=False)
        components, values = grow_components(data.X, settings, np.random.default_rng(settings.seed),
                                              data.prior, data.pi_s)
        save_library(outdir, components, values, data.names)
    else:
        components, values = [], np.zeros((0, 0), np.float32)
    try:
        result = search(data, settings, outdir, components, values, resume,
                        stop_after_generation)
    except StopRequested as stop:
        record.update({"status": "stopped", "message": str(stop),
                       "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        write_json(outdir / RUN_RECORD, record)
        log(str(stop))
        return record
    pairs = pair_audit(result["top"], data, result["values"])
    pairs.to_csv(outdir / "pair_audit.csv", index=False)
    ids = sorted({cid for df in result["top"].values() for raw in df["ast_json"]
                  for cid in component_ids(json.loads(raw))})
    table = component_table(result["components"], data.names)
    table[table["component_id"].isin(ids)].sort_values("component_id").to_csv(
        outdir / "finalist_components.csv", index=False)
    recommended_exports(result["top"], pairs, data, result["values"], outdir)
    record.update({"status": "finalized", "best": result["best"],
                   "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    write_json(outdir / RUN_RECORD, record)
    log("finished")
    return record


# ------------------------------------------------------------------- replay


REPLAYED_FIELDS = ("formula_id", "auc", "raw_auc", "orientation", "threshold", "bacc", "plateau",
                   "margin_spearman", "trajectory_auc", "trajectory_auc_n", "transition_score",
                   "star_eta2", "star_group_n", "complexity", "q99_q50", "physical_penalty",
                   "skeleton_complexity", "expanded_operator_nodes", "expanded_leaf_count",
                   "expanded_ratio_nodes", "expanded_depth", "selection_score",
                   "tp", "tn", "fp", "fn", "acc")


def resolve_names(node: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    """Replace ``{"op": "feature", "name": ...}`` leaves by descriptor indices."""
    out = dict(node)
    if out["op"] == "feature" and "name" in out:
        return {"op": "feature", "id": list(names).index(out["name"])}
    for key in ("arg", "left", "right"):
        if key in out:
            out[key] = resolve_names(out[key], names)
    return out


def load_edge_formulas(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Formula records and their evaluation conventions.

    Accepts a top list (CSV), ``recommended_formulas.json`` or a coordinate
    file ``{"pi_s_fit_on_samples": ..., "coordinates": [...]}`` whose records
    hold descriptor-level trees (``ast``) and the stored metrics (``stored``).
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, float_precision="round_trip").to_dict("records"), {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and "coordinates" in value:
        return list(value["coordinates"]), {k: v for k, v in value.items() if k != "coordinates"}
    if isinstance(value, dict) and {"V", "C"} <= set(value):
        return [value["V"], value["C"]], {}
    return ([value] if isinstance(value, dict) else list(value)), {}


def replay(records: Sequence[Mapping[str, Any]], data: EdgeData,
           components: Sequence[Mapping[str, Any]] | None = None) -> tuple[list[dict[str, Any]],
                                                                          pd.DataFrame]:
    """Re-evaluate saved formulas and compare every stored metric.

    Search formulas are skeletons over components; their components are
    rebuilt from ``components`` and rounded to single precision, as in the
    search.  Coordinate records (descriptor leaves) may set their own
    ``ratio_epsilon`` and ``spearman_ranks: "average"`` (tie-averaged ranks
    in the margin agreement instead of the search's ordinal ranks).
    """
    cache: dict[int, np.ndarray] = {}

    def component_values(tree: Mapping[str, Any]) -> np.ndarray | None:
        ids = sorted(set(component_ids(tree)))
        if not ids:
            return None
        width = max(ids) + 1
        M = np.zeros((width, len(data.sample_ids)), dtype=np.float32)
        for cid in ids:
            if cid not in cache:
                cache[cid] = np.asarray(evaluate(components[cid], data.X, None, data.pi_s),
                                        dtype=np.float32)
            M[cid] = cache[cid]
        return M

    reports, scores = [], []
    for position, record in enumerate(records, 1):
        edge = str(record["edge"])
        tree = record.get("ast")
        tree = json.loads(record["ast_json"]) if tree is None else resolve_names(tree, data.names)
        values = component_values(tree)
        ratio_eps = float(record.get("ratio_epsilon", RATIO_EPSILON))
        s = np.asarray(evaluate(tree, data.X, values, data.pi_s, ratio_eps), dtype=float)
        comp, est = expanded_complexity(tree, components or [])
        target = data.targets[edge]
        m = candidate_metrics(s, target, comp)
        if m.get("safe") and record.get("spearman_ranks") == "average":
            # The published coordinates were scored with tie-averaged ranks.
            mm = target.has_margin & target.available
            ra = pd.Series(s[mm] * m["orientation"]).rank(method="average").to_numpy()
            rb = pd.Series(target.margin[mm]).rank(method="average").to_numpy()
            m["margin_spearman"] = float(np.corrcoef(ra, rb)[0, 1])
        m.update({"formula_id": tree_hash(tree),
                  "skeleton_complexity": float(tree_ops(tree) + 0.5 * tree_leaves(tree) + 0.7 * sum(
                      1 for p in node_paths(tree) if g.node_at(tree, p).get("op") == "ratio")),
                  "expanded_operator_nodes": int(est["ops"]),
                  "expanded_leaf_count": int(est["leaves"]),
                  "expanded_ratio_nodes": int(est["ratios"]), "expanded_depth": int(est["depth"])})
        m["selection_score"] = selection_score(m)
        if m.get("safe"):
            y = target.y[target.available]
            p = predict(s[target.available], m).astype(bool)
            m.update(tp=int(np.sum(p & (y == 1))), tn=int(np.sum(~p & (y == 0))),
                     fp=int(np.sum(p & (y == 0))), fn=int(np.sum(~p & (y == 1))),
                     acc=float(np.mean(p == (y == 1))))
        stored = record.get("stored", record)
        checks = {}
        for f in REPLAYED_FIELDS:
            if f not in stored or f not in m:
                continue
            a, b = stored[f], m[f]
            if isinstance(a, float) and math.isnan(a):
                same = isinstance(b, float) and math.isnan(b)
            elif isinstance(a, str) or isinstance(b, str):
                same = str(a) == str(b)
            else:
                same = float(a) == float(b)
            checks[f] = {"stored": a, "replayed": b, "equal": bool(same)}
        reports.append({"rank": record.get("rank", position), "edge": edge,
                        "formula_id": record.get("formula_id", m["formula_id"]),
                        "all_equal": all(c["equal"] for c in checks.values()), "checks": checks})
        scores.extend({"rank": record.get("rank", position), "edge": edge,
                       "formula_id": m["formula_id"], "sample_id": data.sample_ids[i],
                       "score": float(s[i])} for i in range(len(s)))
    return reports, pd.DataFrame(scores)


# ------------------------------------------------- four states (perovskites)

STATES = {(1, 1): "COLOCATED", (0, 1): "V_MISMATCH_CARRIER", (1, 0): "C_MISMATCH_CARRIER",
          (0, 0): "DISTRIBUTED_MISMATCH"}


def four_states(valence: Path, conduction: Path, rank: int = 1
                ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Combine out-of-fold predictions of a valence and a conduction relation.

    Each input lists, per sample, the truth (1 = extremum at the momentum of
    the lowest direct transition) and the out-of-fold prediction of one edge.
    The pair of signs gives one of four states; a state is correct when both
    edges are.
    """
    def side(path: Path) -> pd.DataFrame:
        t = pd.read_csv(path, float_precision="round_trip")
        return t[t["rank"] == rank].reset_index(drop=True)
    v, c = side(valence), side(conduction)
    if v.sample_id.tolist() != c.sample_id.tolist():
        raise ValueError("the two prediction tables list different samples")
    table = pd.DataFrame({
        "sample_id": v.sample_id, "curve_id": v.curve_id, "fold_id": v.fold_id,
        "A_V": v.truth, "pred_V": v.oof_prediction, "margin_V": v.oof_margin, "score_V": v.score,
        "A_C": c.truth, "pred_C": c.oof_prediction, "margin_C": c.oof_margin, "score_C": c.score})
    table["state"] = [STATES[(int(a), int(b))] for a, b in zip(table.A_V, table.A_C)]
    table["state_pred"] = [STATES[(int(a), int(b))] for a, b in zip(table.pred_V, table.pred_C)]

    def bacc(y: pd.Series, p: pd.Series) -> float:
        return 0.5 * (float(np.mean(p[y == 1] == 1)) + float(np.mean(p[y == 0] == 0)))
    both = int(((table.A_V == table.pred_V) & (table.A_C == table.pred_C)).sum())
    return table, {"rows": len(table), "valence_oof_bacc": bacc(table.A_V, table.pred_V),
                   "conduction_oof_bacc": bacc(table.A_C, table.pred_C),
                   "both_edges_correct": both, "four_state_accuracy": both / len(table),
                   "state_counts": table.state.value_counts().to_dict(),
                   "valence_formula_id": str(v.formula_id.iloc[0]),
                   "conduction_formula_id": str(c.formula_id.iloc[0])}
