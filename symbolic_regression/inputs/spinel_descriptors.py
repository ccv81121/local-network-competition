"""Spinel descriptors AB₂X₄ (B = B/B' mixture) computed from the structures.

``descriptor_table(structures, elements)`` gives one row per structure of the
1003-structure database (``data/descriptor_tables/structures/spinel_structures.csv``)
with the 214 descriptors of ``data/descriptor_tables/spinel`` and ``spinel_edges``,
in their order. Each structure is the embedded POSCAR (the 14-atom primitive cell
for the canonicalised path endpoints), expanded to a diagonal supercell of at
least 56 sites. The descriptors fall into these groups:

* cell: volume, lattice lengths and their anisotropy;
* elements: A-, B-mixture and X-site properties (Bmix = (1 − x) B + x B', with x
  the B' fraction counted on the B sites), B–X and A–X mismatches, B/B' alloy
  variances and drifts;
* BX₆ octahedra: B–X bond statistics, distortion, shape and crystal-field
  anisotropy of the bond directions, X–B–X angles, relative p–d hopping
  t = [(r_B + r_X)/d]^3.5 / (1 + |χ_B − χ_X|) (covalent radii, Pauling χ);
* B–X–B network: path counts, angles (edge ≈ 90°, corner ≈ 180°), lengths and
  effective B–B hopping √(t₁t₂);
* B–B and X–X sublattices: first- and second-shell distances and graph
  statistics on the first-shell graph (cutoff 1.03 × median sixth-neighbour
  distance), relative X–X hopping (2 r_X / d)³;
* B/B' configuration: entropies, local fractions, Warren–Cowley parameters,
  B' clusters;
* A–X cage: distances to the 12 nearest X, off-centring;
* combinations of the above (on-site alignment, d filling, compatibilities).

Hopping and crystal-field quantities are dimensionless relative proxies.
"""
from __future__ import annotations

import io
import itertools
import math
from collections import Counter, deque
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .poscar import SYMBOL, Structure, decode, parse_poscar

EPS = 1.0e-12
ID_COLUMN = "Task Directory"
STRUCTURE_COLUMNS = (ID_COLUMN, "A_site", "B_original", "substitute", "X_site",
                     "endpoint_canonicalized", "spinel_poscar_gz_b64", "primitive_poscar_gz_b64")
SHIFTS_27 = np.array(list(itertools.product((-1.0, 0.0, 1.0), repeat=3)), dtype=float)

# Element properties: column of elements_primary.csv / elements_extended.csv -> name.
PRIMARY_COLUMNS = {
    "number": "Z", "Period": "period", "Group": "group", "covalent_radius": "covalent_radius_pm",
    "number_of_valence_electrons": "valence_electrons", "FirstIonization": "first_ionization_kJmol",
    "dipole_polarizability": "polarizability_A3", "Electronegativity": "pauling_EN",
}
EXTENDED_COLUMNS = {
    "Atomic number": "Z", "Period number": "period", "Group number": "group",
    "Mendeleev number": "mendeleev", "Covalent radius (Å)": "covalent_radius_A",
    "Ionic radius (Å)": "ionic_radius_A", "Pauling EN": "pauling_EN",
    "Number of valence electrons": "valence_electrons", "Number of d electrons": "d_electrons",
    "First ionization energy (kJ/mol)": "first_ionization_kJmol",
    "Polarizability": "polarizability_A3",
}
SITE_PROPERTIES = ("Z", "period", "group", "mendeleev", "covalent_radius_A", "ionic_radius_A",
                   "pauling_EN", "valence_electrons", "d_electrons", "first_ionization_kJmol",
                   "polarizability_A3")
MISMATCH_PROPERTIES = ("Z", "period", "covalent_radius_A", "ionic_radius_A", "pauling_EN",
                       "valence_electrons", "d_electrons", "first_ionization_kJmol",
                       "polarizability_A3")
AX_MISMATCH_PROPERTIES = {"Z", "covalent_radius_A", "ionic_radius_A", "pauling_EN",
                          "first_ionization_kJmol", "polarizability_A3"}
ALLOY_PROPERTIES = ("pauling_EN", "ionic_radius_A", "d_electrons", "valence_electrons",
                    "first_ionization_kJmol", "polarizability_A3", "Z", "period")
# Not in the tables: equal to B_valence_filling_mean, to B_d_filling_mean, constant.
OMITTED_SITE_PROPERTIES = {("Bmix", "valence_electrons"), ("Bmix", "d_electrons"),
                           ("X", "d_electrons")}


# ---------------------------------------------------------------- statistics

def _finite(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    return a[np.isfinite(a)]


def _mean(values: Sequence[float]) -> float:
    a = _finite(values)
    return float(np.mean(a)) if a.size else np.nan


def _std(values: Sequence[float]) -> float:
    a = _finite(values)
    return float(np.std(a)) if a.size else np.nan


def _cv(values: Sequence[float]) -> float:
    m, s = _mean(values), _std(values)
    return float(s / max(abs(m), EPS)) if np.isfinite(m) and np.isfinite(s) else np.nan


def _skew(values: Sequence[float]) -> float:
    a = _finite(values)
    if a.size < 3:
        return 0.0
    m, s = float(np.mean(a)), float(np.std(a))
    return 0.0 if s <= EPS else float(np.mean(((a - m) / s) ** 3))


def _stats(values: Sequence[float], prefix: str, quantiles: bool = False) -> dict[str, float]:
    a = _finite(values)
    out = {f"{prefix}_mean": float(np.mean(a)), f"{prefix}_std": float(np.std(a)),
           f"{prefix}_cv": _cv(a), f"{prefix}_min": float(np.min(a)),
           f"{prefix}_max": float(np.max(a))}
    if quantiles:
        out.update({f"{prefix}_q10": float(np.quantile(a, 0.10)),
                    f"{prefix}_q90": float(np.quantile(a, 0.90)), f"{prefix}_skew": _skew(a)})
    return out


def _binary_entropy(x: float) -> float:
    if not np.isfinite(x) or x <= EPS or x >= 1.0 - EPS:
        return 0.0
    return float(-(x * math.log(x) + (1.0 - x) * math.log(1.0 - x)) / math.log(2.0))


def _abs_difference(a: float, b: float) -> float:
    """2|a − b| / (|a| + |b|)."""
    return float(2.0 * abs(a - b) / max(abs(a) + abs(b), EPS))


def _signed_difference(a: float, b: float) -> float:
    """2(a − b) / (|a| + |b|)."""
    return float(2.0 * (a - b) / max(abs(a) + abs(b), EPS))


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 <= EPS or n2 <= EPS:
        return np.nan
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))


# ---------------------------------------------------------------- inputs

def _symbol(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return text if SYMBOL.match(text) else ""


def _number(value) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def load_elements(primary: Path, extended: Path) -> pd.DataFrame:
    """Element properties indexed by symbol; the extended table overrides the primary one."""
    records: dict[str, dict[str, float]] = {}
    for path, columns in ((primary, PRIMARY_COLUMNS), (extended, EXTENDED_COLUMNS)):
        table = pd.read_csv(path)
        table.columns = [str(c).strip() for c in table.columns]
        for _, row in table.iterrows():
            symbol = _symbol(row.get("symbol"))
            if not symbol:
                continue
            record = records.setdefault(symbol, {"symbol": symbol})
            for column, name in columns.items():
                if column not in row or pd.isna(row[column]):
                    continue
                value = _number(row[column])
                if columns is EXTENDED_COLUMNS:
                    record[name] = value
                elif name == "covalent_radius_pm":   # pm in the primary table
                    record["covalent_radius_A"] = value / 100.0 if value > 5 else value
                elif name not in record or not np.isfinite(_number(record[name])):
                    record[name] = value
    return pd.DataFrame(records.values()).set_index("symbol").sort_index()


def _element(elements: pd.DataFrame, symbol: str, prop: str) -> float:
    value = _number(elements.at[symbol, prop])
    if not np.isfinite(value):
        raise ValueError(f"missing element property {prop} for {symbol}")
    return value


def read_structures(path: Path) -> pd.DataFrame:
    """The columns of the structure table that the descriptors use."""
    return pd.read_csv(path, usecols=list(STRUCTURE_COLUMNS), low_memory=False)


def structure(row: pd.Series) -> Structure:
    """Embedded POSCAR of a row: the primitive cell for canonicalised endpoints."""
    endpoint = str(row["endpoint_canonicalized"]).strip().lower() == "true"
    return parse_poscar(decode(row["primitive_poscar_gz_b64" if endpoint
                                   else "spinel_poscar_gz_b64"]))


def supercell(cell: Structure, target_sites: int = 56) -> Structure:
    """Smallest diagonal supercell with at least ``target_sites`` sites (most isotropic first)."""
    if cell.nsites >= target_sites:
        return cell
    lengths = np.linalg.norm(cell.lattice, axis=1)
    candidates = []
    for reps in itertools.product(range(1, 5), repeat=3):
        n = cell.nsites * int(np.prod(reps))
        if n >= target_sites:
            score = (n - target_sites, _cv(lengths * np.asarray(reps, dtype=float)),
                     max(reps), sum(reps), reps)
            candidates.append((score, reps))
    reps = min(candidates, key=lambda item: item[0])[1]
    rep_array = np.asarray(reps, dtype=float)
    lattice = cell.lattice.copy()
    for axis in range(3):
        lattice[axis] *= reps[axis]
    frac, symbols = [], []
    for shift in itertools.product(range(reps[0]), range(reps[1]), range(reps[2])):
        for symbol, f in zip(cell.symbols, cell.frac):
            symbols.append(symbol)
            frac.append((f + np.array(shift, dtype=float)) / rep_array)
    return Structure(lattice, tuple(symbols), np.asarray(frac, dtype=float))


# ---------------------------------------------------------------- geometry

def minimum_image(cell: Structure) -> tuple[np.ndarray, np.ndarray]:
    """Pair distances (∞ on the diagonal) and vectors i → j over the 27 nearest images."""
    frac_delta = cell.frac[None, :, :] - cell.frac[:, None, :]
    candidates = frac_delta[:, :, None, :] + SHIFTS_27[None, None, :, :]
    cart = np.einsum("...k,kl->...l", candidates, cell.lattice, optimize=True)
    norm2 = np.einsum("...k,...k->...", cart, cart, optimize=True)
    choice = np.argmin(norm2, axis=2)
    ii, jj = np.arange(cell.nsites)[:, None], np.arange(cell.nsites)[None, :]
    vectors = cart[ii, jj, choice]
    distances = np.sqrt(norm2[ii, jj, choice])
    np.fill_diagonal(distances, np.inf)
    vectors[np.arange(cell.nsites), np.arange(cell.nsites)] = 0.0
    return distances, vectors


def _coordination_scores(site: int, x_sites: list[int], distances: np.ndarray,
                         vectors: np.ndarray) -> tuple[float, float]:
    """Octahedral and tetrahedral misfit of the X shell around a cation (lower fits better)."""
    nearest = sorted(x_sites, key=lambda j: distances[site, j])
    if len(nearest) < 6:
        return np.inf, np.inf
    six, four = nearest[:6], nearest[:4]
    d6 = np.array([distances[site, j] for j in six], dtype=float)
    d4 = d6[:4]
    angles6 = [_angle(vectors[site, a], vectors[site, b]) for a, b in itertools.combinations(six, 2)]
    angles4 = [_angle(vectors[site, a], vectors[site, b]) for a, b in itertools.combinations(four, 2)]
    dev6 = [min(abs(a - 90.0), abs(a - 180.0)) / 90.0 for a in angles6 if np.isfinite(a)]
    dev4 = [abs(a - 109.4712206) / 109.4712206 for a in angles4 if np.isfinite(a)]
    oct_gap = (distances[site, nearest[6]] - d6[-1]) / max(d6[-1], EPS) if len(nearest) > 6 else 0.0
    tet_gap = (d6[4] - d4[-1]) / max(d4[-1], EPS)
    return (_cv(d6) + _mean(dev6) - 0.20 * max(oct_gap, 0.0),
            _cv(d4) + _mean(dev4) - 0.20 * max(tet_gap, 0.0))


def sublattices(cell: Structure, A: str, B0: str, B1: str, X: str, distances: np.ndarray,
                vectors: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    """A, B and X site indices: species first, octahedral fit for species on both sublattices."""
    x_sites = [i for i, s in enumerate(cell.symbols) if s == X]
    cations = [i for i, s in enumerate(cell.symbols) if s != X]
    if not x_sites:
        raise ValueError(f"no {X} sites in the structure")
    n_b, n_a = int(round(len(x_sites) / 2.0)), int(round(len(x_sites) / 4.0))
    if n_b <= 0 or n_b + n_a != len(cations):
        n_b = int(round(2.0 * len(cations) / 3.0))
        n_a = len(cations) - n_b
    scores = {i: _coordination_scores(i, x_sites, distances, vectors) for i in cations}
    b_species, a_species = {s for s in (B0, B1) if s}, {A} if A else set()
    only_b = [i for i in cations if cell.symbols[i] in b_species and cell.symbols[i] not in a_species]
    only_a = [i for i in cations if cell.symbols[i] in a_species and cell.symbols[i] not in b_species]
    if len(only_b) > n_b or len(only_a) > n_a:
        raise ValueError("more B (or A) species sites than spinel B (or A) sites")

    def octahedral_first(sites):
        return sorted(sites, key=lambda i: (scores[i][0] - scores[i][1], scores[i][0], i))

    ambiguous = [i for i in cations if i not in set(only_b) and i not in set(only_a)]
    b_sites = sorted(only_b + octahedral_first(ambiguous)[:n_b - len(only_b)])
    if len(b_sites) < n_b:   # complete geometrically
        rest = octahedral_first([i for i in cations if i not in set(b_sites)])
        b_sites = sorted(b_sites + rest[:n_b - len(b_sites)])
    a_sites = sorted(i for i in cations if i not in set(b_sites))
    if len(b_sites) != n_b or len(a_sites) != n_a:
        raise ValueError("sublattice assignment failed")
    return a_sites, b_sites, x_sites


def _shell_adjacency(dist: np.ndarray, k: int) -> np.ndarray:
    """First-shell graph: pairs closer than 1.03 × median distance to the k-th neighbour."""
    k = min(k, dist.shape[0] - 1)
    cutoff = float(np.median(np.sort(dist, axis=1)[:, k - 1]) * 1.03)
    adj = ((dist <= cutoff) & np.isfinite(dist)).astype(np.int8)
    np.fill_diagonal(adj, 0)
    return np.maximum(adj, adj.T)


def _components(adj: np.ndarray) -> list[list[int]]:
    seen, components = set(), []
    for start in range(len(adj)):
        if start in seen:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            i = stack.pop()
            component.append(i)
            for j in np.flatnonzero(adj[i]):
                if int(j) not in seen:
                    seen.add(int(j))
                    stack.append(int(j))
        components.append(sorted(component))
    return components


def _graph(adj: np.ndarray, prefix: str) -> dict[str, float]:
    n = len(adj)
    A = adj.astype(float)
    degrees = A.sum(axis=1)
    components = _components(adj)
    triples = float(np.sum(degrees * np.maximum(degrees - 1.0, 0.0) / 2.0))
    triangles = float(np.trace(A @ A @ A) / 6.0)
    eig_l = np.sort(np.linalg.eigvalsh(np.diag(degrees) - A))
    paths: list[int] = []   # breadth-first path lengths of connected pairs
    for source in range(n):
        distance = [-1] * n
        distance[source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in np.flatnonzero(adj[u]):
                if distance[int(v)] < 0:
                    distance[int(v)] = distance[u] + 1
                    queue.append(int(v))
        paths.extend(distance[j] for j in range(source + 1, n) if distance[j] > 0)
    if paths:
        counts = np.array(list(Counter(paths).values()), dtype=float)
        probs = counts / counts.sum()
        path_entropy = (float(-np.sum(probs * np.log(probs)) / math.log(max(len(counts), 2)))
                        if len(counts) > 1 else 0.0)
        path_mean = float(np.mean(paths))
    else:
        path_entropy, path_mean = 0.0, 0.0
    # simple 4-cycles: (tr A⁴ − 2m − 4 Σ C(deg, 2)) / 8
    edges = float(np.sum(A) / 2.0)
    cycle4 = max((float(np.trace(np.linalg.matrix_power(A, 4))) - 2.0 * edges - 4.0 * triples)
                 / 8.0, 0.0)
    return {
        f"{prefix}_coordination_mean": float(np.mean(degrees)),
        f"{prefix}_coordination_std": float(np.std(degrees)),
        f"{prefix}_graph_degree_cv": _cv(degrees),
        f"{prefix}_graph_clustering": float(3.0 * triangles / triples) if triples > EPS else 0.0,
        f"{prefix}_graph_component_count": float(len(components)),
        f"{prefix}_graph_largest_component_fraction": float(max(map(len, components)) / n),
        f"{prefix}_graph_spectral_radius": float(np.max(np.abs(np.linalg.eigvalsh(A)))) / max(n, 1),
        f"{prefix}_graph_laplacian_gap": float(max(eig_l[1], 0.0)) if n > 1 else 0.0,
        f"{prefix}_graph_shortest_path_mean": path_mean,
        f"{prefix}_graph_path_entropy": path_entropy,
        f"{prefix}_graph_cycle_3_density": triangles / max(n, 1),
        f"{prefix}_graph_cycle_4_density": cycle4 / max(n, 1),
    }


def _shells(dist: np.ndarray, prefix: str, k1: int = 6, k2: int = 6) -> dict[str, float]:
    """First (k1 nearest) and second (next k2) neighbour-shell distance statistics."""
    n = len(dist)
    sorted_dist = np.sort(dist, axis=1)
    k1 = min(k1, n - 1)
    shell1 = sorted_dist[:, :k1].ravel()
    out = {f"{prefix}_d1_mean": _mean(shell1), f"{prefix}_d1_std": _std(shell1),
           f"{prefix}_d1_cv": _cv(shell1), f"{prefix}_d1_min": float(np.min(shell1)),
           f"{prefix}_d1_max": float(np.max(shell1))}
    out[f"{prefix}_shell_gap"] = (
        _mean((sorted_dist[:, k1] - sorted_dist[:, k1 - 1]) / np.maximum(sorted_dist[:, k1 - 1], EPS))
        if k1 < n - 1 else 0.0)
    if n - 1 > k1:
        shell2 = sorted_dist[:, k1:min(k1 + k2, n - 1)].ravel()
        out.update({f"{prefix}_d2_mean": _mean(shell2), f"{prefix}_d2_std": _std(shell2),
                    f"{prefix}_d2_cv": _cv(shell2), f"{prefix}_d2_min": float(np.min(shell2)),
                    f"{prefix}_d2_max": float(np.max(shell2))})
    near = sorted_dist[:, :min(max(k1, 1), n - 1)]
    out[f"{prefix}_distance_anisotropy"] = float(np.std(np.mean(near, axis=1))
                                                 / max(np.mean(near), EPS))
    return out


def _pairs(adj: np.ndarray) -> list[tuple[int, int]]:
    return [(i, j) for i in range(len(adj)) for j in range(i + 1, len(adj)) if adj[i, j]]


def _unlike_fraction(labels: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    return float(np.mean([labels[i] != labels[j] for i, j in pairs])) if pairs else 0.0


def _warren_cowley(labels: np.ndarray, pairs: list[tuple[int, int]], x: float) -> float:
    expected = 2.0 * x * (1.0 - x)
    if expected <= EPS or not pairs:
        return 0.0
    return float(1.0 - _unlike_fraction(labels, pairs) / expected)


# ---------------------------------------------------------------- descriptors

def descriptors(row: pd.Series, cell: Structure, elements: pd.DataFrame) -> dict[str, float]:
    """The 214 descriptors of one structure (``cell`` is the computational supercell)."""
    A, B0, B1, X = (_symbol(row[c]) for c in ("A_site", "B_original", "substitute", "X_site"))
    if not all((A, B0, X)):
        raise ValueError(f"invalid site symbols A={A} B={B0} X={X}")
    B1 = B1 or B0
    distances, vectors = minimum_image(cell)
    A_idx, B_idx, X_idx = sublattices(cell, A, B0, B1, X, distances, vectors)
    labels = np.array([1 if B1 != B0 and cell.symbols[i] == B1 else 0 for i in B_idx], dtype=int)
    x = float(np.mean(labels))   # B' fraction on the B sites

    def el(symbol: str, prop: str) -> float:
        return _element(elements, symbol, prop)

    f: dict[str, float] = {}
    lengths = np.linalg.norm(cell.lattice, axis=1)
    angles = np.array([_angle(cell.lattice[i], cell.lattice[j])
                       for i, j in ((1, 2), (0, 2), (0, 1))], dtype=float)
    f["cell_volume_A3"] = cell.volume
    f["volume_per_atom_A3"] = cell.volume / cell.nsites
    for label, value in zip("abc", lengths):
        f[f"lattice_{label}_A"] = float(value)
    f["lattice_length_anisotropy"] = _cv(lengths)
    f["lattice_angle_anisotropy"] = _std(angles) / 90.0

    # element properties of the sites
    bmix = {p: (1.0 - x) * el(B0, p) + x * el(B1, p)
            for p in set(SITE_PROPERTIES) | set(MISMATCH_PROPERTIES) | set(ALLOY_PROPERTIES)}
    for prop in SITE_PROPERTIES:
        for site, value in (("A", el(A, prop)), ("Bmix", bmix[prop]), ("X", el(X, prop))):
            if (site, prop) not in OMITTED_SITE_PROPERTIES:
                f[f"elem_{site}_{prop}"] = value
    for prop in MISMATCH_PROPERTIES:
        f[f"BX_{prop}_mismatch_norm"] = _abs_difference(bmix[prop], el(X, prop))
        if prop in AX_MISMATCH_PROPERTIES:
            f[f"AX_{prop}_mismatch_norm"] = _abs_difference(el(A, prop), el(X, prop))
    for prop in ALLOY_PROPERTIES:
        delta = _signed_difference(el(B1, prop), el(B0, prop))
        f[f"alloy_{prop}_variance"] = x * (1.0 - x) * delta * delta
        f[f"alloy_{prop}_signed_drift"] = x * delta

    # BX6 octahedra and relative p-d hopping
    bx_bonds, bx_t, distortion, distortion_rms, shape, tetragonal, trigonal, xbx_angles = (
        [] for _ in range(8))
    bonds_of_x: dict[int, list[tuple[int, np.ndarray, float, float]]] = {j: [] for j in X_idx}
    rX, chiX = el(X, "covalent_radius_A"), el(X, "pauling_EN")
    for local_b, bi in enumerate(B_idx):
        nearest = sorted(X_idx, key=lambda j: distances[bi, j])[:6]
        ds = np.array([distances[bi, j] for j in nearest], dtype=float)
        mean_d = float(np.mean(ds))
        deviations = (ds - mean_d) / max(mean_d, EPS)
        distortion.append(float(np.mean(np.abs(deviations))))
        distortion_rms.append(float(np.sqrt(np.mean(deviations ** 2))))
        units = np.array([vectors[bi, j] / max(distances[bi, j], EPS) for j in nearest], dtype=float)
        eig = np.sort(np.linalg.eigvalsh((units.T @ units) / len(units)))
        shape.append(float(np.std(eig) / max(np.mean(eig), EPS)))
        tetragonal.append(float(abs(eig[-1] - eig[0])))
        trigonal.append(float(abs(eig[1] - 0.5 * (eig[0] + eig[2]))))
        for ja, jb in itertools.combinations(nearest, 2):
            a = _angle(vectors[bi, ja], vectors[bi, jb])
            if np.isfinite(a):
                xbx_angles.append(a)
        rB, chiB = el(cell.symbols[bi], "covalent_radius_A"), el(cell.symbols[bi], "pauling_EN")
        for xj, d in zip(nearest, ds):
            t = float(((rB + rX) / max(d, EPS)) ** 3.5 / (1.0 + abs(chiB - chiX)))
            bx_bonds.append(float(d))
            bx_t.append(t)
            bonds_of_x[xj].append((local_b, -vectors[bi, xj], float(d), t))
    f.update({k: v for k, v in _stats(bx_bonds, "BX_distance", quantiles=True).items()
              if k != "BX_distance_cv"})   # = D_BX6_bond_cv
    f["BX6_distortion_mean"] = _mean(distortion)
    f["BX6_distortion_std"] = _std(distortion)
    f["BX6_distortion_cv"] = _cv(distortion)
    f["D_BX6_distortion_RMS"] = _mean(distortion_rms)
    f["D_BX6_bond_cv"] = _cv(bx_bonds)
    f["G_BX6_bond_regular"] = 1.0 / (1.0 + _cv(bx_bonds))
    f["octahedral_shape_anisotropy_mean"] = _mean(shape)
    f["octahedral_shape_anisotropy_std"] = _std(shape)
    f["crystal_field_tetragonal_split"] = _mean(tetragonal)
    f["crystal_field_trigonal_split"] = _mean(trigonal)
    xbx_dev = [min(abs(a - 90.0), abs(a - 180.0)) / 90.0 for a in xbx_angles]
    f["D_XBX_angle_disorder_norm"] = float(np.sqrt(np.mean(np.square(xbx_dev))))
    f["G_XBX_angle_regular"] = 1.0 / (1.0 + f["D_XBX_angle_disorder_norm"])
    f["XBX_angle_mean_deg"] = _mean(xbx_angles)
    f["XBX_angle_std_deg"] = _std(xbx_angles)
    f.update(_stats(bx_t, "BX_t_pd_sigma_rel"))

    # B-X-B paths through each X (its four nearest B) and effective B-B hopping
    bxb_angles, bxb_lengths, bxb_t, bxb_signed, bxb_class = [], [], [], [], []
    for items in bonds_of_x.values():
        for (bi, vi, di, ti), (bj, vj, dj, tj) in itertools.combinations(
                sorted(items, key=lambda z: z[2])[:4], 2):
            if bi == bj:
                continue
            angle = _angle(vi, vj)
            if not np.isfinite(angle):
                continue
            t = math.sqrt(max(ti * tj, 0.0))
            bxb_angles.append(angle)
            bxb_lengths.append(di + dj)
            bxb_t.append(t)
            bxb_signed.append(t * math.cos(math.radians(angle)))
            bxb_class.append("edge" if abs(angle - 90.0) <= 20.0
                             else "corner" if abs(angle - 180.0) <= 25.0 else "other")
    f["BXB_path_count_per_B"] = len(bxb_angles) / max(len(B_idx), 1)
    f.update(_stats(bxb_angles, "BXB_angle"))
    f["BXB_edge_fraction"] = float(np.mean(np.array(bxb_class) == "edge")) if bxb_class else 0.0
    f["BXB_corner_fraction"] = float(np.mean(np.array(bxb_class) == "corner")) if bxb_class else 0.0
    probs = np.array([bxb_class.count(c) for c in ("edge", "corner", "other")], dtype=float)
    probs = probs[probs > 0]
    f["BXB_path_entropy"] = (float(-np.sum((probs / probs.sum()) * np.log(probs / probs.sum()))
                                   / math.log(3.0)) if probs.size else 0.0)
    f.update(_stats(bxb_lengths, "BXB_path_length"))
    f["D_BXB_angle_disorder"] = _std(bxb_angles) / 90.0
    f["G_BXB_angle_coherence"] = 1.0 / (1.0 + f["D_BXB_angle_disorder"])
    f["D_BXB_connectivity_disorder_proxy"] = f["D_BXB_angle_disorder"] + 0.1 * f["D_BX6_bond_cv"]
    f.update(_stats(bxb_t, "BB_t_eff_rel"))
    edge_t = [t for t, c in zip(bxb_t, bxb_class) if c == "edge"]
    corner_t = [t for t, c in zip(bxb_t, bxb_class) if c == "corner"]
    f["BB_edge_t_eff_rel"] = _mean(edge_t) if edge_t else 0.0
    f["BB_corner_t_eff_rel"] = _mean(corner_t) if corner_t else 0.0
    f["BB_t_eff_signed_coherence"] = (abs(_mean(bxb_signed)) / max(_mean(np.abs(bxb_signed)), EPS)
                                      if bxb_signed else 0.0)

    # B-B and X-X sublattices
    b_dist = distances[np.ix_(B_idx, B_idx)].copy()
    x_dist = distances[np.ix_(X_idx, X_idx)].copy()
    b_adj, x_adj = _shell_adjacency(b_dist, 6), _shell_adjacency(x_dist, 6)
    f.update(_shells(b_dist, "BB"))
    f.update(_graph(b_adj, "BB"))
    f.update(_shells(x_dist, "XX"))
    f.update(_graph(x_adj, "XX"))
    xx_t = [((2.0 * rX) / max(x_dist[i, j], EPS)) ** 3.0 for i, j in _pairs(x_adj)]
    f.update(_stats(xx_t, "XX_t_pp_sigma_rel"))
    xx_strength = _mean(xx_t)   # = XX_t_pp_sigma_rel_mean
    f["XX_network_dephasing"] = _cv(xx_t) + f["XX_graph_degree_cv"]
    f["XX_network_coherence"] = xx_strength / (1.0 + f["XX_network_dephasing"])

    # B/B' configuration on the B-B graph
    shell1_pairs = _pairs(b_adj)
    order = np.argsort(b_dist, axis=1)
    nb = len(B_idx)
    shell2_pairs = sorted({tuple(sorted((i, int(j))))
                           for i in range(nb) for j in order[i, min(6, nb - 1):min(12, nb - 1)]
                           if i != int(j)})
    local = [float(np.mean(labels[np.flatnonzero(b_adj[i])])) if b_adj[i].any() else x
             for i in range(nb)]
    f["configuration_entropy"] = _binary_entropy(x)
    f["composition_disorder_B"] = x * (1.0 - x)
    f["composition_disorder_B_scaled4"] = 4.0 * x * (1.0 - x)
    f["local_environment_entropy"] = _mean([_binary_entropy(v) for v in local])
    f["octahedral_neighbor_mismatch"] = _mean(np.abs(np.asarray(local) - x))
    f["D_substitution_connectivity_dephasing"] = _std(local)
    f["Warren_Cowley_shell1"] = _warren_cowley(labels, shell1_pairs, x)
    f["Warren_Cowley_shell2"] = _warren_cowley(labels, shell2_pairs, x)
    f["B_Bprime_nn_pair_fraction"] = _unlike_fraction(labels, shell1_pairs)
    f["B_Bprime_nnn_pair_fraction"] = _unlike_fraction(labels, shell2_pairs)
    substituted = [i for i, v in enumerate(labels) if v == 1]
    sizes = ([len(c) for c in _components(b_adj[np.ix_(substituted, substituted)])]
             if substituted else [0])
    f["substitution_cluster_size_max_fraction"] = max(sizes) / max(nb, 1)
    f["substitution_cluster_size_mean_fraction"] = float(np.mean(sizes)) / max(nb, 1)

    # A-X cage (12 nearest X)
    ax_dist, a_off = [], []
    for ai in A_idx:
        nearest = sorted(X_idx, key=lambda j: distances[ai, j])[:min(12, len(X_idx))]
        ds = [distances[ai, j] for j in nearest]
        ax_dist.extend(ds)
        centroid = np.mean([vectors[ai, j] for j in nearest], axis=0)
        a_off.append(float(np.linalg.norm(centroid) / max(np.mean(ds), EPS)))
    f.update(_stats(ax_dist, "AX_distance"))
    f["A_offcentering"] = _mean(a_off)
    f["A_offcentering_std"] = _std(a_off)

    # combinations
    chiA, chiB = el(A, "pauling_EN"), bmix["pauling_EN"]
    rA, rB, rXi = el(A, "ionic_radius_A"), bmix["ionic_radius_A"], el(X, "ionic_radius_A")
    dB = bmix["d_electrons"]
    f["onsite_BX_chi_abs"] = abs(chiB - chiX)
    f["onsite_AX_chi_abs"] = abs(chiA - chiX)
    f["onsite_alignment_inverse"] = 1.0 / (1.0 + abs(chiB - chiX))
    f["B_d_filling_mean"] = dB
    f["onsite_B_d_filling"] = dB / 10.0
    f["B_valence_filling_mean"] = bmix["valence_electrons"]
    f["alloy_d_filling_dephasing"] = x * (1.0 - x) * ((el(B1, "d_electrons")
                                                      - el(B0, "d_electrons")) / 10.0) ** 2
    f["C_BX_EN_compatibility"] = math.exp(-abs(chiB - chiX) / 3.0)
    f["C_BX_radial_overlap_proxy"] = 1.0 / (1.0 + _abs_difference(rB + rXi, _mean(bx_bonds)))
    f["C_Bmix_d_count_compatibility"] = 1.0 / (1.0 + abs(dB - 5.0) / 10.0)
    f["D_config_chemical_disorder"] = _binary_entropy(x) * (
        abs(el(B1, "pauling_EN") - el(B0, "pauling_EN")) / 3.0
        + _abs_difference(el(B1, "ionic_radius_A"), el(B0, "ionic_radius_A")))
    f["A_radius_ratio_correction"] = (rA + rXi) / max(rB + rXi, EPS)
    f["A_EN_cage_compatibility"] = math.exp(-abs(chiA - chiX) / 3.0)
    f["A_cage_lattice_anisotropy"] = f["lattice_length_anisotropy"] + f["lattice_angle_anisotropy"]
    f["A_cage_volume_packing_proxy"] = ((rA + rXi) / max(f["AX_distance_mean"], EPS)) ** 3
    f["A_network_tilt_coupling"] = f["AX_distance_cv"] * f["D_BXB_angle_disorder"]
    f["crystal_field_oct_strength_rel"] = (2.0 / max(_mean(bx_bonds), EPS)) ** 5
    f["local_symmetry_breaking"] = (f["D_BX6_distortion_RMS"] + f["D_XBX_angle_disorder_norm"]
                                    + f["octahedral_shape_anisotropy_mean"])
    f["BX_second_order_hopping_rel"] = _mean(bx_t) ** 2 / (1.0 + abs(chiB - chiX))
    f["BB_network_dephasing"] = (f["D_BXB_angle_disorder"] + f["BB_d1_cv"]
                                 + (1.0 - f["BB_t_eff_signed_coherence"]))
    f["BB_network_coherence"] = f["BB_t_eff_signed_coherence"] / (1.0 + f["BB_network_dephasing"])

    f = {k: float(v) for k, v in f.items()}
    if not all(np.isfinite(v) for v in f.values()):
        raise ValueError(f"non-finite descriptor for {row[ID_COLUMN]}")
    return f


def published_values(table: pd.DataFrame) -> pd.DataFrame:
    """The values as the published searches read them.

    Their descriptor matrices passed once through a CSV written with ``%.17g`` and read
    with pandas' default float parser, which can move a value by a few units in the last
    place; this repeats that step.
    """
    text = table.to_csv(index=False, float_format="%.17g")
    names = list(table.columns[1:])
    return pd.read_csv(io.StringIO(text)).astype({n: np.float64 for n in names})


def descriptor_table(structures: pd.DataFrame, elements: pd.DataFrame,
                     sample_ids: Sequence[str] | None = None) -> pd.DataFrame:
    """``sample_id`` and the 214 descriptors, one row per structure (in ``sample_ids`` order)."""
    rows = structures.set_index(structures[ID_COLUMN].astype(str), drop=False)
    if sample_ids is not None:
        rows = rows.loc[[str(s) for s in sample_ids]]
    values = [descriptors(row, supercell(structure(row)), elements) for _, row in rows.iterrows()]
    table = pd.DataFrame(values)
    table.insert(0, "sample_id", rows.index.to_numpy())
    return published_values(table)
