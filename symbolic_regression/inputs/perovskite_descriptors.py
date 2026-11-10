"""Single (SP) and double (DP) perovskite descriptors computed from the structures.

``descriptor_table(structures, family)`` gives one row per state of
``data/descriptor_tables/structures/perovskite_structures.csv`` with the
descriptors of ``data/descriptor_tables/sp`` (111) or ``dp`` (182), in
alphabetical order. Each state is its relaxed POSCAR (the mode-designed SP
supercells) or the 40-atom geometry. The groups are

* ``cell_*``: volume per atom, lattice lengths, mass density;
* ``chem_{A,B,X}_*``: mean, spread and range of six element primitives over the
  A, B and X sites; ``chem_{BX,AX}_*_mismatch``: |B − X| and |A − X| of the means;
* ``oct_BX6_*`` and ``cage_AX_*``: bond lengths, off-centring and (BX₆) bond-angle
  deviations from 90°/180° of the 6 (12) nearest X around each B (A);
* ``network_*``: B–X–B angles through each X and the nearest B–B distances;
* ``order_*``: B-site order from Fourier amplitudes of the B atomic numbers at
  R, X and M, their spread and the species entropy;
* DP only: the same chemistry and octahedra on the substituted (``Bvar``) and
  fixed (``Bref``) B sublattices and their contrasts ``dp_*_contrast``.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .poscar import Structure, decode, parse_poscar

ID_COLUMN = "Task Directory"
FAMILY = {"sp": "perovskite", "dp": "double_perovskite"}

# Element primitives: atomic number, group, period, atomic mass, covalent and
# van der Waals radii (Å).
ELEMENTS: dict[str, dict[str, float]] = {
    "O": {"Z": 8, "group": 16, "period": 2, "mass": 15.999, "rcov": 0.66, "rvdw": 1.55},
    "Na": {"Z": 11, "group": 1, "period": 3, "mass": 22.990, "rcov": 1.66, "rvdw": 2.40},
    "Mg": {"Z": 12, "group": 2, "period": 3, "mass": 24.305, "rcov": 1.41, "rvdw": 2.20},
    "Al": {"Z": 13, "group": 13, "period": 3, "mass": 26.982, "rcov": 1.21, "rvdw": 2.10},
    "K": {"Z": 19, "group": 1, "period": 4, "mass": 39.098, "rcov": 2.03, "rvdw": 2.80},
    "Ca": {"Z": 20, "group": 2, "period": 4, "mass": 40.078, "rcov": 1.76, "rvdw": 2.40},
    "Sc": {"Z": 21, "group": 3, "period": 4, "mass": 44.956, "rcov": 1.70, "rvdw": 2.30},
    "Ti": {"Z": 22, "group": 4, "period": 4, "mass": 47.867, "rcov": 1.60, "rvdw": 2.15},
    "Zn": {"Z": 30, "group": 12, "period": 4, "mass": 65.38, "rcov": 1.22, "rvdw": 2.10},
    "Ga": {"Z": 31, "group": 13, "period": 4, "mass": 69.723, "rcov": 1.22, "rvdw": 2.10},
    "Zr": {"Z": 40, "group": 4, "period": 5, "mass": 91.224, "rcov": 1.75, "rvdw": 2.30},
    "Nb": {"Z": 41, "group": 5, "period": 5, "mass": 92.906, "rcov": 1.64, "rvdw": 2.15},
    "Mo": {"Z": 42, "group": 6, "period": 5, "mass": 95.95, "rcov": 1.54, "rvdw": 2.10},
    "In": {"Z": 49, "group": 13, "period": 5, "mass": 114.818, "rcov": 1.42, "rvdw": 2.20},
    "Hf": {"Z": 72, "group": 4, "period": 6, "mass": 178.49, "rcov": 1.75, "rvdw": 2.25},
    "Ta": {"Z": 73, "group": 5, "period": 6, "mass": 180.948, "rcov": 1.70, "rvdw": 2.20},
    "W": {"Z": 74, "group": 6, "period": 6, "mass": 183.84, "rcov": 1.62, "rvdw": 2.10},
    "La": {"Z": 57, "group": 3, "period": 6, "mass": 138.905, "rcov": 2.07, "rvdw": 2.50},
    "Sr": {"Z": 38, "group": 2, "period": 5, "mass": 87.62, "rcov": 1.95, "rvdw": 2.55},
    "Ba": {"Z": 56, "group": 2, "period": 6, "mass": 137.327, "rcov": 2.15, "rvdw": 2.70},
}
PROPERTIES = ("Z", "group", "period", "mass", "rcov", "rvdw")
SHIFTS_27 = np.asarray([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
                       float)


def _present(value) -> bool:
    return (value is not None and not (isinstance(value, float) and np.isnan(value))
            and str(value).strip() not in {"", "nan", "None"})


def structure(row: pd.Series) -> Structure:
    """Relaxed supercell of the mode-designed SP states, else the 40-atom geometry."""
    if _present(row.get("POSCAR_relaxed_supercell")):
        return parse_poscar(str(row["POSCAR_relaxed_supercell"]))
    return parse_poscar(decode(row["geometry_40atom_poscar_gz_b64"]))


def _displacements(frac_from: np.ndarray, frac_to: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Minimum-image vectors from one point to many (27-image search, safe for skewed cells)."""
    base = np.asarray(frac_to, float) - np.asarray(frac_from, float)[None, :]
    cart = (base[:, None, :] + SHIFTS_27[None, :, :]) @ lattice
    pick = np.argmin(np.sum(cart * cart, axis=2), axis=1)
    return cart[np.arange(len(base)), pick]


def _stats(prefix: str, values: Iterable[float], out: dict[str, float]) -> None:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    mean, std = float(np.mean(x)), float(np.std(x))
    out[f"{prefix}_mean"] = mean
    out[f"{prefix}_std"] = std
    out[f"{prefix}_min"] = float(np.min(x))
    out[f"{prefix}_max"] = float(np.max(x))
    out[f"{prefix}_cv"] = std / max(abs(mean), 1e-12)


def _sites(symbols: tuple[str, ...], allowed: set[str]) -> np.ndarray:
    return np.asarray([i for i, s in enumerate(symbols) if s in allowed], dtype=np.int32)


def _prop(symbol: str, name: str) -> float:
    return float(ELEMENTS[symbol][name])


def _chemistry(cell: Structure, sites: np.ndarray, prefix: str, out: dict[str, float]) -> None:
    symbols = [cell.symbols[int(i)] for i in sites]
    for prop in PROPERTIES:
        values = np.asarray([_prop(s, prop) for s in symbols], float)
        out[f"{prefix}_{prop}_mean"] = float(np.mean(values))
        out[f"{prefix}_{prop}_std"] = float(np.std(values))
        out[f"{prefix}_{prop}_range"] = float(np.max(values) - np.min(values))


def _coordination(cell: Structure, centers: np.ndarray, ligands: np.ndarray, prefix: str,
                  out: dict[str, float], n_neighbours: int) -> None:
    """Bonds to the n nearest ligands of each center; angle deviations for octahedra (n = 6)."""
    bonds, site_stds, site_means, offcenter, angle_devs = [], [], [], [], []
    ligand_frac = cell.frac[ligands]
    for center in centers:
        vec = _displacements(cell.frac[int(center)], ligand_frac, cell.lattice)
        dist = np.linalg.norm(vec, axis=1)
        order = np.argsort(dist, kind="stable")[:n_neighbours]
        v, d = vec[order], dist[order]
        bonds.extend(d.tolist())
        site_stds.append(float(np.std(d)))
        site_means.append(float(np.mean(d)))
        offcenter.append(float(np.linalg.norm(np.mean(v, axis=0)) / max(float(np.mean(d)), 1e-12)))
        if n_neighbours == 6:
            for a in range(6):
                for b in range(a + 1, 6):
                    den = max(np.linalg.norm(v[a]) * np.linalg.norm(v[b]), 1e-15)
                    angle = math.degrees(math.acos(float(np.clip(np.dot(v[a], v[b]) / den, -1, 1))))
                    angle_devs.append(min(abs(angle - 90.0), abs(angle - 180.0)))
    bonds = np.asarray(bonds, float)
    out[f"{prefix}_bond_mean"] = float(np.mean(bonds))
    out[f"{prefix}_bond_std"] = float(np.std(bonds))
    out[f"{prefix}_bond_cv"] = float(np.std(bonds) / max(np.mean(bonds), 1e-12))
    out[f"{prefix}_bond_min"] = float(np.min(bonds))
    out[f"{prefix}_bond_max"] = float(np.max(bonds))
    out[f"{prefix}_site_bond_std_mean"] = float(np.mean(site_stds))
    out[f"{prefix}_site_bond_mean_std"] = float(np.std(site_means))
    out[f"{prefix}_offcenter_mean"] = float(np.mean(offcenter))
    out[f"{prefix}_offcenter_std"] = float(np.std(offcenter))
    if angle_devs:
        devs = np.asarray(angle_devs, float)
        out[f"{prefix}_angle_dev_mean"] = float(np.mean(devs))
        out[f"{prefix}_angle_dev_rms"] = float(np.sqrt(np.mean(devs * devs)))
        out[f"{prefix}_angle_dev_max"] = float(np.max(devs))
    else:
        out[f"{prefix}_angle_dev_mean"] = out[f"{prefix}_angle_dev_rms"] = 0.0
        out[f"{prefix}_angle_dev_max"] = 0.0


def _network(cell: Structure, b_sites: np.ndarray, x_sites: np.ndarray,
             out: dict[str, float]) -> None:
    """B–X–B angle through each X to its two nearest B; six nearest B–B distances per B."""
    angles, hetero = [], []
    b_frac = cell.frac[b_sites]
    for xi in x_sites:
        vec = _displacements(cell.frac[int(xi)], b_frac, cell.lattice)
        order = np.argsort(np.linalg.norm(vec, axis=1), kind="stable")[:2]
        v1, v2 = vec[order[0]], vec[order[1]]
        den = max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-15)
        angles.append(math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / den, -1, 1)))))
        hetero.append(float(cell.symbols[int(b_sites[order[0]])]
                            != cell.symbols[int(b_sites[order[1]])]))
    angle = np.asarray(angles, float)
    dev = 180.0 - angle
    out["network_BXB_angle_mean"] = float(np.mean(angle))
    out["network_BXB_angle_std"] = float(np.std(angle))
    out["network_BXB_deviation_mean"] = float(np.mean(np.abs(dev)))
    out["network_BXB_deviation_rms"] = float(np.sqrt(np.mean(dev * dev)))
    out["network_BXB_cos_mean"] = float(np.mean(np.cos(np.deg2rad(angle))))
    out["network_heterobridge_fraction"] = float(np.mean(hetero))
    nearest = []
    for i, bi in enumerate(b_sites):
        others = np.delete(b_sites, i)
        d = np.linalg.norm(_displacements(cell.frac[int(bi)], cell.frac[others], cell.lattice), axis=1)
        nearest.extend(np.sort(d)[:min(6, len(d))].tolist())
    _stats("network_BB_distance", nearest, out)


def _order(cell: Structure, b_sites: np.ndarray, out: dict[str, float]) -> None:
    """|Σ w e^{2πi q·r}| / Σ|w| over the B sites, w = Z − mean Z (w = 1 when all Z are equal)."""
    z = np.asarray([_prop(cell.symbols[int(i)], "Z") for i in b_sites], float)
    weights = z - np.mean(z)
    norm = float(np.sum(np.abs(weights)))
    if norm <= 1e-12:
        norm = float(len(weights))
        weights = np.ones_like(weights)
    b_frac = cell.frac[b_sites]

    def amplitude(q):
        return float(abs(np.sum(weights * np.exp(2j * np.pi * (b_frac @ np.asarray(q, float)))))
                     / norm)

    out["order_R"] = amplitude((1, 1, 1))
    out["order_X_rms"] = float(np.sqrt(np.mean(np.square(
        [amplitude((1, 0, 0)), amplitude((0, 1, 0)), amplitude((0, 0, 1))]))))
    out["order_M_rms"] = float(np.sqrt(np.mean(np.square(
        [amplitude((1, 1, 0)), amplitude((1, 0, 1)), amplitude((0, 1, 1))]))))
    out["order_species_Z_std"] = float(np.std(z))
    _, counts = np.unique([cell.symbols[int(i)] for i in b_sites], return_counts=True)
    fractions = counts / counts.sum()
    out["order_species_entropy"] = float(-np.sum(fractions * np.log(np.maximum(fractions, 1e-15))))


def descriptors(row: pd.Series, cell: Structure, family: str) -> dict[str, float]:
    """Descriptors of one state (family ``sp`` or ``dp``), alphabetical."""
    a_sites = _sites(cell.symbols, {str(row["A_site"]).strip()})
    x_sites = _sites(cell.symbols, {str(row["X_site"]).strip()})
    b_symbols = {str(row["B_original"]).strip(), str(row["substitute"]).strip()}
    if _present(row.get("B_fixed")):
        b_symbols.add(str(row["B_fixed"]).strip())
    b_sites = _sites(cell.symbols, b_symbols - {"nan", ""})
    if len(a_sites) == 0 or len(b_sites) < 2 or len(x_sites) < 6:
        raise ValueError(f"site assignment failed for {row[ID_COLUMN]}")
    out: dict[str, float] = {}
    lengths = np.linalg.norm(cell.lattice, axis=1)
    volume = abs(float(np.linalg.det(cell.lattice)))
    out["cell_volume_per_atom"] = volume / max(cell.nsites, 1)
    out["cell_lattice_mean"] = float(np.mean(lengths))
    out["cell_lattice_std"] = float(np.std(lengths))
    out["cell_lattice_anisotropy"] = float(np.std(lengths) / max(np.mean(lengths), 1e-12))
    masses = np.asarray([_prop(s, "mass") for s in cell.symbols], float)
    out["cell_mass_density_proxy"] = float(np.sum(masses) / max(volume, 1e-12))
    _chemistry(cell, a_sites, "chem_A", out)
    _chemistry(cell, b_sites, "chem_B", out)
    _chemistry(cell, x_sites, "chem_X", out)
    for prop in PROPERTIES:
        out[f"chem_BX_{prop}_mismatch"] = abs(out[f"chem_B_{prop}_mean"] - out[f"chem_X_{prop}_mean"])
        out[f"chem_AX_{prop}_mismatch"] = abs(out[f"chem_A_{prop}_mean"] - out[f"chem_X_{prop}_mean"])
    _coordination(cell, b_sites, x_sites, "oct_BX6", out, 6)
    _coordination(cell, a_sites, x_sites, "cage_AX", out, min(12, len(x_sites)))
    _network(cell, b_sites, x_sites, out)
    _order(cell, b_sites, out)
    if family == "dp":
        # substituted B sublattice (original/substitute) against the fixed one
        varied = _sites(cell.symbols, {str(row["B_original"]).strip(), str(row["substitute"]).strip()})
        fixed = _sites(cell.symbols, {str(row["B_fixed"]).strip()})
        _chemistry(cell, varied, "chem_Bvar", out)
        _chemistry(cell, fixed, "chem_Bref", out)
        _coordination(cell, varied, x_sites, "oct_BvarX6", out, 6)
        _coordination(cell, fixed, x_sites, "oct_BrefX6", out, 6)
        for prop in PROPERTIES:
            out[f"dp_BBprime_{prop}_contrast"] = abs(out[f"chem_Bvar_{prop}_mean"]
                                                     - out[f"chem_Bref_{prop}_mean"])
        for key in ("bond_mean", "bond_std", "bond_cv", "offcenter_mean", "angle_dev_rms"):
            out[f"dp_oct_{key}_contrast"] = abs(out[f"oct_BvarX6_{key}"] - out[f"oct_BrefX6_{key}"])
    return {k: out[k] for k in sorted(out)}


def descriptor_table(structures: pd.DataFrame, family: str,
                     sample_ids: Sequence[str] | None = None) -> pd.DataFrame:
    """``sample_id`` and the descriptors of every state of the family (in ``sample_ids`` order)."""
    rows = structures.loc[structures["family"].astype(str) == FAMILY[family]]
    rows = rows.set_index(rows[ID_COLUMN].astype(str), drop=False)
    if sample_ids is not None:
        rows = rows.loc[[str(s) for s in sample_ids]]
    table = pd.DataFrame([descriptors(row, structure(row), family) for _, row in rows.iterrows()])
    table = table.astype(np.float64)
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError("non-finite descriptor")
    table.insert(0, "sample_id", rows.index.to_numpy())
    return table
