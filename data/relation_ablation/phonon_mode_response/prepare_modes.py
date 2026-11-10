"""T3 frozen-phonon inputs.
Mass-weighted dynamical matrix D = M^-1/2 H M^-1/2 (translations projected out); eigenmodes classified by the
bond-preserving share of their network displacement (>=0.7 bending, <=0.3 stretching); the 6 lowest-frequency modes of
each class (omega^2 > 0) are displaced so that the largest atomic displacement is 0.04 A (+ and -). Writes POSCARs under
OUTDIR/<system directory>/<job>/ (k12, ca_tio3, ca_sno3) and modes.json.
usage: python prepare_modes.py OUTDIR"""
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2]
FORCE_BALANCE_DIR = DATA / "force_balance_reference"
PRESSURE_PATH_DIR = DATA / "spinel_pressure_path" / "force_balance"
BENDING_FULL_DIR = DATA / "bending_enrichment" / "full_force_balance"
MASS = {"Mg": 24.305, "Sc": 44.956, "Sb": 121.760, "Se": 78.971, "Ca": 40.078, "Ti": 47.867, "Sn": 118.710, "O": 15.999}
MAX_DISPLACEMENT = 0.04
SYSTEM_DIRECTORY = {"k12": "k12", "CaTiO3": "ca_tio3", "CaSnO3": "ca_sno3"}  # directory of each system on disk
OUT_DIR = Path(sys.argv[1])


def hessian_from_forces(read_forces, N, delta):
    hessian = np.zeros((3 * N, 3 * N))
    for i in range(N):
        for a, ax in enumerate("xyz"):
            hessian[3 * i + a] = -(read_forces(f"d{i:02d}{ax}p") - read_forces(f"d{i:02d}{ax}m")).ravel() / (2 * delta)
    return 0.5 * (hessian + hessian.T)


def bp_projector(f, L, bonds, net, N):
    col = {int(a): c for c, a in enumerate(net)}
    J = np.zeros((len(bonds), 3 * len(net)))
    for r, (i, j, img) in enumerate(bonds):
        v = (f[j] - f[i] + np.array(img)) @ L
        e = v / np.linalg.norm(v)
        J[r, 3 * col[i]:3 * col[i] + 3] = -e
        J[r, 3 * col[j]:3 * col[j] + 3] = e
    _, s, Vt = np.linalg.svd(J)
    r = int(np.sum(s > 1e-9 * s[0]))
    R = Vt[:r].T  # row space = bond-changing
    # share of the network displacement u that is bond-preserving
    return lambda u: 1 - np.sum((R.T @ u) ** 2) / (u @ u)


def poscar(L, species, counts, frac, title):
    s = ([title, "1.0"] + ["  %.16f %.16f %.16f" % tuple(r) for r in L]
         + ["  " + "  ".join(species), "  " + "  ".join(map(str, counts)), "Direct"])
    return "\n".join(s + ["  %.16f %.16f %.16f" % tuple(x % 1.0) for x in frac]) + "\n"


def build(name, H, L, species, counts, fref, bonds, net):
    species_per_atom = np.repeat(species, counts)
    N = len(species_per_atom)
    masses = np.repeat([MASS[x] for x in species_per_atom], 3)
    inv_sqrt_mass = 1 / np.sqrt(masses)
    T = np.zeros((3 * N, 3))
    for a in range(3):
        T[a::3, a] = np.sqrt(masses[a::3])
        T[:, a] /= np.linalg.norm(T[:, a])
    P = np.eye(3 * N) - T @ T.T
    dynamical_matrix = P @ (inv_sqrt_mass[:, None] * H * inv_sqrt_mass[None, :]) @ P
    w2, E = np.linalg.eigh(dynamical_matrix)
    keep = np.abs(E.T @ T).max(axis=1) < 1e-6  # drop the 3 translations
    share = bp_projector(fref, L, bonds, net, N)
    idx = np.ravel([[3 * a, 3 * a + 1, 3 * a + 2] for a in net])
    modes = []
    for k in np.flatnonzero(keep):
        u = (inv_sqrt_mass * E[:, k])
        un = u[idx].reshape(-1, 3)
        un = (un - un.mean(0)).ravel()
        modes.append({"k": int(k), "w2_eV_A2_amu": float(w2[k]), "bp_share": float(share(un)), "u": u})
    neg = [x for x in modes if x["w2_eV_A2_amu"] <= 0]
    selected = {"bend": [x for x in modes if x["bp_share"] >= 0.7 and x["w2_eV_A2_amu"] > 0][:6],
                "stretch": [x for x in modes if x["bp_share"] <= 0.3 and x["w2_eV_A2_amu"] > 0][:6]}
    system_dir = OUT_DIR / SYSTEM_DIRECTORY[name]
    (system_dir / "ref").mkdir(parents=True, exist_ok=True)
    (system_dir / "ref" / "POSCAR").write_text(poscar(L, species, counts, fref, f"{name} T3 reference"))
    record = {"system": name, "n_modes": len(modes), "n_negative_w2": len(neg),
              "n_bend_total": sum(x["bp_share"] >= 0.7 for x in modes),
              "n_stretch_total": sum(x["bp_share"] <= 0.3 for x in modes), "selected": []}
    for cls, mode_list in selected.items():
        for n, x in enumerate(mode_list):
            u = x["u"].reshape(-1, 3)
            amplitude_scale = MAX_DISPLACEMENT / np.linalg.norm(u, axis=1).max()  # Q amplitude in sqrt(amu)*A
            for direction, tag in ((1, "p"), (-1, "m")):
                frac_displaced = fref + (direction * amplitude_scale * u) @ np.linalg.inv(L)
                job_dir = system_dir / f"{cls}{n}_{tag}"
                job_dir.mkdir(exist_ok=True)
                job_dir.joinpath("POSCAR").write_text(poscar(
                    L, species, counts, frac_displaced,
                    f"{name} T3 {cls}{n} {tag} Q={direction * amplitude_scale:.6f}"))
            record["selected"].append({"label": f"{cls}{n}", "class": cls, "mode_index": x["k"],
                                       "w2_eV_A2_amu": x["w2_eV_A2_amu"], "bp_share": x["bp_share"],
                                       "Q_amp_sqrtamu_A": float(amplitude_scale),
                                       "eigvec_cart_per_sqrtamu": x["u"].tolist()})
    print(name, {k: v for k, v in record.items() if k != "selected"},
          [(r["label"], round(r["w2_eV_A2_amu"], 4), round(r["bp_share"], 3)) for r in record["selected"]])
    return record


out = []
meta = json.loads((PRESSURE_PATH_DIR / "meta.json").read_text())
symbols = list(np.repeat(meta["species"], meta["counts"]))
N = len(symbols)
H = hessian_from_forces(lambda c: np.load(FORCE_BALANCE_DIR / "forces" / f"{c}.npy"), N, meta["delta_A"])
bonds = json.loads((FORCE_BALANCE_DIR / "inputs" / "pressure_bond_accommodation.json").read_text())["bonds"]
out.append(build("k12", H, np.array(meta["lattice_5GPa"]), meta["species"], meta["counts"],
                 np.array(meta["frac_5GPa"]), bonds, np.flatnonzero(np.isin(symbols, ["Sc", "Sb", "Se"]))))
for mname, subdir in (("CaTiO3", "ca_tio3"), ("CaSnO3", "ca_sno3")):
    work_dir = BENDING_FULL_DIR / subdir / "p0_p5"
    meta = json.loads((work_dir / "meta.json").read_text())
    N = sum(meta["counts"])
    H = hessian_from_forces(lambda c: np.load(work_dir / "hessian" / c / "forces.npy"), N, meta["delta_A"])
    bonds = meta["bonds"]
    net = np.array(sorted({b[0] for b in bonds} | {b[1] for b in bonds}))
    out.append(build(mname, H, np.array(meta["lattice_P2"]), meta["species"], meta["counts"], np.array(meta["frac_P2"]),
                     bonds, net))
(OUT_DIR / "modes.json").write_text(json.dumps(out))
