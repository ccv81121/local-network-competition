"""Thermal (ZG special-displacement and Monte Carlo) configurations at the tight 4- and 5-GPa structures.
H: 168x168 force constants of the affine reference (force_balance_reference, assembled as in
relation_ablation/phonon_mode_response/prepare_modes.py).
Modes: mass-weighted, 3 translations dropped, w2 <= 0 excluded and counted; class by the bond-preserving share of the
centred network displacement (bend >= 0.7, stretch <= 0.3, else mixed).
<Q^2> = (hbar/2w) coth(hbar w/2kT) in amu A^2. ZG: modes sorted by frequency, signs alternate (+,-,+,...);
pair = (+u, -u). MC: Q_nu ~ N(0, <Q^2>), seed 20261025, 4 antithetic pairs. Displacements are Cartesian, added to the
reference POSCAR.
Needs inputs/p4/POTCAR and inputs/p5/POTCAR (VASP licence, not deposited).
usage: python thermal_prep.py OUTDIR
(writes OUTDIR/jobs/<job>/{POSCAR,INCAR,KPOINTS,POTCAR}, modes_thermal.json, thermal_input_hashes.json)"""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
FORCE_BALANCE_DIR = DATA / "force_balance_reference"
PRESSURE_PATH_DIR = DATA / "spinel_pressure_path" / "force_balance"
INPUT_DIR = HERE / "inputs"
OUT_DIR = Path(sys.argv[1])
MASS = {"Mg": 24.305, "Sc": 44.956, "Sb": 121.760, "Se": 78.971}
EV, AMU, ANGSTROM, HBAR, KB = 1.602176634e-19, 1.66053906660e-27, 1e-10, 1.054571817e-34, 1.380649e-23


def file_sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


meta = json.loads((PRESSURE_PATH_DIR / "meta.json").read_text())
species, counts = meta["species"], meta["counts"]
symbols = list(np.repeat(species, counts))
N = len(symbols)
delta = meta["delta_A"]
H = np.zeros((3 * N, 3 * N))
for i in range(N):
    for a, ax in enumerate("xyz"):
        H[3 * i + a] = -(np.load(FORCE_BALANCE_DIR / "forces" / f"d{i:02d}{ax}p.npy")
                         - np.load(FORCE_BALANCE_DIR / "forces" / f"d{i:02d}{ax}m.npy")).ravel() / (2 * delta)
H = 0.5 * (H + H.T)
masses = np.repeat([MASS[x] for x in symbols], 3)
inv_sqrt_mass = 1 / np.sqrt(masses)
T = np.zeros((3 * N, 3))
for a in range(3):
    T[a::3, a] = np.sqrt(masses[a::3])
    T[:, a] /= np.linalg.norm(T[:, a])
P = np.eye(3 * N) - T @ T.T
w2, E = np.linalg.eigh(P @ (inv_sqrt_mass[:, None] * H * inv_sqrt_mass[None, :]) @ P)
keep = np.abs(E.T @ T).max(axis=1) < 1e-6

# bond-preserving share of the centred network displacement (same construction as prepare_modes.bp_projector)
bonds = json.loads((FORCE_BALANCE_DIR / "inputs" / "pressure_bond_accommodation.json").read_text())["bonds"]
lattice_5gpa = np.array(meta["lattice_5GPa"])
frac_5gpa = np.array(meta["frac_5GPa"])
network_sites = np.flatnonzero(np.isin(symbols, ["Sc", "Sb", "Se"]))
site_column = {int(a): c for c, a in enumerate(network_sites)}
J = np.zeros((len(bonds), 3 * len(network_sites)))
for r, (i, j, img) in enumerate(bonds):
    v = (frac_5gpa[j] - frac_5gpa[i] + np.array(img)) @ lattice_5gpa
    e = v / np.linalg.norm(v)
    J[r, 3 * site_column[i]:3 * site_column[i] + 3] = -e
    J[r, 3 * site_column[j]:3 * site_column[j] + 3] = e
_, s, Vt = np.linalg.svd(J)
R = Vt[:int(np.sum(s > 1e-9 * s[0]))].T
network_dofs = np.ravel([[3 * a, 3 * a + 1, 3 * a + 2] for a in network_sites])
modes = []
for k in np.flatnonzero(keep):
    u = inv_sqrt_mass * E[:, k]
    un = u[network_dofs].reshape(-1, 3)
    un = (un - un.mean(0)).ravel()
    bp = float(1 - np.sum((R.T @ un) ** 2) / (un @ un))
    modes.append({"k": int(k), "w2": float(w2[k]), "bp": bp,
                  "cls": "bend" if bp >= 0.7 else ("stretch" if bp <= 0.3 else "mixed"), "u": u})
neg = [x for x in modes if x["w2"] <= 0]
modes = sorted([x for x in modes if x["w2"] > 0], key=lambda x: x["w2"])


def q2(w2v, Tk):
    w = np.sqrt(w2v * EV / (ANGSTROM ** 2 * AMU))
    c = 1.0 if Tk == 0 else 1 / np.tanh(HBAR * w / (2 * KB * Tk))
    return HBAR / (2 * w) * c / (AMU * ANGSTROM ** 2)


def zg(Tk, cls=None, scale=1.0):
    sel = [x for x in modes if cls is None or x["cls"] == cls]
    return scale * sum(((-1) ** n) * np.sqrt(q2(x["w2"], Tk)) * x["u"] for n, x in enumerate(sel))


rng = np.random.default_rng(20261025)


def mc(Tk):
    return sum(rng.normal(0.0, np.sqrt(q2(x["w2"], Tk))) * x["u"] for x in modes)


def read_pos(p):
    L = p.read_text().splitlines()
    lat = np.array([[float(x) for x in L[i].split()[:3]] for i in (2, 3, 4)]) * float(L[1])
    assert L[5].split() == species and [int(x) for x in L[6].split()] == counts and L[7].strip()[0] in "dD"
    return lat, np.array([[float(x) for x in L[8 + j].split()[:3]] for j in range(N)])


def write_job(name, pres, u):
    reference_dir = INPUT_DIR / pres.lower()  # inputs/p4, inputs/p5
    lat, fr = read_pos(reference_dir / "POSCAR")
    d = OUT_DIR / "jobs" / name
    d.mkdir(parents=True, exist_ok=True)
    fr2 = fr + u.reshape(-1, 3) @ np.linalg.inv(lat)
    lines = ([f"{name} thermal-valley", "1.0"] + ["  %.16f %.16f %.16f" % tuple(r) for r in lat]
             + ["  " + "  ".join(species), "  " + "  ".join(map(str, counts)), "Direct"]
             + ["  %.16f %.16f %.16f" % tuple(x % 1.0) for x in fr2])
    (d / "POSCAR").write_text("\n".join(lines) + "\n")
    (d / "INCAR").write_text((reference_dir / "INCAR").read_text().replace(
        f"SYSTEM = tight path P={pres[1:]} k4", f"SYSTEM = thermal valley {name}"))
    for f in ("KPOINTS", "POTCAR"):
        shutil.copy(reference_dir / f, d / f)
    jobs[name] = {"pressure": pres, "max_disp_A": float(np.linalg.norm(u.reshape(-1, 3), axis=1).max()),
                  "rms_disp_A": float(np.sqrt(np.mean(np.sum(u.reshape(-1, 3) ** 2, axis=1)))),
                  "POSCAR_sha256": file_sha256(d / "POSCAR")}


jobs = {}
for pres in ("P4", "P5"):
    for Tk in (0, 150, 300):
        u = zg(Tk)
        write_job(f"zg_all_T{Tk}_{pres}_p", pres, u)
        write_job(f"zg_all_T{Tk}_{pres}_m", pres, -u)
    for cls in ("bend", "stretch"):
        u = zg(300, cls)
        write_job(f"zg_{cls}_T300_{pres}_p", pres, u)
        write_job(f"zg_{cls}_T300_{pres}_m", pres, -u)
for n in range(1, 5):
    u = mc(300)
    write_job(f"mc{n}_T300_P5_p", "P5", u)
    write_job(f"mc{n}_T300_P5_m", "P5", -u)
for sc in (0.9, 1.1):
    u = zg(300, scale=sc)
    write_job(f"amp{sc}_T300_P5_p", "P5", u)
    write_job(f"amp{sc}_T300_P5_m", "P5", -u)

summary = {"n_modes_positive": len(modes), "n_excluded_w2_le_0": len(neg),
           "n_bend": sum(x["cls"] == "bend" for x in modes),
           "n_stretch": sum(x["cls"] == "stretch" for x in modes), "n_mixed": sum(x["cls"] == "mixed" for x in modes),
           "lowest_freq_cm1": [float(np.sqrt(x["w2"] * EV / (ANGSTROM ** 2 * AMU)) / (2 * np.pi * 2.99792458e10))
                               for x in modes[:6]]}
(OUT_DIR / "modes_thermal.json").write_text(json.dumps(
    {**summary, "modes": [{k: v for k, v in x.items() if k != "u"} for x in modes]}, indent=1))
hashes = {"forces_dir_files": len(list((FORCE_BALANCE_DIR / "forces").glob("*.npy"))),
          "H_sha256": hashlib.sha256(H.tobytes()).hexdigest(),
          "ref_POSCAR": {p: file_sha256(INPUT_DIR / p.lower() / "POSCAR") for p in ("P4", "P5")},
          "POTCAR": file_sha256(INPUT_DIR / "p5" / "POTCAR"),
          "script_sha256": file_sha256(Path(__file__)), "jobs": jobs, "summary": summary}
(OUT_DIR / "thermal_input_hashes.json").write_text(json.dumps(hashes, indent=1))
print(json.dumps(summary, indent=1))
print({k: (round(v["max_disp_A"], 3), round(v["rms_disp_A"], 4)) for k, v in jobs.items() if k.endswith("_p")})
