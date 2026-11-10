"""(001) biaxial strain of k12 and S11 (56-atom cells).
prepare : c-scan jobs. In-plane x,y components of the lattice scaled by (1+e), z components by
          s = (1-0.86e)(1+d), d in {-1,0,+1}%; ISIF 2 relaxations (Gamma 3x3x3). e = 0 gets only a final job (s = 1).
stage2  : quadratic fit E(s) -> s*; if s* is outside the scanned range, one extra point beyond the edge (once); else
          final jobs: ISIF 2 relaxation at s* (from the nearest scan point), then a readout static (path-split INCAR,
          ISYM 0, Gamma 4x4x4).
stage3  : global margins; for each system the first Lambda_C sign change next to e = 0 defines e1 -> e2
          (|e1| < |e2|); mechanism jobs: affine (e1 relaxed fractions in the e2 cell) force static (force-constant
          settings) and readout, states 10 and 01.
eval    : Lambda(e), c*/c0, elastic energy per atom, e_c (linear interpolation), enrichment (enrichment_all.py
          share) and phi.
Reads the path-split statics and the tight-spinel statics on the cluster ($WORK/...) and the VASP outputs of its own
runs: not deposited; strain_result.json, fits.json and brackets.json are the results.
usage: python strain.py prepare|stage2|stage3|eval
"""
import hashlib, json, re, shutil, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "four_state_leverage" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "alloy_leverage"))
from lev_common import read_poscar, write_poscar, read_eig, kernel_rows, split_b
from alloy_leverage import hess_incar, FORCE_BALANCE_DIR

OUTPUT_DIR = Path(__file__).resolve().parent
PATH_SPLIT_DIR = Path("$WORK/PRL_MECHANISM_TEST_20261014/Path_Split/work/comp")
SOURCE_DIRS = {"k12": Path("$WORK/PRL_TIGHT_SPINEL_20261015/static_k4/P0"),
               "S11": PATH_SPLIT_DIR / "S11" / "end"}
READOUT_INCAR = (PATH_SPLIT_DIR / "S11" / "end" / "INCAR").read_text()
READOUT_KPOINTS = (PATH_SPLIT_DIR / "S11" / "end" / "KPOINTS").read_text()
STRAINS = [-3, -2, -1, 1, 2, 3]
KAPPA = 0.86
RELAX_INCAR = """SYSTEM = epitaxial strain relax
PREC = Accurate
ENCUT = 520
EDIFF = 1E-6
ISMEAR = 0
SIGMA = 0.05
ISPIN = 1
LREAL = Auto
LASPH = .TRUE.
ADDGRID = .TRUE.
ALGO = Normal
NELM = 200
IBRION = 2
ISIF = 2
NSW = 200
EDIFFG = -0.01
LWAVE = .FALSE.
LCHARG = .FALSE.
"""
RELAX_KPOINTS = "Gamma\n0\nGamma\n3 3 3\n0 0 0\n"


def base_structure(system_name):
    L, sp, n, fr = read_poscar(SOURCE_DIRS[system_name]
                               / ("CONTCAR" if (SOURCE_DIRS[system_name] / "CONTCAR").exists() else "POSCAR"))
    return L, sp, n, fr


def strained(L, e, s):
    M = np.diag([1 + e / 100, 1 + e / 100, s])
    return L @ M


def job(d, L, sp, n, fr, incar, kp, title, hashes, chain=False):
    d.mkdir(parents=True, exist_ok=True)
    write_poscar(d / "POSCAR", title, L, sp, n, fr)
    (d / "INCAR").write_text(incar)
    (d / "KPOINTS").write_text(kp)
    shutil.copy(SOURCE_POTCAR, d / "POTCAR")
    if chain:
        (d / "CHAIN").write_text("readout\n")
    hashes[str(d.relative_to(OUTPUT_DIR))] = hashlib.sha256((d / "POSCAR").read_bytes()).hexdigest()


def energy(d):
    return float(re.findall(r"free  energy   TOTEN  =\s+(\S+)", (d / "OUTCAR").read_text())[-1])


def hashes_load():
    p = OUTPUT_DIR / "strain_input_hashes.json"
    return json.loads(p.read_text()) if p.exists() else {}


def hashes_save(input_hashes):
    (OUTPUT_DIR / "strain_input_hashes.json").write_text(json.dumps(input_hashes, indent=1))


def scan_s(e):
    return [(1 - KAPPA * e / 100) * (1 + d / 100) for d in (-1, 0, 1)]


def prepare():
    global SOURCE_POTCAR
    input_hashes = hashes_load()
    jobs = []
    for system_name in SOURCE_DIRS:
        SOURCE_POTCAR = SOURCE_DIRS[system_name] / "POTCAR"
        L, sp, n, fr = base_structure(system_name)
        for e in STRAINS:
            for i, s in enumerate(scan_s(e)):
                d = OUTPUT_DIR / "runs" / system_name / f"e{e:+d}" / f"scan{i}"
                job(d, strained(L, e, s), sp, n, fr, RELAX_INCAR, RELAX_KPOINTS, f"{system_name} e{e} s{s:.5f}",
                    input_hashes)
                (d / "S").write_text(f"{s:.8f}\n")
                jobs.append(str(d.relative_to(OUTPUT_DIR)))
    hashes_save(input_hashes)
    (OUTPUT_DIR / "jobs_scan.txt").write_text("\n".join(jobs) + "\n")
    print(len(jobs), "scan jobs")


def stage2():
    global SOURCE_POTCAR
    input_hashes = hashes_load()
    extra, finals, fits = [], [], {}
    for system_name in SOURCE_DIRS:
        SOURCE_POTCAR = SOURCE_DIRS[system_name] / "POTCAR"
        L, sp, n, fr = base_structure(system_name)
        fits[system_name] = {}
        for e in [0] + STRAINS:
            strain_dir = OUTPUT_DIR / "runs" / system_name / f"e{e:+d}"
            if e == 0:
                s_star, start = 1.0, fr
            else:
                pts = sorted([(float((d / "S").read_text()), energy(d), d) for d in strain_dir.glob("scan*")
                              if (d / "DONE").exists()])
                ss = np.array([p[0] for p in pts])
                ee = np.array([p[1] for p in pts])
                a, b, c = np.polyfit(ss, ee, 2)
                s_star = -b / (2 * a) if a > 0 else ss[np.argmin(ee)]
                fits[system_name][e] = {"s": ss.tolist(), "E": ee.tolist(), "s_star": float(s_star), "curv": float(a)}
                if not (ss.min() <= s_star <= ss.max()):
                    if len(pts) < 4:
                        s_new = ss.max() * 1.01 if s_star > ss.max() else ss.min() / 1.01
                        d = strain_dir / "scan3"
                        job(d, strained(L, e, s_new), sp, n, fr, RELAX_INCAR, RELAX_KPOINTS,
                            f"{system_name} e{e} s{s_new:.5f}", input_hashes)
                        (d / "S").write_text(f"{s_new:.8f}\n")
                        extra.append(str(d.relative_to(OUTPUT_DIR)))
                        continue
                    s_star = float(np.clip(s_star, ss.min(), ss.max()))
                    fits[system_name][e]["clipped"] = True
                near = min(pts, key=lambda p: abs(p[0] - s_star))[2]
                _, _, _, start = read_poscar(near / "CONTCAR")
            d = strain_dir / "final"
            job(d, strained(L, e, s_star), sp, n, start, RELAX_INCAR, RELAX_KPOINTS,
                f"{system_name} e{e} final s{s_star:.5f}", input_hashes, chain=True)
            (d / "S").write_text(f"{s_star:.8f}\n")
            finals.append(str(d.relative_to(OUTPUT_DIR)))
    hashes_save(input_hashes)
    (OUTPUT_DIR / "fits.json").write_text(json.dumps(fits, indent=1))
    if extra:
        (OUTPUT_DIR / "jobs_extra.txt").write_text("\n".join(extra) + "\n")
        print("extra", extra)
        sys.exit(3)
    (OUTPUT_DIR / "jobs_final.txt").write_text("\n".join(finals) + "\n")
    print(len(finals), "final jobs")


def margins(d):
    K, V, C, toten = read_eig(d)
    gamma = np.flatnonzero(np.all(np.abs(K) < 1e-6, axis=1))[0]
    non_gamma = np.ones(len(K), bool)
    non_gamma[gamma] = False
    return {"V": 1e3 * (V[gamma] - V[non_gamma].max()), "C": 1e3 * (C[non_gamma].min() - C[gamma]), "TOTEN": toten,
            "kV": K[non_gamma][np.argmax(V[non_gamma])].tolist(), "kC": K[non_gamma][np.argmin(C[non_gamma])].tolist()}


def sequence(system_name):
    out = {}
    for e in [0] + STRAINS:
        d = OUTPUT_DIR / "runs" / system_name / f"e{e:+d}" / "final"
        if (d / "readout" / "DONE").exists():
            out[e] = margins(d / "readout")
    return out


def bracket(seq):
    for side in ([1, 2, 3], [-1, -2, -3]):
        prev = 0
        for e in side:
            if e in seq and prev in seq and np.sign(seq[e]["C"]) != np.sign(seq[prev]["C"]):
                return prev, e
            prev = e
    return None


def stage3():
    global SOURCE_POTCAR
    input_hashes = hashes_load()
    jobs = []
    brackets = {}
    for system_name in SOURCE_DIRS:
        SOURCE_POTCAR = SOURCE_DIRS[system_name] / "POTCAR"
        seq = sequence(system_name)
        crossing = bracket(seq)
        brackets[system_name] = crossing
        if crossing is None:
            continue
        e1, e2 = crossing
        d1 = OUTPUT_DIR / "runs" / system_name / f"e{e1:+d}" / "final"
        d2 = OUTPUT_DIR / "runs" / system_name / f"e{e2:+d}" / "final"
        _, sp, n, f1 = read_poscar(d1 / "CONTCAR")
        L2, _, _, f2 = read_poscar(d2 / "CONTCAR")
        N = sum(n)
        spec = np.repeat(sp, n)
        du = f2 - f1
        du -= np.round(du)
        u = (du @ L2).ravel()
        b_sites = [i for i in range(N) if spec[i] in ("Sc", "Sb")]
        X = [i for i in range(N) if spec[i] == "Se"]
        bonds = []
        for i in b_sites:
            dd = f1[X] - f1[i]
            img = -np.round(dd)
            r = np.linalg.norm((dd + img) @ L2, axis=1)
            for k in np.argsort(r)[:6]:
                bonds.append((i, X[k], img[k].astype(int).tolist()))
        net = np.array(sorted(b_sites + X))
        Rb = kernel_rows(f1, L2, bonds, net)
        ub = split_b(u, Rb, net, N)
        inv = np.linalg.inv(L2)
        mech_dir = OUTPUT_DIR / "runs" / system_name / "mech"
        mech_dir.mkdir(parents=True, exist_ok=True)
        np.save(mech_dir / "u_act.npy", u)
        np.save(mech_dir / "u_b.npy", ub)
        (mech_dir / "BONDS.json").write_text(json.dumps({"bonds": bonds, "net": net.tolist(), "e1": e1, "e2": e2}))
        ne = int(round(float(re.search(r"NELECT =\s+(\S+)", (d2 / "OUTCAR").read_text()).group(1))))
        job(mech_dir / "aff_force", L2, sp, n, f1, hess_incar(ne), (FORCE_BALANCE_DIR / "KPOINTS").read_text(),
            f"{system_name} aff force", input_hashes)
        for tag, v in (("s00", 0 * u), ("s10", ub), ("s01", u - ub)):
            job(mech_dir / tag, L2, sp, n, f1 + v.reshape(-1, 3) @ inv, READOUT_INCAR, READOUT_KPOINTS,
                f"{system_name} {tag}", input_hashes)
        jobs += [str((mech_dir / t).relative_to(OUTPUT_DIR)) for t in ("aff_force", "s00", "s10", "s01")]
    hashes_save(input_hashes)
    (OUTPUT_DIR / "brackets.json").write_text(json.dumps(brackets))
    (OUTPUT_DIR / "jobs_mech.txt").write_text("\n".join(jobs) + ("\n" if jobs else ""))
    print("brackets", brackets, len(jobs), "mech jobs")


def kernel(f_start, L, bonds, net):  # as in bending_enrichment/enrichment_all.py
    col = {int(a): c for c, a in enumerate(net)}
    vec = np.array([(f_start[j] - f_start[i] + np.array(img)) @ L for i, j, img in bonds])
    J = np.zeros((len(bonds), 3 * len(net)))
    for r, (i, j, _) in enumerate(bonds):
        e = vec[r] / np.linalg.norm(vec[r])
        J[r, 3 * col[i]:3 * col[i] + 3] = -e
        J[r, 3 * col[j]:3 * col[j] + 3] = e
    _, s, Vt = np.linalg.svd(J, full_matrices=False)
    return Vt[: int(np.sum(s > s[0] * 1e-10))], J


def share(u, B, net):  # as in bending_enrichment/enrichment_all.py
    v = u.reshape(-1, 3)[net]
    v = v - v.mean(axis=0)
    v = v.ravel()
    p = v - B.T @ (B @ v)
    return float(p @ p / (v @ v))


def read_forces(d, N):
    force_lines = (d / "OUTCAR").read_text().split("TOTAL-FORCE (eV/Angst)")[-1].splitlines()[2:2 + N]
    return np.array([[float(x) for x in line.split()[3:6]] for line in force_lines]).ravel()


def evaluate():
    results = {}
    brackets = json.loads((OUTPUT_DIR / "brackets.json").read_text()) if (OUTPUT_DIR / "brackets.json").exists() else {}
    fits = json.loads((OUTPUT_DIR / "fits.json").read_text())
    for system_name in SOURCE_DIRS:
        seq = sequence(system_name)
        n_atoms = None
        result = {"points": {}}
        for e, m in sorted(seq.items()):
            d = OUTPUT_DIR / "runs" / system_name / f"e{e:+d}" / "final"
            n_atoms = n_atoms or sum(read_poscar(d / "CONTCAR")[2])
            result["points"][e] = {**m, "s_star": float((d / "S").read_text()), "E_relax_eV": energy(d)}
        E0 = result["points"][0]["E_relax_eV"]
        for e in result["points"]:
            result["points"][e]["elastic_meV_atom"] = 1e3 * (result["points"][e]["E_relax_eV"] - E0) / n_atoms
        result["fits"] = fits.get(system_name, {})
        crossing = brackets.get(system_name)
        if crossing:
            e1, e2 = crossing
            c1, c2 = seq[e1]["C"], seq[e2]["C"]
            result["e_c"] = e1 + (e2 - e1) * c1 / (c1 - c2)
            mech_dir = OUTPUT_DIR / "runs" / system_name / "mech"
            bonds_data = json.loads((mech_dir / "BONDS.json").read_text())
            d2 = OUTPUT_DIR / "runs" / system_name / f"e{e2:+d}" / "final"
            L2, sp, n, _ = read_poscar(d2 / "CONTCAR")
            _, _, _, f1 = read_poscar(mech_dir / "s00" / "POSCAR")
            B, _ = kernel(f1, L2, bonds_data["bonds"], np.array(bonds_data["net"]))
            N = sum(n)
            f = read_forces(mech_dir / "aff_force", N)
            u = np.load(mech_dir / "u_act.npy")
            drive_share, actual_share = (share(f, B, np.array(bonds_data["net"])),
                                         share(u, B, np.array(bonds_data["net"])))
            states_C = {t: margins(mech_dir / t)["C"] for t in ("s00", "s10", "s01")}
            states_C["s11"] = seq[e2]["C"]
            tot = states_C["s11"] - states_C["s00"]
            bend = 0.5 * ((states_C["s10"] - states_C["s00"]) + (states_C["s11"] - states_C["s01"]))
            result["mech"] = {"e1": e1, "e2": e2, "drive_share": drive_share, "actual_share": actual_share,
                              "enrichment_actual": actual_share / drive_share, "states_C": states_C,
                              "dL_C_total": tot, "phi": bend / tot,
                              "additivity_resid": states_C["s11"] - states_C["s10"] - states_C["s01"] + states_C["s00"],
                              "S2": bool(actual_share / drive_share >= 9 and bend / tot >= 0.5)}
            result["S1_accessible"] = bool(abs(result["e_c"]) <= 2)
        results[system_name] = result
    (OUTPUT_DIR / "strain_result.json").write_text(json.dumps(results, indent=1, default=float))
    print(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    {"prepare": prepare, "stage2": stage2, "stage3": stage3, "eval": evaluate}[sys.argv[1]]()
