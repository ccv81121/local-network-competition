"""Force balance for the alloy accommodation: u = H+ f_swap at the path-split affine state F (species of S_b,
fractional coordinates of S_a in the S_b lattice) predicts the internal relaxation F -> S_b.
prepare : Hessian inputs (F +- 0.01 A along each of the 168 Cartesian coordinates, force-balance reference settings,
          NBANDS = NELECT/2 + 16), and the measured four states 10 = F + P_b u_act, 01 = F + (1 - P_b) u_act
          (readout settings).
stage2  : H, f, u_pred, u_b, u_c; finite-difference inputs F +- 0.25|u_i| d_i (readout settings).
eval    : L1 cos(u_pred, u_act); L2 dLambda_int predicted vs Lambda(S_b) - Lambda(F); L3 phi_pred vs four-state phi.
Readout: path-split statics (ISYM 0, Gamma 4x4x4); competing k fixed at that of the end structure.
Network: Sc, Sb and Se; bonds = six nearest Se of each Sc/Sb (minimum image); projector as
lev_common.kernel_rows/split_b.
Reads the path-split statics and force-balance reference on the cluster ($WORK/...) and writes runs/ there: not
deposited; runs/<iv>/{H,f}.npy are the deposited results.
usage: python alloy_leverage.py prepare|stage2|eval
"""
import hashlib, json, re, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "four_state_leverage" / "scripts"))
from lev_common import read_poscar, write_poscar, kernel_rows, split_b, read_eig

OUTPUT_DIR = Path(__file__).resolve().parent
PATH_SPLIT_DIR = Path("$WORK/PRL_MECHANISM_TEST_20261014/Path_Split/work/comp")
FORCE_BALANCE_DIR = Path("$WORK/PRL_FORCEBALANCE_20261007/cases/ref")
INTERVALS = {"i0506": ("step_05_06", "S06", "V"), "i1112": ("step_11_12", "S12", "C")}
FD_STEP = 0.01


def hess_incar(n_electrons):
    t = (FORCE_BALANCE_DIR / "INCAR").read_text()
    t = re.sub(r"NBANDS = \d+", f"NBANDS = {n_electrons // 2 + 16}", t)
    return t.replace("LWAVE = .TRUE.", "LWAVE = .FALSE.").replace("SYSTEM = force-balance finite difference",
                                                                  "SYSTEM = alloy-leverage Hessian")


def write_job(d, src_incar_text, kpoints_text, potcar, lat, sp, n, fr, title):
    d.mkdir(parents=True, exist_ok=True)
    write_poscar(d / "POSCAR", title, lat, sp, n, fr)
    (d / "INCAR").write_text(src_incar_text)
    (d / "KPOINTS").write_text(kpoints_text)
    shutil.copy(potcar, d / "POTCAR")


def geometry(iv):
    step, end, _ = INTERVALS[iv]
    L, sp, n, fF = read_poscar(PATH_SPLIT_DIR / step / "aff" / "POSCAR")
    L2, sp2, n2, fE = read_poscar(PATH_SPLIT_DIR / end / "end" / "POSCAR")
    assert sp == sp2 and n == n2 and np.allclose(L, L2, atol=1e-6)
    spec = np.repeat(sp, n)
    N = len(spec)
    atom_map = []
    for i in range(N):
        d = fE - fF[i]
        d -= np.round(d)
        r = np.linalg.norm(d @ L, axis=1)
        r[spec != spec[i]] = 9
        j = int(np.argmin(r))
        assert r[j] < 0.4, (iv, i, r[j])
        atom_map.append(j)
    assert len(set(atom_map)) == N
    du = fE[atom_map] - fF
    du -= np.round(du)
    u_act = (du @ L).ravel()
    B = [i for i in range(N) if spec[i] in ("Sc", "Sb")]
    X = [i for i in range(N) if spec[i] == "Se"]
    bonds = []
    for i in B:
        d = fF[X] - fF[i]
        img = -np.round(d)
        r = np.linalg.norm((d + img) @ L, axis=1)
        for k in np.argsort(r)[:6]:
            bonds.append((i, X[k], img[k].astype(int).tolist()))
    net = np.array(sorted(B + X))
    Rb = kernel_rows(fF, L, bonds, net)
    return dict(L=L, sp=sp, n=n, fF=fF, spec=spec, N=N, u_act=u_act, net=net, Rb=Rb, bonds=bonds, map=atom_map)


def read_nelect(outcar):
    return int(round(float(re.search(r"NELECT =\s+(\S+)", Path(outcar).read_text()).group(1))))


def prepare():
    hashes = {}
    jobs = []
    for iv, (step, end, _) in INTERVALS.items():
        geom = geometry(iv)
        src = PATH_SPLIT_DIR / step / "aff"
        ne = read_nelect(src / "OUTCAR")
        run_dir = OUTPUT_DIR / "runs" / iv
        kp3 = (FORCE_BALANCE_DIR / "KPOINTS").read_text()
        kp4 = (src / "KPOINTS").read_text()
        readout_incar = (src / "INCAR").read_text()
        inv = np.linalg.inv(geom["L"])
        cases = {"ref": np.zeros(3 * geom["N"])}
        for i in range(geom["N"]):
            for a, ax in enumerate("xyz"):
                for s, t in ((1, "p"), (-1, "m")):
                    v = np.zeros(3 * geom["N"])
                    v[3 * i + a] = s * FD_STEP
                    cases[f"d{i:02d}{ax}{t}"] = v
        for c, v in cases.items():
            d = run_dir / "hessian" / c
            write_job(d, hess_incar(ne), kp3, src / "POTCAR", geom["L"], geom["sp"], geom["n"],
                      geom["fF"] + v.reshape(-1, 3) @ inv, f"{iv} hessian {c}")
            hashes[f"{iv}/hessian/{c}"] = hashlib.sha256((d / "POSCAR").read_bytes()).hexdigest()
            jobs.append(f"runs/{iv}/hessian/{c}")
        ub = split_b(geom["u_act"], geom["Rb"], geom["net"], geom["N"])
        uc = geom["u_act"] - ub
        for tag, v in (("fs10", ub), ("fs01", uc)):
            d = run_dir / tag
            write_job(d, readout_incar, kp4, src / "POTCAR", geom["L"], geom["sp"], geom["n"],
                      geom["fF"] + v.reshape(-1, 3) @ inv, f"{iv} fourstate {tag}")
            hashes[f"{iv}/{tag}"] = hashlib.sha256((d / "POSCAR").read_bytes()).hexdigest()
            jobs.append(f"runs/{iv}/{tag}")
        np.save(run_dir / "u_act.npy", geom["u_act"])
        np.save(run_dir / "u_act_b.npy", ub)
        (run_dir / "geom.json").write_text(json.dumps(
            {"N": geom["N"], "nelect": ne, "nbands": ne // 2 + 16, "n_bonds": len(geom["bonds"]),
             "kernel_dim": int(geom["Rb"].shape[0]), "net": geom["net"].tolist(), "map_F_to_end": geom["map"],
             "u_act_norm_A": float(np.linalg.norm(geom["u_act"])),
             "u_act_b_share": float(ub @ ub / (geom["u_act"] @ geom["u_act"]))}, indent=1))
        print(iv, "N", geom["N"], "nelect", ne, "bonds", len(geom["bonds"]), "kernel", geom["Rb"].shape[0],
              "|u_act| %.4f" % np.linalg.norm(geom["u_act"]),
              "bend share %.3f" % (ub @ ub / (geom["u_act"] @ geom["u_act"])))
    (OUTPUT_DIR / "alloy_leverage_input_hashes.json").write_text(json.dumps(hashes, indent=1))
    (OUTPUT_DIR / "jobs_phase1.txt").write_text("\n".join(jobs) + "\n")


def read_forces(d, N):
    force_lines = Path(d / "OUTCAR").read_text().split("TOTAL-FORCE (eV/Angst)")[-1].splitlines()[2:2 + N]
    return np.array([[float(x) for x in line.split()[3:6]] for line in force_lines]).ravel()


def stage2():
    jobs = []
    for iv, (step, end, _) in INTERVALS.items():
        geom = geometry(iv)
        run_dir = OUTPUT_DIR / "runs" / iv
        N = geom["N"]
        src = PATH_SPLIT_DIR / step / "aff"
        inv = np.linalg.inv(geom["L"])
        H = np.zeros((3 * N, 3 * N))
        for i in range(N):
            for a, ax in enumerate("xyz"):
                H[3 * i + a] = -(read_forces(run_dir / "hessian" / f"d{i:02d}{ax}p", N)
                                 - read_forces(run_dir / "hessian" / f"d{i:02d}{ax}m", N)) / (2 * FD_STEP)
        asym = float(np.abs(H - H.T).max())
        H = 0.5 * (H + H.T)
        T = np.zeros((3 * N, 3))
        for a in range(3):
            T[a::3, a] = 1 / np.sqrt(N)
        Q = np.eye(3 * N) - T @ T.T
        w, V = np.linalg.eigh(Q @ H @ Q)
        keep = np.argsort(np.abs(w))[3:]
        w, V = w[keep], V[:, keep]
        f = read_forces(run_dir / "hessian" / "ref", N)
        u = V @ ((V.T @ (Q @ f)) / w)
        ub = split_b(u, geom["Rb"], geom["net"], N)
        uc = u - ub
        np.save(run_dir / "H.npy", H)
        np.save(run_dir / "f.npy", f)
        np.save(run_dir / "u_pred.npy", u)
        np.save(run_dir / "u_b.npy", ub)
        np.save(run_dir / "u_c.npy", uc)
        meta = {"H_asym_max": asym, "n_negative_modes": int((w < 0).sum()), "min_eig": float(w.min()),
                "f_max": float(np.abs(f.reshape(-1, 3)).max()),
                "u_pred_norm": float(np.linalg.norm(u)), "u_b_norm": float(np.linalg.norm(ub)),
                "u_c_norm": float(np.linalg.norm(uc)),
                "harmonic_dE_meV": float(-0.5 * (Q @ f) @ u * 1e3)}
        for tag, v in (("b", ub), ("c", uc)):
            nv = np.linalg.norm(v)
            delta = 0.25 * nv
            meta[f"delta_{tag}"] = float(delta)
            for s, t in ((1, "p"), (-1, "m")):
                d = run_dir / f"fd_{tag}{t}"
                write_job(d, (src / "INCAR").read_text(), (src / "KPOINTS").read_text(), src / "POTCAR", geom["L"],
                          geom["sp"], geom["n"], geom["fF"] + (s * delta * v / nv).reshape(-1, 3) @ inv,
                          f"{iv} fd {tag}{t}")
                jobs.append(f"runs/{iv}/fd_{tag}{t}")
        (run_dir / "PRED.json").write_text(json.dumps(meta, indent=1))
        print(iv, json.dumps(meta))
    (OUTPUT_DIR / "jobs_phase2.txt").write_text("\n".join(jobs) + "\n")


def margins(d, kV, kC):
    K, Vb, Cb, en = read_eig(d)
    gamma = np.flatnonzero(np.all(np.abs(K) < 1e-6, axis=1))[0]
    iV = int(np.argmin(np.abs(K - kV).sum(1)))
    iC = int(np.argmin(np.abs(K - kC).sum(1)))
    return {"V": 1e3 * (Vb[gamma] - Vb[iV]), "C": 1e3 * (Cb[iC] - Cb[gamma]), "TOTEN": en}


def evaluate():
    results = {}
    for iv, (step, end, edge) in INTERVALS.items():
        geom = geometry(iv)
        run_dir = OUTPUT_DIR / "runs" / iv
        N = geom["N"]
        pred = json.loads((run_dir / "PRED.json").read_text())
        K, Vb, Cb, _ = read_eig(PATH_SPLIT_DIR / end / "end")
        non_gamma = ~np.all(np.abs(K) < 1e-6, axis=1)
        kV = K[non_gamma][np.argmax(Vb[non_gamma])]
        kC = K[non_gamma][np.argmin(Cb[non_gamma])]
        lam = {"00": margins(PATH_SPLIT_DIR / step / "aff", kV, kC),
               "11": margins(PATH_SPLIT_DIR / end / "end", kV, kC),
               "10": margins(run_dir / "fs10", kV, kC), "01": margins(run_dir / "fs01", kV, kC)}
        fd = {t: margins(run_dir / f"fd_{t}", kV, kC) for t in ("bp", "bm", "cp", "cm")}
        u = np.load(run_dir / "u_pred.npy")
        T = np.zeros((3 * N, 3))
        for a in range(3):
            T[a::3, a] = 1 / np.sqrt(N)
        Q = np.eye(3 * N) - T @ T.T
        ua = Q @ geom["u_act"]
        cos = float(u @ ua / np.linalg.norm(u) / np.linalg.norm(ua))
        result = {"edge": edge, "k_competitor": {"V": kV.tolist(), "C": kC.tolist()}, "margins": lam, "fd": fd,
                  "pred_meta": pred, "L1_cos": cos, "L1": bool(cos >= 0.90),
                  "u_pred_over_u_act": float(np.linalg.norm(u) / np.linalg.norm(ua))}
        for band in ("V", "C"):
            gb = (fd["bp"][band] - fd["bm"][band]) / (2 * pred["delta_b"])
            gc = (fd["cp"][band] - fd["cm"][band]) / (2 * pred["delta_c"])
            pb, pc = gb * pred["u_b_norm"], gc * pred["u_c_norm"]
            meas = lam["11"][band] - lam["00"][band]
            bend = 0.5 * ((lam["10"][band] - lam["00"][band]) + (lam["11"][band] - lam["01"][band]))
            result[band] = {"g_b": gb, "g_c": gc, "dL_b_pred": pb, "dL_c_pred": pc, "dL_int_pred": pb + pc,
                            "dL_int_meas": meas, "phi_pred": pb / (pb + pc), "phi_meas": bend / meas,
                            "additivity_resid": (lam["11"][band] - lam["10"][band] - lam["01"][band]
                                                 + lam["00"][band])}
        switching = result[edge]
        result["L2"] = bool(np.sign(switching["dL_int_pred"]) == np.sign(switching["dL_int_meas"])
                            and abs(switching["dL_int_pred"] - switching["dL_int_meas"])
                            <= max(0.30 * abs(switching["dL_int_meas"]), 3.0))
        result["L3_applies"] = bool(abs(switching["dL_int_meas"]) >= 3.0)
        result["L3"] = bool(result["L3_applies"] and abs(switching["phi_pred"] - switching["phi_meas"]) <= 0.15
                            and (switching["phi_pred"] - 0.5) * (switching["phi_meas"] - 0.5) > 0)
        results[iv] = result
    (OUTPUT_DIR / "al_result.json").write_text(json.dumps(results, indent=1, default=float))
    print(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    {"prepare": prepare, "stage2": stage2, "eval": evaluate}[sys.argv[1]]()
