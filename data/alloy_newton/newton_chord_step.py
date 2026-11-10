"""Second (chord) Newton step of force balance for the alloy accommodation.
r1 = F_aff + u_pred; f1 = DFT force at r1 (Hessian settings); u2 = H+ Q f1 with the same H at F_aff; r2 = r1 + u2.
Readout static at r2 (path-split settings) and force static at r2. Lambda(r1) is taken from the alloy-segment runs
<iv>/end.
Flags: N0 |Qf1| < |Qf0|; N1 |Q(u_cum - u_act)| decreases; N2 switching edge: S11->S12 |L(r2)-L(S_b)| <= 2.5 meV,
S05->S06 <= 1 meV.
Needs the path-split statics and the alloy-segment runs of the cluster (/data/...) and the VASP outputs of runs/: not
deposited; newton_result.json is the result.
usage: python newton_chord_step.py prepare|stage2|eval
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
ALLOY_LEVERAGE_DIR = Path(__file__).resolve().parents[1] / "alloy_leverage"
ALLOY_SEGMENT_DIR = Path("$WORK/spinel/ALLOY_SEGMENT_20261026")
sys.path.insert(0, str(ALLOY_LEVERAGE_DIR))
from alloy_leverage import (INTERVALS, PATH_SPLIT_DIR, FORCE_BALANCE_DIR, geometry, write_job, hess_incar,
                            read_nelect, read_forces, margins)

OUTPUT_DIR = Path(__file__).resolve().parent


def solver(iv, N):
    H = np.load(ALLOY_LEVERAGE_DIR / "runs" / iv / "H.npy")
    T = np.zeros((3 * N, 3))
    for a in range(3):
        T[a::3, a] = 1 / np.sqrt(N)
    Q = np.eye(3 * N) - T @ T.T
    w, V = np.linalg.eigh(Q @ H @ Q)
    keep = np.argsort(np.abs(w))[3:]
    return Q, V[:, keep], w[keep]


def write_job_case(d, kind, geom, src, ne, cart, title, hashes, iv):
    inv = np.linalg.inv(geom["L"])
    if kind == "force":
        write_job(d, hess_incar(ne), (FORCE_BALANCE_DIR / "KPOINTS").read_text(), src / "POTCAR", geom["L"],
                  geom["sp"], geom["n"], geom["fF"] + cart.reshape(-1, 3) @ inv, title)
    else:
        write_job(d, (src / "INCAR").read_text(), (src / "KPOINTS").read_text(), src / "POTCAR", geom["L"],
                  geom["sp"], geom["n"], geom["fF"] + cart.reshape(-1, 3) @ inv, title)
    hashes[f"{iv}/{d.name}"] = hashlib.sha256((d / "POSCAR").read_bytes()).hexdigest()


def prepare():
    hashes = {}
    jobs = []
    for iv, (step, _, _) in INTERVALS.items():
        geom = geometry(iv)
        src = PATH_SPLIT_DIR / step / "aff"
        ne = read_nelect(src / "OUTCAR")
        u1 = np.load(ALLOY_LEVERAGE_DIR / "runs" / iv / "u_pred.npy")
        write_job_case(OUTPUT_DIR / "runs" / iv / "f1", "force", geom, src, ne, u1, f"{iv} newton f1", hashes, iv)
        jobs.append(f"runs/{iv}/f1")
    (OUTPUT_DIR / "alloy_newton_input_hashes.json").write_text(json.dumps(hashes, indent=1))
    (OUTPUT_DIR / "jobs_phase1.txt").write_text("\n".join(jobs) + "\n")


def stage2():
    hashes = json.loads((OUTPUT_DIR / "alloy_newton_input_hashes.json").read_text())
    jobs = []
    for iv, (step, _, _) in INTERVALS.items():
        geom = geometry(iv)
        N = geom["N"]
        src = PATH_SPLIT_DIR / step / "aff"
        ne = read_nelect(src / "OUTCAR")
        run_dir = OUTPUT_DIR / "runs" / iv
        Q, V, w = solver(iv, N)
        u1 = np.load(ALLOY_LEVERAGE_DIR / "runs" / iv / "u_pred.npy")
        f1 = read_forces(run_dir / "f1", N)
        u2 = V @ ((V.T @ (Q @ f1)) / w)
        np.save(run_dir / "f1.npy", f1)
        np.save(run_dir / "u2.npy", u2)
        for tag, kind in (("rd2", "readout"), ("f2", "force")):
            write_job_case(run_dir / tag, kind, geom, src, ne, u1 + u2, f"{iv} newton {tag}", hashes, iv)
            jobs.append(f"runs/{iv}/{tag}")
        print(iv, "|Qf1| %.4f  |u2| %.4f" % (np.linalg.norm(Q @ f1), np.linalg.norm(u2)))
    (OUTPUT_DIR / "alloy_newton_input_hashes.json").write_text(json.dumps(hashes, indent=1))
    (OUTPUT_DIR / "jobs_phase2.txt").write_text("\n".join(jobs) + "\n")


def evaluate():
    leverage_result = json.loads((ALLOY_LEVERAGE_DIR / "al_result.json").read_text())
    results = {}
    for iv, (step, end, edge) in INTERVALS.items():
        geom = geometry(iv)
        N = geom["N"]
        run_dir = OUTPUT_DIR / "runs" / iv
        leverage_iv = leverage_result[iv]
        Q, V, w = solver(iv, N)
        kV, kC = np.array(leverage_iv["k_competitor"]["V"]), np.array(leverage_iv["k_competitor"]["C"])
        u1 = np.load(ALLOY_LEVERAGE_DIR / "runs" / iv / "u_pred.npy")
        u2 = np.load(run_dir / "u2.npy")
        ua = geom["u_act"]
        f0 = np.load(ALLOY_LEVERAGE_DIR / "runs" / iv / "f.npy")
        f1 = np.load(run_dir / "f1.npy")
        f2 = read_forces(run_dir / "f2", N)
        L0 = leverage_iv["margins"]["00"]
        Lb = leverage_iv["margins"]["11"]
        L1 = margins(ALLOY_SEGMENT_DIR / "runs" / iv / "end", kV, kC)
        L2 = margins(run_dir / "rd2", kV, kC)

        def cosine(a, b):
            return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))

        Qa = Q @ ua
        result = {"edge": edge, "norm_Qf0": float(np.linalg.norm(Q @ f0)), "norm_Qf1": float(np.linalg.norm(Q @ f1)),
                  "norm_Qf2": float(np.linalg.norm(Q @ f2)),
                  "dist_r1_Sb": float(np.linalg.norm(Q @ u1 - Qa)),
                  "dist_r2_Sb": float(np.linalg.norm(Q @ (u1 + u2) - Qa)),
                  "cos_1": cosine(u1, Qa), "cos_2": cosine(u1 + u2, Qa),
                  "ratio_1": float(np.linalg.norm(u1) / np.linalg.norm(Qa)),
                  "ratio_2": float(np.linalg.norm(u1 + u2) / np.linalg.norm(Qa)), "norm_u2": float(np.linalg.norm(u2)),
                  "dE_r1_r2_meV": 1e3 * (L2["TOTEN"] - L1["TOTEN"])}
        for band in ("V", "C"):
            result[band] = {"L_aff": L0[band], "L_r1": L1[band], "L_r2": L2[band], "L_Sb": Lb[band],
                            "err_r1": L1[band] - Lb[band], "err_r2": L2[band] - Lb[band],
                            "anharmonic": L2[band] - L1[band]}
        switching = result[edge]
        tol = 2.5 if iv == "i1112" else 1.0
        result["N0"] = bool(result["norm_Qf1"] < result["norm_Qf0"])
        result["N1"] = bool(result["dist_r2_Sb"] < result["dist_r1_Sb"])
        result["N2"] = bool(abs(switching["err_r2"]) <= tol)
        results[iv] = result
    (OUTPUT_DIR / "newton_result.json").write_text(json.dumps(results, indent=1, default=float))
    print(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    {"prepare": prepare, "stage2": stage2, "eval": evaluate}[sys.argv[1]]()
