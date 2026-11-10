"""Supplementary epitaxial-strain calculations.
1. Mechanism split for the compressive S11 crossing -1 -> -2 % (strain.py's bracket() returns only the tensile
   crossing).
   Same construction as strain.stage3, written to runs/S11/mech_neg.
2. 6x6x6 readouts for strain points with |Lambda_C| < 10 meV (S11 -1 % and +3 %); same INCAR as the
   4x4x4 readout, only the mesh changes; written to <final>/readout_k6.
Needs the cluster inputs and run outputs of strain.py (not deposited).
usage: python strain_supp.py prepare | eval
"""
import json, re, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strain as strain_main
from strain import (read_poscar, kernel_rows, split_b, hess_incar, FORCE_BALANCE_DIR, READOUT_INCAR, READOUT_KPOINTS,
                    OUTPUT_DIR)

SYSTEM_NAME, E1, E2 = "S11", -1, -2
K6_READOUTS = [("S11", -1), ("S11", 3)]


def prepare():
    strain_main.SOURCE_POTCAR = strain_main.SOURCE_DIRS[SYSTEM_NAME] / "POTCAR"
    input_hashes = strain_main.hashes_load()
    jobs = []
    d1 = OUTPUT_DIR / "runs" / SYSTEM_NAME / f"e{E1:+d}" / "final"
    d2 = OUTPUT_DIR / "runs" / SYSTEM_NAME / f"e{E2:+d}" / "final"
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
    mech_dir = OUTPUT_DIR / "runs" / SYSTEM_NAME / "mech_neg"
    mech_dir.mkdir(parents=True, exist_ok=True)
    np.save(mech_dir / "u_act.npy", u)
    np.save(mech_dir / "u_b.npy", ub)
    (mech_dir / "BONDS.json").write_text(json.dumps({"bonds": bonds, "net": net.tolist(), "e1": E1, "e2": E2}))
    ne = int(round(float(re.search(r"NELECT =\s+(\S+)", (d2 / "OUTCAR").read_text()).group(1))))
    strain_main.job(mech_dir / "aff_force", L2, sp, n, f1, hess_incar(ne), (FORCE_BALANCE_DIR / "KPOINTS").read_text(),
                    f"{SYSTEM_NAME} aff force (neg)", input_hashes)
    for tag, v in (("s00", 0 * u), ("s10", ub), ("s01", u - ub)):
        strain_main.job(mech_dir / tag, L2, sp, n, f1 + v.reshape(-1, 3) @ inv, READOUT_INCAR, READOUT_KPOINTS,
                        f"{SYSTEM_NAME} {tag} (neg)", input_hashes)
    jobs += [str((mech_dir / t).relative_to(OUTPUT_DIR)) for t in ("aff_force", "s00", "s10", "s01")]
    kp6 = READOUT_KPOINTS.replace("4 4 4", "6 6 6")
    assert kp6 != READOUT_KPOINTS
    for system_name, e in K6_READOUTS:
        strain_main.SOURCE_POTCAR = strain_main.SOURCE_DIRS[system_name] / "POTCAR"
        d = OUTPUT_DIR / "runs" / system_name / f"e{e:+d}" / "final"
        L, sp_, n_, fr = read_poscar(d / "CONTCAR")
        strain_main.job(d / "readout_k6", L, sp_, n_, fr, READOUT_INCAR, kp6, f"{system_name} e{e} readout 6x6x6",
                        input_hashes)
        jobs.append(str((d / "readout_k6").relative_to(OUTPUT_DIR)))
    strain_main.hashes_save(input_hashes)
    (OUTPUT_DIR / "jobs_supp.txt").write_text("\n".join(jobs) + "\n")
    print(len(jobs), "supplementary jobs")


def evaluate():
    seq = strain_main.sequence(SYSTEM_NAME)
    result = {}
    c1, c2 = seq[E1]["C"], seq[E2]["C"]
    result["e_c_neg"] = E1 + (E2 - E1) * c1 / (c1 - c2)
    mech_dir = OUTPUT_DIR / "runs" / SYSTEM_NAME / "mech_neg"
    bonds_data = json.loads((mech_dir / "BONDS.json").read_text())
    d2 = OUTPUT_DIR / "runs" / SYSTEM_NAME / f"e{E2:+d}" / "final"
    L2, sp, n, _ = read_poscar(d2 / "CONTCAR")
    _, _, _, f1 = read_poscar(mech_dir / "s00" / "POSCAR")
    B, _ = strain_main.kernel(f1, L2, bonds_data["bonds"], np.array(bonds_data["net"]))
    N = sum(n)
    f = strain_main.read_forces(mech_dir / "aff_force", N)
    u = np.load(mech_dir / "u_act.npy")
    drive_share, actual_share = (strain_main.share(f, B, np.array(bonds_data["net"])),
                                 strain_main.share(u, B, np.array(bonds_data["net"])))
    states_C = {t: strain_main.margins(mech_dir / t)["C"] for t in ("s00", "s10", "s01")}
    states_C["s11"] = seq[E2]["C"]
    tot = states_C["s11"] - states_C["s00"]
    bend = 0.5 * ((states_C["s10"] - states_C["s00"]) + (states_C["s11"] - states_C["s01"]))
    result["mech_neg"] = {"e1": E1, "e2": E2, "drive_share": drive_share, "actual_share": actual_share,
                          "enrichment_actual": actual_share / drive_share, "states_C": states_C,
                          "dL_C_total": tot, "dL_C_affine": states_C["s00"] - seq[E1]["C"], "phi": bend / tot,
                          "additivity_resid": states_C["s11"] - states_C["s10"] - states_C["s01"] + states_C["s00"],
                          "S2": bool(actual_share / drive_share >= 9 and bend / tot >= 0.5)}
    result["k6"] = {}
    for system_name, e in K6_READOUTS:
        d = OUTPUT_DIR / "runs" / system_name / f"e{e:+d}" / "final"
        result["k6"][f"{system_name} {e:+d}"] = {"k4": strain_main.margins(d / "readout"),
                                                 "k6": strain_main.margins(d / "readout_k6")}
    (OUTPUT_DIR / "strain_supp_result.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    {"prepare": prepare, "eval": evaluate}[sys.argv[1]]()
