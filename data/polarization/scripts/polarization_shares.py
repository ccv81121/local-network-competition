"""Berry-phase polarization of the k12_t45 four-state construction.
predict: reads only the four FD points (bp, bm, cp, cm); writes the predicted channel vectors.
measure: reads the four states and r_aff; evaluates phi' (relative to r_aff), phi (relative to state00) and G1
         (|phi_pred - phi'| <= 0.15 on the same side of 1/2, evaluated when |P11 - P00| >= 0.05 uC/cm^2).
Input: dipole files (dipoles_fd.txt, dipoles_states.txt) with lines "<name> elc px py pz" and "<name> ion px py pz"
(|e| A, from OUTCAR p[elc] / p[ion]).
Ionic part: VASP's p[ion] wraps with the fractional coordinates (cp jumped by one ionic quantum), so the ionic dipole is
recomputed on one branch as sum_i ZVAL_i * (r_raff,i + minimal-image(r_i - r_raff,i)) from poscar_<name> (same lattice).
usage: python polarization_shares.py predict|measure DIR
(DIR holds dipoles_fd.txt / dipoles_states.txt and poscar_raff; writes polarization_channel_derivatives.json /
polarization_result.json there)
"""
import hashlib, json, sys, datetime
from pathlib import Path
import numpy as np

E_PER_A2_TO_UC_CM2 = 1602.176634
UB, UC = 0.02758228562109436, 0.044246836532355255  # |u_b|, |u_c| (A), prepare_report.json k12_t45
DB, DC = 0.00689557140527359, 0.011061709133088814  # delta_b, delta_c (A)


def volume(poscar):
    lines = Path(poscar).read_text().split("\n")
    A = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]) * float(lines[1])
    return abs(np.linalg.det(A)), A


ZVAL = {"Mg": 2.0, "Sb": 5.0, "Sc": 11.0, "Se": 6.0}  # POTCAR: PAW_PBE Mg, Sb, Sc_sv, Se


def frac(poscar):
    lines = Path(poscar).read_text().split("\n")
    species = lines[5].split()
    n = list(map(int, lines[6].split()))
    X = np.array([[float(x) for x in lines[8 + k].split()[:3]] for k in range(sum(n))])
    return X, np.array(sum([[ZVAL[a]] * c for a, c in zip(species, n)], []))


def ion_branch(case_dir, name, A):
    X0, Z = frac(case_dir / "poscar_raff")
    X, Z2 = frac(case_dir / f"poscar_{name}")
    assert (Z == Z2).all()
    d = X - X0
    d -= np.round(d)
    return (Z[:, None] * ((X0 + d) @ A)).sum(axis=0)


def read(path):
    d = {}
    for line in Path(path).read_text().split("\n"):
        t = line.split()
        if len(t) == 5:
            d.setdefault(t[0], {})[t[1]] = np.array([float(x) for x in t[2:]])
    case_dir = Path(path).parent
    _, A = volume(case_dir / "poscar_raff")
    out = {}
    for k, v in d.items():
        ion = ion_branch(case_dir, k, A)
        out[k] = {"elc": v["elc"], "ion": ion, "ion_vasp": v["ion"], "tot": v["elc"] + ion}
    return out


def main(stage, case_dir):
    case_dir = Path(case_dir)
    Om, A = volume(case_dir / "poscar_raff")
    scale = E_PER_A2_TO_UC_CM2 / Om
    if stage == "predict":
        dipoles = read(case_dir / "dipoles_fd.txt")
        out = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "units": "uC/cm^2",
               "volume_A3": Om}
        for part in ("tot", "elc"):
            gb = (dipoles["bp"][part] - dipoles["bm"][part]) * scale / (2 * DB)
            gc = (dipoles["cp"][part] - dipoles["cm"][part]) * scale / (2 * DC)
            out[part] = {"g_b_vec": gb.tolist(), "g_c_vec": gc.tolist(),
                         "dP_b_pred_vec": (gb * UB).tolist(), "dP_c_pred_vec": (gc * UC).tolist()}
        # branch check: electronic change relative to the polarization quantum 2|e|R (ISPIN = 1);
        # the ionic part is on one branch by construction
        q = 2 * min(np.linalg.norm(A, axis=1))
        out["branch_check_max_elc_diff_over_quantum"] = float(max(
            np.abs(dipoles[a]["elc"] - dipoles[b]["elc"]).max() / q for a, b in (("bp", "bm"), ("cp", "cm"))))
        out["ion_branch_vs_vasp_max_diff_eA"] = {k: float(np.abs(dipoles[k]["ion"] - dipoles[k]["ion_vasp"]).max())
                                                 for k in dipoles}
        f = case_dir / "polarization_channel_derivatives.json"
        f.write_text(json.dumps(out, indent=1))
        print(f.name, hashlib.sha256(f.read_bytes()).hexdigest())
        print(json.dumps(out, indent=1))
        return
    derivatives = json.loads((case_dir / "polarization_channel_derivatives.json").read_text())
    dipoles = read(case_dir / "dipoles_states.txt")
    res = {}
    for part in ("tot", "elc"):
        X = {k: dipoles[k][part] * scale for k in ("state00", "state10", "state01", "state11", "raff")}
        e = X["state11"] - X["state00"]
        e = e / np.linalg.norm(e)  # e_hat = (P11 - P00)/|.|
        x = {k: float(v @ e) for k, v in X.items()}
        bend = 0.5 * ((x["state10"] - x["state00"]) + (x["state11"] - x["state01"]))
        phi_p = bend / (x["state11"] - x["raff"])
        phi = bend / (x["state11"] - x["state00"])
        b = float(np.array(derivatives[part]["dP_b_pred_vec"]) @ e)
        c = float(np.array(derivatives[part]["dP_c_pred_vec"]) @ e)
        pred = b / (b + c)
        res[part] = {"proj_uC_cm2": x, "phi_prime": phi_p, "phi": phi, "phi_pred": pred,
                     "dP11_00_uC_cm2": x["state11"] - x["state00"], "dP11_raff_uC_cm2": x["state11"] - x["raff"],
                     "dP_b_pred": b, "dP_c_pred": c, "e_hat": e.tolist()}
    t = res["tot"]
    t["G1_applies"] = abs(t["dP11_00_uC_cm2"]) >= 0.05
    t["G1"] = bool(t["G1_applies"] and abs(t["phi_pred"] - t["phi_prime"]) <= 0.15
                   and (t["phi_pred"] - 0.5) * (t["phi_prime"] - 0.5) > 0)
    (case_dir / "polarization_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
