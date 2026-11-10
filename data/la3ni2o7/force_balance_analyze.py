"""Force-balance analysis: symmetric-block force constants, force-balance prediction and filtering for the four
intervals.

Input: force_balance/modes.json and the VASP forces of every run in force_balance/forces/ (<interval>_<case>.dat, interval in lower case:
an "energy ... efermi ..." header line followed by the N x 3 forces).
For each interval, with B the 10 orthonormal symmetric modes (rows), e_l a mode, and delta_l = step*sqrt(natoms_l):
  generalized forces F = B f;  H_kl = -[F_k(+l) - F_k(-l)] / (2 delta_l), symmetrized;
  u_pred = B^T H^-1 F_ref;  compared with u_actual = r_end - r_aff (cosine, amplitude ratio).
Filtering: s_f = |P_b f|^2/|f|^2, s_u = |P_b u|^2/|u|^2, enrichment eps = s_u/s_f <= 1/s_f,
with P_b the bond-preserving projector (La fixed, translation removed; la327_network.py).
Also: residual force at the end structure, symmetric content of f, eigenvalues of H, and for P2 the +-0.02 A block.
Writes force_balance/force_balance_result.json and prints a summary. Missing runs are reported, not filled in.
"""
import json
from pathlib import Path

import numpy as np
from ase.io import read

import la327_network as net

HERE = Path(__file__).parent
FORCE_BALANCE_DIR = HERE / "force_balance"


def load(name, case):
    p = FORCE_BALANCE_DIR / "forces" / f"{name.lower()}_{case}.dat"
    if not p.exists():
        return None
    head = p.read_text().splitlines()[0].split()
    return dict(f=np.loadtxt(p).ravel(), energy=float(head[2]), efermi=float(head[4]))


def block(name, meta, suffix):
    B = np.array([m["vec"] for m in meta["modes"]])
    step = meta["step"] if suffix == "" else meta["check"]
    H = np.full((len(B), len(B)), np.nan)
    missing = []
    for col, m in enumerate(meta["modes"]):
        p, q = load(name, f"s{col:02d}p{suffix}"), load(name, f"s{col:02d}m{suffix}")
        if p is None or q is None:
            missing.append(col)
            continue
        delta = step * np.sqrt(m["natoms"])
        H[:, col] = -(B @ p["f"] - B @ q["f"]) / (2 * delta)
    return B, H, missing


def main():
    modes = json.loads((FORCE_BALANCE_DIR / "modes.json").read_text())
    out = {}
    for name, meta in modes.items():
        ref, end = load(name, "ref"), load(name, "end")
        B, H, missing = block(name, meta, "")
        r = dict(missing_modes=missing, ref_done=ref is not None, end_done=end is not None)
        u_act = np.array(meta["u_actual"])
        aff = read(FORCE_BALANCE_DIR / "calculations" / name.lower() / "ref" / "POSCAR")
        bond_list = net.bonds(aff)
        Pb, dim_b, C = net.bond_preserving_projector(aff, bond_list)
        r.update(n_bonds=len(bond_list), dim_b=dim_b, s_u_actual=float(u_act @ Pb @ u_act / (u_act @ u_act)))
        if ref is not None:
            f = ref["f"] - np.repeat(ref["f"].reshape(-1, 3).mean(0)[None], len(aff), 0).ravel()
            s_f = float(f @ Pb @ f / (f @ f))
            r.update(f_norm=float(np.linalg.norm(f)), f_symmetric_fraction=float(np.linalg.norm(B @ f) ** 2 / (f @ f)),
                     s_f=s_f, bound=1 / s_f, enrichment_actual=r["s_u_actual"] / s_f)
        if end is not None:
            r["end_max_force"] = float(np.abs(end["f"]).reshape(-1, 3).max())
        if ref is not None and not missing:
            Hs = (H + H.T) / 2
            r["H_asymmetry"] = float(np.linalg.norm(H - H.T) / np.linalg.norm(Hs))
            r["H_eigenvalues"] = np.linalg.eigvalsh(Hs).tolist()
            u_pred = B.T @ np.linalg.solve(Hs, B @ f)
            s_up = float(u_pred @ Pb @ u_pred / (u_pred @ u_pred))
            r.update(cos_pred_actual=float(u_pred @ u_act / np.linalg.norm(u_pred) / np.linalg.norm(u_act)),
                     cos_force_actual=float(f @ u_act / np.linalg.norm(f) / np.linalg.norm(u_act)),
                     amp_ratio=float(np.linalg.norm(u_pred) / np.linalg.norm(u_act)),
                     s_u_pred=s_up, enrichment_pred=s_up / r["s_f"],
                     harmonic_energy_pred=float(-f @ u_pred / 2))
            if end is not None and ref is not None:
                r["energy_drop_actual"] = end["energy"] - ref["energy"]
            if meta.get("check"):
                _, H2, miss2 = block(name, meta, "x2")
                if not miss2:
                    H2s = (H2 + H2.T) / 2
                    e1, e2 = np.linalg.eigvalsh(Hs), np.linalg.eigvalsh(H2s)
                    r["linearity_eig_rel_change"] = (np.abs(e2 - e1) / np.abs(e1)).tolist()
                    u2 = B.T @ np.linalg.solve(H2s, B @ f)
                    r["linearity_cos_u"] = float(u2 @ u_pred / np.linalg.norm(u2) / np.linalg.norm(u_pred))
        out[name] = r
        keys = ["s_f", "bound", "s_u_actual", "enrichment_actual", "cos_pred_actual", "cos_force_actual", "amp_ratio",
                "s_u_pred", "end_max_force", "missing_modes"]
        print(name, {k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k)) for k in keys if k in r})
    (FORCE_BALANCE_DIR / "force_balance_result.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
