"""Real-space fields of the Ti->Sn substitution in the 80-atom CaTiO3 cell, for the real-field figure.
Re-runs the first part of analyse_dispersion_v1.py (H, drive, prediction, relaxed u, bond projector; no file written by
it) and stores: lattice, s1 positions and species, the drive f, the prediction u_pred = H^+ f, the relaxed u (centroid
removed), their bond-preserving parts (network atoms, network-centred, kernel of the B-X bond Jacobian), the shell
averages of |u| against the distance from Sn and the exponential fits. Writes real_field.npz."""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
src = (HERE / "analyse_dispersion_v1.py").read_text(encoding="utf-8")
src = src.split("# ---- T3")[0] + src.split("# ---- T3")[1].split("# kernel:")[0]
g = {"__file__": str(HERE / "analyse_dispersion_v1.py"), "__name__": "real_field"}
exec(compile(src, "analyse_dispersion_v1_head", "exec"), g)
N, net, Kn = g["N"], g["net"], g["Kn"]


def bond_preserving(x):
    v = x.reshape(-1, 3)[net]; v = (v - v.mean(0)).ravel(); b = np.zeros((N, 3)); b[net] = (Kn @ v).reshape(-1, 3)
    r = np.zeros((N, 3)); r[net] = v.reshape(-1, 3) - b[net]
    return b, r


up, ua = g["upred"].reshape(-1, 3), g["uact_c"].reshape(-1, 3)
upb, upr = bond_preserving(g["upred"]); uab, uar = bond_preserving(g["uact_c"])
fit = lambda r: (r["exp"]["A"], r["exp"]["xi_A"])
np.savez_compressed(HERE / "real_field.npz", lattice=g["lat"], frac=g["fr1"], species=g["sp1"], sn=g["sn"],
                    f=g["Fa"], u_pred=up, u_act=ua, u_pred_b=upb, u_act_b=uab, u_act_r=uar, dist=g["dist"],
                    shells_act=g["shA"], shells_pred=g["shP"], fit_act=fit(g["T3"]["actual"]),
                    fit_pred=fit(g["T3"]["pred_supercell"]))
print("ok", g["T3"]["actual"]["exp"]["xi_A"], g["T3"]["pred_supercell"]["exp"]["xi_A"], float(np.abs(g["Fa"]).max()))
