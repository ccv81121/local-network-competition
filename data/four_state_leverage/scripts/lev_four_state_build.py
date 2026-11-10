"""Four-state construction of sector_force_balance_prepare.py (build_fourstate; used by lev_prepare.py and lev_build_group_b.py)."""
import numpy as np
from lev_common import kernel_rows


def build_fourstate(meta):
    """Four states of sector_force_balance_prepare.py: network at r_aff (centroid shifted), A at P2 positions,
    P2 lattice."""
    lat2 = np.array(meta["lattice_P2"])
    f1 = np.array(meta["frac_start"])
    f2 = np.array(meta["frac_P2"])
    net = np.array(meta["network"])
    bonds = meta["bonds"]
    Rb = kernel_rows(f1, lat2, bonds, net)
    df = f2 - f1
    df -= np.round(df)
    raw = df[net] @ lat2
    u = (raw - raw.mean(axis=0)).ravel()
    u_par = Rb.T @ (Rb @ u)
    u_perp = u - u_par
    act2 = (f1[net] + df[net]) @ lat2
    base = act2 - u.reshape(-1, 3)
    states = {"state00": base, "state10": base + u_perp.reshape(-1, 3), "state01": base + u_par.reshape(-1, 3),
              "state11": act2}
    inv2 = np.linalg.inv(lat2)
    out = {}
    for k, cart in states.items():
        fr = f2.copy()
        fr[net] = cart @ inv2
        out[k] = fr
    return lat2, out
