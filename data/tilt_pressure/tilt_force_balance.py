"""Octahedral tilt response to pressure in CaTiO3 and CaSnO3 from the force constants (0->5 GPa, fixed 5-GPa lattice).
Tilt coordinate: mean B-O-B angle theta(r).  Delta theta of internal relaxation = g.u, g = d theta / d r at r_aff.
Split of the predicted g.H^-1 f along the unit tilt direction e = g/|g|:
  direct   = (e.f) * (e.H^-1 e)        force on the tilt itself, times the tilt compliance (others relaxed)
  coupled  = e.H^-1 (f - (e.f) e)      tilt produced by forces on the other coordinates (bond compression etc.)
Input per material: ../bending_enrichment/full_force_balance/<ca_tio3|ca_sno3>/p0_p5/{meta.json, H.npy,
hessian/ref/forces.npy} (lattices, fractional coordinates and B-O bond list of the 0 and 5 GPa structures, force-constant
matrix, forces of the reference run).
Prints the angles, the tilt response and its direct/coupled split for both materials.
"""
import json, sys
import numpy as np
from pathlib import Path
sys.stdout.reconfigure(encoding="utf8")
ROOT = Path(__file__).resolve().parents[1] / "bending_enrichment" / "full_force_balance"


def theta(pos, L, pairs):
    a = []
    for o, b1, i1, b2, i2 in pairs:
        d1 = pos[b1] + np.array(i1) @ L - pos[o]
        d2 = pos[b2] + np.array(i2) @ L - pos[o]
        a.append(np.degrees(np.arccos(d1 @ d2 / np.linalg.norm(d1) / np.linalg.norm(d2))))
    return np.mean(a)


for mat, material_dir in (("CaTiO3", "ca_tio3"), ("CaSnO3", "ca_sno3")):
    case_dir = ROOT / material_dir / "p0_p5"
    m = json.loads((case_dir / "meta.json").read_text())
    L2 = np.array(m["lattice_P2"])
    f1 = np.array(m["frac_start"])
    f2 = np.array(m["frac_P2"])
    sp = np.repeat(m["species"], m["counts"])
    N = len(sp)
    # B-O-B pairs from the bond list: each O has two B neighbours
    neighbours_of_o = {}
    for i, j, img in m["bonds"]:
        neighbours_of_o.setdefault(j, []).append((i, img))
    pairs = []
    for o, nb in neighbours_of_o.items():
        assert len(nb) == 2, (o, nb)
        (b1, i1), (b2, i2) = nb
        # bond vector B->O uses O image img: r_O + img - r_B ; so B seen from O sits at -img
        pairs.append((o, b1, [-x for x in i1], b2, [-x for x in i2]))
    r_aff = f1 @ L2
    d = f2 - f1
    d -= np.round(d)
    r_rel = (f1 + d) @ L2
    th_aff = theta(r_aff, L2, pairs)
    th_rel = theta(r_rel, L2, pairs)
    # P1 structure in its own lattice: need lattice_P1 if stored
    L1 = np.array(m["lattice_P1"]) if "lattice_P1" in m else None
    th_P1 = theta(f1 @ L1, L1, pairs) if L1 is not None else float("nan")
    # gradient of theta at r_aff
    h = 1e-4
    g = np.zeros(3 * N)
    for k in range(3 * N):
        p = r_aff.copy()
        p[k // 3, k % 3] += h
        q = r_aff.copy()
        q[k // 3, k % 3] -= h
        g[k] = (theta(p, L2, pairs) - theta(q, L2, pairs)) / (2 * h)
    H = np.load(case_dir / "H.npy")

    def read_forces(c):
        return np.load(case_dir / "hessian" / c / "forces.npy")

    T = np.zeros((3 * N, 3))
    for a in range(3):
        T[a::3, a] = 1 / np.sqrt(N)
    Q = np.eye(3 * N) - T @ T.T
    w, V = np.linalg.eigh(Q @ H @ Q)
    keep = np.argsort(np.abs(w))[3:]
    Hp = V[:, keep] @ np.diag(1 / w[keep]) @ V[:, keep].T
    f = Q @ read_forces("ref").ravel()
    u = Q @ (r_rel - r_aff).ravel()
    g = Q @ g
    e = g / np.linalg.norm(g)
    pred = g @ Hp @ f
    act = g @ u
    fe = e @ f
    comp = e @ Hp @ e
    direct = np.linalg.norm(g) * fe * comp
    coupled = np.linalg.norm(g) * (e @ Hp @ (f - fe * e))
    print(f"{mat}: theta P1={th_P1:.3f}  aff={th_aff:.3f}  relaxed={th_rel:.3f} deg")
    print(f"   internal relaxation d(theta): actual {act:+.3f}, linear g.u {act:+.3f}, "
          f"predicted g.H^-1 f {pred:+.3f} deg")
    print(f"   force on tilt e.f = {fe:+.4f} eV/A ; tilt compliance e.H^-1 e = {comp:.4f} A^2/eV "
          f"(stiffness {1/comp:.2f} eV/A^2) ; |g| = {np.linalg.norm(g):.2f} deg/A")
    print(f"   direct {direct:+.3f} deg ({direct/pred*100:.1f}%), coupled {coupled:+.3f} deg ({coupled/pred*100:.1f}%)")
    print(f"   |f| = {np.linalg.norm(f):.4f} eV/A ; share of squared force on tilt direction {fe**2/(f@f)*100:.2f}%")
