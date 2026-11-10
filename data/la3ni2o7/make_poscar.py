"""Conventional Amam cell of La3Ni2O7 (48 atoms) from Ling et al., J. Solid State Chem. 152, 517 (2000), COD 1521521.
Writes poscar_exp (order La Ni O) and prints Ni-O bond lengths and Ni-O-Ni angles as a check."""
import itertools
from pathlib import Path
import numpy as np

a, b, c = 5.39283, 5.44856, 20.51849
OPS = ["x,y,z", "x+1/2,-y,-z", "-x,-y,z", "-x+1/2,y,-z", "-x,-y,-z", "-x-1/2,y,z", "x,y,-z", "x-1/2,-y,z"]
CENT = [(0, 0, 0), (0, .5, .5)]
SITES = {"La1": (0.25, 0.2534, 0.5), "La2": (0.25, 0.2599, 0.31886), "Ni1": (0.25, 0.2548, 0.09568),
         "O1": (0.25, 0.2938, 0.0), "O2": (0.25, 0.2189, 0.2038), "O3": (0.0, 0.5, 0.10442), "O4": (0.5, 0.0, 0.08837)}


def apply(op, p):
    x, y, z = p
    return [eval(e.replace("1/2", "0.5"), {}, {"x": x, "y": y, "z": z}) for e in op.split(",")]


atoms = []
for lab, p in SITES.items():
    pts = []
    for op in OPS:
        for t in CENT:
            q = (np.array(apply(op, p)) + t) % 1.0
            if not any(np.linalg.norm(((q - r + 0.5) % 1) - 0.5) < 1e-4 for r in pts):
                pts.append(q)
    atoms += [(lab, q) for q in pts]
order = ["La", "Ni", "O"]
atoms.sort(key=lambda t: order.index(t[0].rstrip("1234")))
counts = [sum(1 for l, _ in atoms if l.rstrip("1234") == e) for e in order]
assert counts == [12, 8, 28], counts
out = Path(__file__).with_name("poscar_exp")
lines = ["La3Ni2O7 Amam exp (Ling 2000, COD 1521521)", "1.0",
         f"{a:.6f} 0 0", f"0 {b:.6f} 0", f"0 0 {c:.6f}", "La Ni O", " ".join(map(str, counts)), "Direct"]
lines += [f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f}" for _, q in atoms]
out.write_text("\n".join(lines) + "\n")
L = np.diag([a, b, c])
ni = [q for l, q in atoms if l == "Ni1"]
for lab in ("O1", "O2", "O3", "O4"):
    q = next(q for l, q in atoms if l == lab)
    nb = [(n + np.array(s) - q) @ L for n in ni for s in itertools.product((-1, 0, 1), repeat=3)]
    nb = [d for d in nb if np.linalg.norm(d) < 2.4]
    msg = f"{lab}: Ni-O " + ", ".join(f"{np.linalg.norm(d):.3f}" for d in nb)
    if len(nb) == 2:
        cos_angle = nb[0] @ nb[1] / np.linalg.norm(nb[0]) / np.linalg.norm(nb[1])
        msg += f"; Ni-O-Ni {np.degrees(np.arccos(cos_angle)):.1f} deg"
    print(msg)
print("wrote", out, counts)
