"""Projected geometry of one La3Ni2O7 bilayer (two NiO6 octahedra sharing the inner apical O1) for figure panels.
The projection plane holds c and the in-plane direction in which O1 is displaced from the Ni-Ni line, i.e. the plane
of the Ni-O1-Ni bend; coordinates are in A relative to O1.
usage: from bilayer_geom import bilayer   (run as a script from this folder it prints three drive-plane structures)"""
import numpy as np

from affine_step import sites_from


def bilayer(poscar):
    poscar_lines = open(poscar).read().splitlines()
    lat = np.array([[float(x) for x in poscar_lines[i].split()] for i in (2, 3, 4)])
    n = sum(int(x) for x in poscar_lines[6].split())
    frac = np.array([[float(x) for x in poscar_lines[8 + i].split()[:3]] for i in range(n)])
    sites = sites_from(poscar)
    o1 = sites["O1"][0]

    def rel(i, ref):
        d = frac[i] - frac[ref]
        return (d - np.round(d)) @ lat

    ni = sorted(sites["Ni"], key=lambda i: np.linalg.norm(rel(i, o1)))[:2]
    octs = []
    for k, a in enumerate(ni):
        ra = rel(a, o1)
        o2 = min(sites["O2"], key=lambda j: np.linalg.norm(rel(j, a)))
        op = sorted(sites["Op"], key=lambda j: np.linalg.norm(rel(j, a)))[:4]
        octs.append(dict(Ni=ra, O2=ra + rel(o2, a), Op=[ra + rel(j, a) for j in op]))
    mid = (octs[0]["Ni"] + octs[1]["Ni"]) / 2
    ez = np.array([0, 0, 1.0])
    side = -mid
    side[2] = 0
    ex = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else np.array([1.0, 0, 0])

    def project(v):
        return np.array([v @ ex, v @ ez])

    out = []
    for o in sorted(octs, key=lambda o: o["Ni"][2]):
        out.append(dict(Ni=project(o["Ni"]), O1=project(np.zeros(3)), O2=project(o["O2"]),
                        Op=[project(v) for v in o["Op"]],
                        dO2=float(np.linalg.norm(o["O2"] - o["Ni"]))))
    v1, v2 = -octs[0]["Ni"], -octs[1]["Ni"]
    ang = float(np.degrees(np.arccos(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2))))
    return out, ang


if __name__ == "__main__":
    for c in ["P0", "P2", "film"]:
        o, a = bilayer(f"drive_plane_grid/{c.lower()}_contcar")
        print(c, f"Ni-O1-Ni {a:.2f}", "Ni-O2", [round(x["dO2"], 3) for x in o], "Ni", [x["Ni"].round(3) for x in o])
