"""Real-space CBM wavefunctions at Gamma and R on the 20-atom cluster of the VESTA template vesta_fig1a.

Reads WAVECAR (pseudo-wavefunction, no PAW augmentation -> smooth), builds psi_nk(r) = exp(2 pi i k.f) u_nk(f),
fixes the global phase so that psi is real (Gamma and R are time-reversal invariant), samples it in an
orthogonal box around the 20 cluster atoms, applies a smooth mask (1 within R_IN of a cluster atom, 0 beyond R_OUT),
and writes Gaussian cube files plus matching .vesta files (same atoms and view as the template) for isosurfaces.

Inputs not deposited: the WAVECAR of the 5-GPa state11 run and the VESTA template of Fig. 1a, placed in
wavefunction_inputs/ next to this script. Outputs go to wavefunction_cluster/.
usage: python wavefunction_cluster.py
"""
import re
import numpy as np
from pathlib import Path
from scipy.ndimage import map_coordinates
from pymatgen.io.vasp.outputs import Wavecar

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "wavefunction_inputs"
TEMPLATE_VESTA = INPUT_DIR / "spinel_5GPa_bond_preserving.vesta"
POSCAR_STATE11 = HERE.parent / "spinel_pressure_path" / "fourstate" / "state11" / "POSCAR"
WAVECAR_PATH = INPUT_DIR / "WAVECAR"
OUTPUT_DIR = HERE / "wavefunction_cluster"
BOHR = 0.529177210903
ATOMIC_NUMBER = {"Mg": 12, "Sb": 51, "Sc": 21, "Se": 34}
R_IN, R_OUT, BOX_PADDING, GRID_STEP = 2.0, 2.8, 3.0, 0.12  # mask radii, box padding, grid spacing (A)
BAND = 157  # CBM (1-based)
CBM_KPOINTS = {"CB_Gamma": (0, np.zeros(3)), "CB_R": (1, np.full(3, 0.5))}

# structure and the template cluster: B sites inside the template BOUND box + their Se neighbours
poscar_lines = POSCAR_STATE11.read_text().split("\n")
lat = np.array([[float(x) for x in poscar_lines[i].split()] for i in (2, 3, 4)])
species, counts = poscar_lines[5].split(), [int(x) for x in poscar_lines[6].split()]
names = [s for s, k in zip(species, counts) for _ in range(k)]
frac_coords = np.array([[float(x) for x in poscar_lines[8 + i].split()[:3]] for i in range(len(names))]) % 1.0
template = TEMPLATE_VESTA.read_text().replace("\r\n", "\n")
box_bounds = [float(x) for x in template.split("\nBOUND\n", 1)[1].split("\n", 1)[0].split()]
lo, hi = np.array(box_bounds[0::2]), np.array(box_bounds[1::2])
cluster_seed = []
for i, s in enumerate(names):
    if s not in ("Sc", "Sb"):
        continue
    for im in np.array(np.meshgrid(*[[-1, 0, 1]] * 3)).T.reshape(-1, 3):
        f = frac_coords[i] + im
        if np.all(f >= lo - 1e-6) and np.all(f <= hi + 1e-6):
            cluster_seed.append((i, f))
cluster = list(cluster_seed)
for i, f in cluster_seed:
    for j, s in enumerate(names):
        if s != "Se":
            continue
        d = frac_coords[j] - f
        d -= np.round(d)
        if np.linalg.norm(d @ lat) < 3.1:
            g = f + d
            if not any(k == j and np.allclose(g, h) for k, h in cluster):
                cluster.append((j, g))
print("cluster atoms:", len(cluster), {s: sum(names[k] == s for k, _ in cluster) for s in ("Sc", "Sb", "Se")})
xyz = np.array([f @ lat for _, f in cluster])
el = [names[k] for k, _ in cluster]
box_origin = xyz.min(0) - BOX_PADDING
size = xyz.max(0) + BOX_PADDING - box_origin
n_grid = np.ceil(size / GRID_STEP).astype(int)
axes = [box_origin[k] + np.arange(n_grid[k]) * GRID_STEP for k in range(3)]
grid_xyz = np.stack(np.meshgrid(*axes, indexing="ij"), -1)  # Cartesian box grid
grid_frac = grid_xyz.reshape(-1, 3) @ np.linalg.inv(lat)  # fractional (unwrapped)
dmin = np.min(np.linalg.norm(grid_xyz[..., None, :] - xyz[None, None, None], axis=-1), axis=-1)
mask = np.clip((R_OUT - dmin) / (R_OUT - R_IN), 0, 1)
mask = mask * mask * (3 - 2 * mask)  # smoothstep

wavecar = Wavecar(str(WAVECAR_PATH))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
levels = {}
for tag, (ik, kf) in CBM_KPOINTS.items():
    assert np.allclose(wavecar.kpoints[ik], kf, atol=1e-4), wavecar.kpoints[ik]
    u = np.fft.ifftn(wavecar.fft_mesh(ik, BAND - 1))  # periodic part on the FFT grid
    m = np.array(u.shape)
    ur = map_coordinates(u.real, ((grid_frac % 1.0) * m).T, order=3, mode="grid-wrap")
    ui = map_coordinates(u.imag, ((grid_frac % 1.0) * m).T, order=3, mode="grid-wrap")
    psi = (ur + 1j * ui) * np.exp(2j * np.pi * (grid_frac @ kf))
    theta = 0.5 * np.angle(np.sum(psi ** 2))
    psi = (psi * np.exp(-1j * theta)).reshape(n_grid)
    print(tag, "imag/real residual %.3f" % (np.linalg.norm(psi.imag) / np.linalg.norm(psi.real)))
    f = psi.real * mask
    f /= np.abs(f).max()
    levels[tag] = 0.25
    # cube (orthogonal box, bohr)
    lines = [f"MgSc0.5Sb1.5Se4 5 GPa {tag} Re psi (band {BAND}), masked to Fig. 1(a) cluster",
             "normalised to max |psi| = 1"]
    lines.append("%5d %12.6f %12.6f %12.6f" % (len(xyz), *(box_origin / BOHR)))
    for k in range(3):
        v = np.zeros(3)
        v[k] = GRID_STEP / BOHR
        lines.append("%5d %12.6f %12.6f %12.6f" % (n_grid[k], *v))
    for s, p in zip(el, xyz):
        lines.append("%5d %12.6f %12.6f %12.6f %12.6f" % (ATOMIC_NUMBER[s], ATOMIC_NUMBER[s], *(p / BOHR)))
    body = [" ".join("%12.5e" % v for v in row[i:i + 6]) for row in f.reshape(-1, n_grid[2])
            for i in range(0, n_grid[2], 6)]
    (OUTPUT_DIR / f"{tag}_psi.cube").write_text("\n".join(lines + body) + "\n")
    np.savez_compressed(OUTPUT_DIR / f"{tag}_psi_box.npz", psi=f, origin=box_origin, h=GRID_STEP)

# .vesta: template styles, box cell with the 20 cluster atoms, full-cell boundary, same view
fbox = (xyz - box_origin) / size
vesta_text = template
cellp = ("CELLP\n  %.6f   %.6f   %.6f  90.000000  90.000000  90.000000\n"
         "  0.000000   0.000000   0.000000   0.000000   0.000000   0.000000\n" % tuple(size))
i, j = vesta_text.index("\nCELLP\n") + 1, vesta_text.index("\nSTRUC\n") + 1
vesta_text = vesta_text[:i] + cellp + vesta_text[j:]
counts_by_element, struc, theri, sitet = {}, ["STRUC"], ["THERI 0"], ["SITET"]
site_style = {m.group(1): m.group(0)
              for m in re.finditer(r"^\s*\d+\s+([A-Z][a-z]?)\d+\s+.*$",
                                   vesta_text.split("\nSITET\n")[1].split("\nVECTR")[0], re.M)}
for k, (s, p) in enumerate(zip(el, fbox), 1):
    counts_by_element[s] = counts_by_element.get(s, 0) + 1
    lab = f"{s}{counts_by_element[s]}"
    struc += ["%3d %-2s %10s  1.0000   %.6f   %.6f   %.6f    1a       1" % (k, s, lab, *p),
              "                            0.000000   0.000000   0.000000  0.00"]
    theri.append("%3d %10s  1.000000" % (k, lab))
    style = site_style[s].split()
    sitet.append("%3d %10s  " % (k, lab) + "  ".join(style[2:]))
struc.append("  0 0 0 0 0 0 0")
theri.append("  0 0 0")
sitet.append("  0 0 0 0 0 0")
i, j = vesta_text.index("\nSTRUC\n") + 1, vesta_text.index("\nSHAPE\n") + 1
vesta_text = vesta_text[:i] + "\n".join(struc) + "\n" + "\n".join(theri) + "\n" + vesta_text[j:]
i, j = vesta_text.index("\nSITET\n") + 1, vesta_text.index("\nVECTR\n") + 1
vesta_text = vesta_text[:i] + "\n".join(sitet) + "\n" + vesta_text[j:]
i, j = vesta_text.index("\nBOUND\n") + 1, vesta_text.index("\nSBOND\n") + 1
vesta_text = (vesta_text[:i] + "BOUND\n       0        1         0        1         0        1\n  0   0   0   0  0\n"
              + vesta_text[j:])
# no periodic bond search
vesta_text = vesta_text.replace("  0  1  0  0  1", "  0  0  0  0  1").replace("  0  1  1  0  1", "  0  0  1  0  1")
i, j = vesta_text.index("\nVECTR\n") + 1, vesta_text.index("\nSPLAN\n") + 1
vesta_text = vesta_text[:i] + "VECTR\n 0 0 0 0 0\nVECTT\n 0 0 0 0 0\n" + vesta_text[j:]
for tag in CBM_KPOINTS:
    lv = levels[tag]
    vesta_state = vesta_text.replace("\nGROUP\n", f"\nIMPORT_DENSITY 1\n+1.000000e+00 {tag}_psi.cube\n\nGROUP\n", 1)
    k = vesta_state.index("\nISURF\n") + 1
    m2 = vesta_state.index("\n", vesta_state.index("\n", k) + 1) + 1
    vesta_state = (vesta_state[:k]
                   + ("ISURF\n  1   1   %.6f 255 205   0 220 255\n"
                      "  2   1  -%.6f  20 190 210 220 255\n  0   0   0   0\n" % (lv, lv))
                   + vesta_state[m2:])
    (OUTPUT_DIR / f"spinel_5GPa_{tag}_psi.vesta").write_text(vesta_state, newline="\r\n")
print("written:", OUTPUT_DIR)
