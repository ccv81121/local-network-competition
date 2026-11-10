"""Full-cell Re(psi) of the CBM at Gamma and R (5 GPa, state11) for isosurfaces in VESTA.

Reads the CBM band of the WAVECAR at Gamma and R and the PROCAR weights of that band. No masking and no re-boxing: the
cube uses the original cell vectors, so VESTA keeps the frame of the template (vesta_fig1a).
Writes, per state: <tag>_psi_cell.cube, a full-cell .vesta, and an inset .vesta centred on the atom with the
largest weight in that state (Sc for Gamma, Sb for R).

Inputs not deposited: the PROCAR and WAVECAR of the 5-GPa state11 run and the VESTA template of Fig. 1a, placed
in wavefunction_inputs/ next to this script. Outputs go to wavefunction_cell/.
usage: python wavefunction_cell.py
"""
import re
import numpy as np
from pathlib import Path
from pymatgen.io.vasp.outputs import Wavecar

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "wavefunction_inputs"
TEMPLATE_VESTA = INPUT_DIR / "spinel_5GPa_bond_preserving.vesta"
POSCAR_STATE11 = HERE.parent / "spinel_pressure_path" / "fourstate" / "state11" / "POSCAR"
PROCAR = INPUT_DIR / "PROCAR"
WAVECAR = INPUT_DIR / "WAVECAR"
OUTPUT_DIR = HERE / "wavefunction_cell"
BOHR = 0.529177210903
ATOMIC_NUMBER = {"Mg": 12, "Sb": 51, "Sc": 21, "Se": 34}
BAND = 157
CBM_KPOINTS = {"CB_Gamma": (0, np.zeros(3)), "CB_R": (1, np.full(3, 0.5))}
ISO_LEVEL = 0.18  # isovalue as a fraction of max |Re psi|
INSET_HALF_WIDTH = 0.21  # inset half-width in fractional units (~2.4 A)
POLYHEDRON_COLOR = (214, 218, 222)
SITE_COLOR = {"Mg": (175, 180, 185), "Sb": (205, 150, 50), "Sc": (45, 140, 130), "Se": (120, 135, 150)}
SITE_RADIUS = {"Mg": 0.30, "Sb": 0.42, "Sc": 0.42, "Se": 0.26}

poscar_lines = POSCAR_STATE11.read_text().split("\n")
lat = np.array([[float(x) for x in poscar_lines[i].split()] for i in (2, 3, 4)])
species, counts = poscar_lines[5].split(), [int(x) for x in poscar_lines[6].split()]
names = [s for s, k in zip(species, counts) for _ in range(k)]
frac_coords = np.array([[float(x) for x in poscar_lines[8 + i].split()[:3]] for i in range(len(names))]) % 1.0

procar_text = PROCAR.read_text()
k_blocks = re.split(r"\n k-point\s+\d+ :", procar_text)[1:]
ion_weights = {}
for tag, blk in zip(CBM_KPOINTS, k_blocks):
    m = re.search(r"band\s+%d # energy.*?\nion[^\n]*\n(.*?)\ntot" % BAND, blk, re.S)
    ion_weights[tag] = np.array([float(r.split()[-1]) for r in m.group(1).split("\n")[:56]])

template = TEMPLATE_VESTA.read_text().replace("\r\n", "\n")
i, j = template.index("VECTR\n"), template.index("SPLAN\n")
template = template[:i] + "VECTR\n 0 0 0 0 0\nVECTT\n 0 0 0 0 0\n" + template[j:]
template = template.replace("POLYP\n 115 0", "POLYP\n  60 0")
template = (template.replace("0  1  0  0  1  0.250", "0  0  1  0  1  0.030")
            .replace("0  1  1  0  1  0.250", "0  1  1  0  1  0.030"))
site = ["SITET"]
counts_by_element = {}
for k, el in enumerate(names, 1):
    counts_by_element[el] = counts_by_element.get(el, 0) + 1
    site.append("%3d %10s  %.4f %3d %3d %3d %3d %3d %3d  60  0" % (k, f"{el}{counts_by_element[el]}", SITE_RADIUS[el],
                                                                   *SITE_COLOR[el], *POLYHEDRON_COLOR))
site.append("  0 0 0 0 0 0")
a, b = template.index("\nSITET\n") + 1, template.index("\nVECTR\n") + 1
template = template[:a] + "\n".join(site) + "\n" + template[b:]


def with_bound(t, lo, hi):
    i, j = t.index("\nBOUND\n") + 1, t.index("\nSBOND\n") + 1
    return t[:i] + ("BOUND\n  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f\n  0   0   0   0  0\n"
                    % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])) + t[j:]


def with_density(t, cube, lv):
    t = t.replace("\nGROUP\n", f"\nIMPORT_DENSITY 1\n+1.000000e+00 {cube}\n\nGROUP\n", 1)
    k = t.index("\nISURF\n") + 1
    m2 = t.index("\n", t.index("\n", k) + 1) + 1
    return t[:k] + ("ISURF\n  1   1   %.6f 255 205   0 230 255\n"
                    "  2   1  -%.6f  20 190 210 230 255\n  0   0   0   0\n" % (lv, lv)) + t[m2:]


wavecar = Wavecar(str(WAVECAR))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for tag, (ik, kf) in CBM_KPOINTS.items():
    assert np.allclose(wavecar.kpoints[ik], kf, atol=1e-4)
    c_g = np.fft.fftshift(wavecar.fft_mesh(ik, BAND - 1))
    pad = [((2 * k) // 2 - k // 2, 2 * k - k - ((2 * k) // 2 - k // 2)) for k in c_g.shape]  # keep G=0 at the centre
    u = np.fft.ifftn(np.fft.ifftshift(np.pad(c_g, pad)))  # Fourier interpolation to a 2x finer grid
    m = np.array(u.shape)
    f = np.stack(np.meshgrid(*[np.arange(k) / k for k in m], indexing="ij"), -1)
    psi = u * np.exp(2j * np.pi * (f @ kf))
    psi *= np.exp(-0.5j * np.angle(np.sum(psi ** 2)))
    print(tag, "grid", m, "imag/real %.3f" % (np.linalg.norm(psi.imag) / np.linalg.norm(psi.real)))
    val = psi.real / np.abs(psi.real).max()
    np.savez_compressed(HERE / f"PSI_CELL_{tag}.npz", psi=val.astype(np.float32), lat=lat, frac=frac_coords,
                        names=np.array(names), weights=ion_weights[tag] / ion_weights[tag].sum())
    lines = [f"MgSc0.5Sb1.5Se4 5 GPa {tag} Re psi band {BAND}", "normalised to max |Re psi| = 1"]
    lines.append("%5d %12.6f %12.6f %12.6f" % (len(names), 0, 0, 0))
    for k in range(3):
        lines.append("%5d %12.6f %12.6f %12.6f" % (m[k], *(lat[k] / m[k] / BOHR)))
    for s, p in zip(names, frac_coords @ lat):
        lines.append("%5d %12.6f %12.6f %12.6f %12.6f" % (ATOMIC_NUMBER[s], ATOMIC_NUMBER[s], *(p / BOHR)))
    body = [" ".join("%12.5e" % v for v in row[i:i + 6]) for row in val.reshape(-1, m[2]) for i in range(0, m[2], 6)]
    cube = f"{tag}_psi_cell.cube"
    (OUTPUT_DIR / cube).write_text("\n".join(lines + body) + "\n")
    full = with_density(with_bound(template, np.zeros(3), np.ones(3)), cube, ISO_LEVEL)
    el = "Sc" if tag == "CB_Gamma" else "Sb"
    idx = max((k for k, s in enumerate(names) if s == el), key=lambda k: ion_weights[tag][k])
    c = frac_coords[idx]
    inset = with_density(with_bound(template, c - INSET_HALF_WIDTH, c + INSET_HALF_WIDTH), cube, ISO_LEVEL)
    (OUTPUT_DIR / f"{tag}_cell.vesta").write_text(full, newline="\r\n")
    (OUTPUT_DIR / f"{tag}_inset_{el}{idx + 1}.vesta").write_text(inset, newline="\r\n")
    print(tag, "inset centred on", f"{el}{idx + 1}", "weight %.3f" % (ion_weights[tag][idx] / ion_weights[tag].sum()))
