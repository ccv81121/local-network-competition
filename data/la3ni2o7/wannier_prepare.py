"""Wannier inputs for La3Ni2O7 (electronic factor of the readout) and the 0-GPa start structure.

calculations/<i>/<state>  states 00, 10, 01, 01ip, 01ap, 11 of each interval (POSCARs from four_state/pbe), plus
                  calculations/p0start/00. Interval directories are lower case; the labels P1..S2 stay in the INCAR titles.
                  PBE, ISMEAR 1 / SIGMA 0.05, Gamma 6x6x2 full mesh (ISYM = -1), NBANDS 288, VASP's built-in
                  wannier90 (LWANNIER90_RUN). Projected Wannier functions (num_iter = 0, disentanglement only):
                  Ni d (8 x 5) + O p (28 x 3) = 124, one projection per atom at its own position, z along c.
                  Outer window E_F - 10 .. E_F + 4 eV, frozen window E_F - 10 .. E_F + 0.5 eV, with E_F taken from
                  the force-balance reference run of the same structure (p0start: the relaxed 0-GPa run).
                  00 and P0start/00 run from scratch; the other states restart from 00's WAVECAR.
readout/p0start/{pbe,u4}  12x12x3 readout of the 0-GPa structure (affine-step baseline for P1 and S1).
Writes basis_map.json per Wannier directory (orbital order, atom, species, O class) and the queues.
"""
import json
import shutil
from pathlib import Path

from ase.io import read

import la327_network as net

HERE = Path(__file__).parent
FOUR_STATE_DIR, WANNIER_DIR = HERE / "four_state", HERE / "wannier"
FERMI_ENERGIES = {"P1": 8.7350, "P2": 9.1166, "S1": 8.4055, "S2": 8.4794, "P0start": 8.3581}
STATES = ["00", "10", "01", "01ip", "01ap", "11"]
D_ORBITALS, P_ORBITALS = ["dz2", "dxz", "dyz", "dx2-y2", "dxy"], ["pz", "px", "py"]
INCAR_WANNIER = """PREC = Accurate
ENCUT = 520
EDIFF = 1E-7
ISMEAR = 1
SIGMA = 0.05
LREAL = .FALSE.
LASPH = .TRUE.
NELM = 200
IBRION = -1
NSW = 0
ISYM = -1
NBANDS = 288
LCHARG = .FALSE.
LWANNIER90_RUN = .TRUE.
LWRITE_MMN_AMN = .TRUE.
NUM_WANN = 124
"""


def o_classes(atoms):
    bond_list = net.bonds(atoms)
    classes_by_o = {}
    for b in bond_list:
        classes_by_o.setdefault(b["o"], []).append(b["kind"])
    out = {}
    for o, kinds in classes_by_o.items():
        out[o] = "O1_inner" if kinds.count("apical") == 2 else "O2_outer" if kinds == ["apical"] else "O_inplane"
    return out


def wannier90_win(atoms, ef):
    sym = atoms.get_chemical_symbols()
    frac = atoms.get_scaled_positions()
    cls = o_classes(atoms)
    proj, basis = [], []
    for el, l, orbs in (("Ni", 2, D_ORBITALS), ("O", 1, P_ORBITALS)):
        for i, s in enumerate(sym):
            if s != el:
                continue
            f = frac[i]
            proj.append(f"f={f[0]:.10f},{f[1]:.10f},{f[2]:.10f}:l={l}")
            for o in orbs:
                basis.append(dict(atom=i, species=s, orbital=o, site=cls.get(i, "Ni")))
    txt = f"""num_wann = 124
dis_win_min = {ef - 10:.4f}
dis_win_max = {ef + 4:.4f}
dis_froz_min = {ef - 10:.4f}
dis_froz_max = {ef + 0.5:.4f}
dis_num_iter = 400
dis_mix_ratio = 0.5
dis_conv_tol = 1.0d-10
num_iter = 0
guiding_centres = true
write_hr = true
write_xyz = true
translate_home_cell = true
search_shells = 400
begin projections
""" + "\n".join(proj) + "\nend projections\n"
    return txt, basis


def put(calc_dir, interval_label, atoms, ef, scratch):
    calc_dir.mkdir(parents=True, exist_ok=True)
    txt, basis = wannier90_win(atoms, ef)
    (calc_dir / "wannier90.win").write_bytes(txt.encode())
    (calc_dir / "basis_map.json").write_text(json.dumps(dict(basis=basis, efermi_window=ef)))
    extra = "ALGO = Normal\nLWAVE = .TRUE.\n" if scratch else "ALGO = Fast\nISTART = 1\nLWAVE = .FALSE.\n"
    (calc_dir / "INCAR").write_bytes(
        f"SYSTEM = La3Ni2O7 wannier {interval_label} {calc_dir.name}\n{INCAR_WANNIER}{extra}".encode())
    (calc_dir / "KPOINTS").write_bytes(b"Gamma\n0\nGamma\n6 6 2\n0 0 0\n")


def main():
    scratch_jobs, restart_jobs = [], []
    for interval in ["P1", "P2", "S1", "S2"]:
        for state in STATES:
            src = FOUR_STATE_DIR / "pbe" / interval.lower() / state / "POSCAR"
            calc_dir = WANNIER_DIR / "calculations" / interval.lower() / state
            calc_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, calc_dir / "POSCAR")
            put(calc_dir, interval, read(src), FERMI_ENERGIES[interval], state == "00")
            (scratch_jobs if state == "00" else restart_jobs).append(f"calculations/{interval.lower()}/{state}")
    p0_poscar = HERE / "relaxation" / "pressure_p0.vasp"
    calc_dir = WANNIER_DIR / "calculations" / "p0start" / "00"
    calc_dir.mkdir(parents=True, exist_ok=True)
    (calc_dir / "POSCAR").write_bytes(p0_poscar.read_bytes().replace(b"\r\n", b"\n"))
    put(calc_dir, "P0start", read(p0_poscar), FERMI_ENERGIES["P0start"], True)
    scratch_jobs.append("calculations/p0start/00")
    for tag, src in (("pbe", FOUR_STATE_DIR / "pbe" / "p1" / "00" / "INCAR"),
                     ("u4", FOUR_STATE_DIR / "pbe_u4" / "p1" / "00" / "INCAR")):
        calc_dir = WANNIER_DIR / "readout" / "p0start" / tag
        calc_dir.mkdir(parents=True, exist_ok=True)
        (calc_dir / "POSCAR").write_bytes(p0_poscar.read_bytes().replace(b"\r\n", b"\n"))
        txt = src.read_text().replace("ALGO = Normal\nLWAVE = .TRUE.\n", "ALGO = Normal\nLWAVE = .FALSE.\n")
        (calc_dir / "INCAR").write_bytes(txt.replace("P1 00", "P0start").encode())
        (calc_dir / "KPOINTS").write_bytes(b"Gamma\n0\nGamma\n12 12 3\n0 0 0\n")
        scratch_jobs.append(f"readout/p0start/{tag}")
    (WANNIER_DIR / "jobs_ref.txt").write_bytes(("\n".join(scratch_jobs) + "\n").encode())
    (WANNIER_DIR / "jobs_rest.txt").write_bytes(("\n".join(restart_jobs) + "\n").encode())
    print(len(scratch_jobs), "from scratch,", len(restart_jobs), "restarts")


if __name__ == "__main__":
    main()
