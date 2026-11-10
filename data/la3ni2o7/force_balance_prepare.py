"""Force-balance inputs for La3Ni2O7: affine references and symmetric-block force constants for the four intervals.

Intervals (start -> end): P1 0->4 GPa, P2 4->8 GPa, S1 0 -> -1 %, S2 -1 % -> -2 % (relaxed CONTCARs in relaxation/).
For each interval:
  ref   r_aff = start fractional coordinates in the end lattice; its forces are the drive f.
  end   the relaxed end structure at the same settings (residual-force check, actual displacement).
  sNN+- r_aff displaced by +-0.01 A per atom along each totally symmetric internal mode (one Wyckoff orbit per mode).
P2 also gets the same modes at +-0.02 A (linearity check). Modes come from the Amam operations found by spglib on
r_aff: each orbit's three Cartesian unit vectors are projected onto the identity representation and orthonormalised.
The actual internal displacement r_end - r_aff must lie in this subspace (printed residual).
Writes force_balance/calculations/<i>/<case>/{POSCAR,INCAR,KPOINTS}, force_balance/modes.json,
force_balance/jobs_ref.txt and force_balance/jobs_disp.txt.
"""
import json
from pathlib import Path

import numpy as np
import spglib
from ase.io import read, write

HERE = Path(__file__).parent
RELAXATION_DIR = HERE / "relaxation"
FORCE_BALANCE_DIR = HERE / "force_balance"
# interval labels P1..S2 are the JSON keys; directories and structure files are lowercase on disk
INTERVALS = {"P1": ("pressure_P0", "pressure_P4"), "P2": ("pressure_P4", "pressure_P8"),
             "S1": ("pressure_P0", "strain_eta-1"), "S2": ("strain_eta-1", "strain_eta-2")}
DISPLACEMENT_STEP = 0.01
CHECK_STEPS = {"P2": 0.02}
INCAR_COMMON = """PREC = Accurate
ENCUT = 520
EDIFF = 1E-8
ISMEAR = 1
SIGMA = 0.05
LREAL = .FALSE.
LASPH = .TRUE.
ISYM = 0
NELM = 200
IBRION = -1
NSW = 0
LCHARG = .FALSE.
"""
INCAR_REFERENCE = INCAR_COMMON + "ALGO = Normal\nLWAVE = .TRUE.\n"
INCAR_DISPLACED = INCAR_COMMON + "ALGO = Fast\nISTART = 1\nLWAVE = .FALSE.\n"
KPOINTS_TEXT = "Gamma\n0\nGamma\n8 8 2\n0 0 0\n"


def symmetric_modes(atoms):
    cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
    symmetry = spglib.get_symmetry(cell, symprec=1e-3)
    dataset = spglib.get_symmetry_dataset(cell, symprec=1e-3)
    lat, frac, n = atoms.cell.array, atoms.get_scaled_positions(), len(atoms)
    ops = []
    for R, t in zip(symmetry["rotations"], symmetry["translations"]):
        new = frac @ R.T + t
        perm = np.empty(n, int)
        for i, p in enumerate(new):
            d = frac - p
            d -= np.round(d)
            j = int(np.argmin(np.linalg.norm(d @ lat, axis=1)))
            assert np.linalg.norm(d[j] @ lat) < 1e-2
            perm[i] = j  # atom i goes to atom perm[i]
        Rc = lat.T @ R @ np.linalg.inv(lat.T)  # Cartesian rotation
        ops.append((Rc, perm))
    modes, orbits = [], sorted(set(dataset.equivalent_atoms))
    for rep in orbits:
        members = np.where(dataset.equivalent_atoms == rep)[0]
        vecs = []
        for a in range(3):
            v = np.zeros((n, 3))
            for Rc, perm in ops:  # average g.e over the group
                e = np.zeros(3)
                e[a] = 1
                v[perm[rep]] += Rc @ e
            v /= len(ops)
            vecs.append(v.ravel())
        U, s, _ = np.linalg.svd(np.array(vecs).T, full_matrices=False)
        for k in range(len(s)):
            if s[0] > 0 and s[k] > 1e-6 * s[0]:
                modes.append(dict(orbit=int(rep), symbol=atoms[int(rep)].symbol, natoms=len(members),
                                  vec=U[:, k]))
    return dataset.international, len(ops), modes


def main():
    FORCE_BALANCE_DIR.mkdir(exist_ok=True)
    meta, reference_jobs, displaced_jobs = {}, [], []
    for name, (start_name, end_name) in INTERVALS.items():
        start, end = (read(RELAXATION_DIR / f"{start_name.lower()}.vasp"),
                      read(RELAXATION_DIR / f"{end_name.lower()}.vasp"))
        aff = end.copy()
        aff.set_scaled_positions(start.get_scaled_positions())
        space_group, n_operations, modes = symmetric_modes(aff)
        B = np.array([m["vec"] for m in modes])  # orthonormal rows
        d = end.get_scaled_positions() - aff.get_scaled_positions()
        d -= np.round(d)
        u = (d @ end.cell.array).ravel()
        u -= np.repeat(u.reshape(-1, 3).mean(0)[None], len(end), 0).ravel()
        residual = np.linalg.norm(u - B.T @ (B @ u)) / np.linalg.norm(u)
        print(f"{name}: {space_group} ({n_operations} ops), {len(modes)} symmetric modes, "
              f"|u| = {np.linalg.norm(u):.4f} A, out-of-subspace residual {residual:.2e}")
        root = FORCE_BALANCE_DIR / "calculations" / name.lower()

        def case(tag, atoms, incar):
            case_dir = root / tag
            case_dir.mkdir(parents=True, exist_ok=True)
            write(case_dir / "POSCAR", atoms, format="vasp", direct=True, sort=False)
            p = case_dir / "POSCAR"  # VASP needs LF line ends; ase writes CRLF on Windows
            p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
            (case_dir / "INCAR").write_text(f"SYSTEM = La3Ni2O7 stage1 {name} {tag}\n" + incar)
            (case_dir / "KPOINTS").write_text(KPOINTS_TEXT)
            return f"calculations/{name.lower()}/{tag}"

        reference_jobs.append(case("ref", aff, INCAR_REFERENCE))
        displaced_jobs.append(case("end", end, INCAR_DISPLACED))
        amps = {"": DISPLACEMENT_STEP, **({"x2": CHECK_STEPS[name]} if name in CHECK_STEPS else {})}
        for k, m in enumerate(modes):
            for suf, step in amps.items():
                amp = step * np.sqrt(m["natoms"])  # every atom of the orbit moves by `step`
                for sgn, lab in ((1, "p"), (-1, "m")):
                    at = aff.copy()
                    at.positions = aff.positions + sgn * amp * m["vec"].reshape(-1, 3)
                    displaced_jobs.append(case(f"s{k:02d}{lab}{suf}", at, INCAR_DISPLACED))
        meta[name] = dict(start=start_name, end=end_name, spacegroup=space_group, n_ops=n_operations,
                          step=DISPLACEMENT_STEP, check=CHECK_STEPS.get(name), u_actual=u.tolist(),
                          out_of_subspace=residual,
                          modes=[dict(orbit=m["orbit"], symbol=m["symbol"], natoms=m["natoms"],
                                      vec=m["vec"].tolist()) for m in modes])
    (FORCE_BALANCE_DIR / "modes.json").write_text(json.dumps(meta))
    (FORCE_BALANCE_DIR / "jobs_ref.txt").write_text("\n".join(reference_jobs) + "\n")
    (FORCE_BALANCE_DIR / "jobs_disp.txt").write_text("\n".join(displaced_jobs) + "\n")
    print(len(reference_jobs), "reference runs,", len(displaced_jobs), "restarted runs")


if __name__ == "__main__":
    main()
