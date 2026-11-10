"""Write r_aff +- 0.25|u_A| d_A POSCARs (u_A = A-site (Mg) part of u_c) for k12_t45 and k12_05.
Reads and writes the leverage and three-channel folders on the cluster ($WORK/spinel): not deposited."""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "four_state_leverage" / "scripts"))
from lev_common import read_poscar, write_poscar

LEVERAGE_DIR = Path("$WORK/spinel/LEVERAGE_20261023")
OUTPUT_DIR = Path("$WORK/spinel/THREECHANNEL_20261024")
prepare_report = json.loads((LEVERAGE_DIR / "prepare_report.json").read_text())
info = {}
for name in ("k12_t45", "k12_05"):
    run_dir = LEVERAGE_DIR / "runs" / name
    lat, sp, cnt, f1 = read_poscar(run_dir / "raff" / "POSCAR")
    specs = sum([[s] * c for s, c in zip(sp, cnt)], [])
    A = [i for i, s in enumerate(specs) if s == "Mg"]
    uc = np.load(run_dir / "u_c.npy").reshape(-1, 3)
    uA = np.zeros_like(uc)
    uA[A] = uc[A]
    nA = float(np.linalg.norm(uA))
    assert abs(nA - prepare_report[name]["uc_A_cation_norm_A"]) < 1e-9, (
        nA, prepare_report[name]["uc_A_cation_norm_A"])
    assert len(A) == prepare_report[name]["N"] - prepare_report[name]["n_network"]
    dA = uA / nA
    inv = np.linalg.inv(lat)
    info[name] = {"uA_norm_A": nA, "delta_A_A": 0.25 * nA, "A_idx": A}
    for tag, sgn in (("ap", 1), ("am", -1)):
        d = OUTPUT_DIR / "runs" / name / tag
        d.mkdir(parents=True, exist_ok=True)
        write_poscar(d / "POSCAR", f"{name} threechannel {tag} (r_aff {'+-'[sgn < 0]}0.25|u_A| d_A)", lat, sp, cnt,
                     f1 + (sgn * 0.25 * nA * dA) @ inv)
        for filename in ("INCAR", "KPOINTS", "POTCAR"):
            shutil.copy(run_dir / "bp" / filename, d / filename)
        info[name][tag + "_sha256"] = hashlib.sha256((d / "POSCAR").read_bytes()).hexdigest()
(OUTPUT_DIR / "a_channel_steps.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
