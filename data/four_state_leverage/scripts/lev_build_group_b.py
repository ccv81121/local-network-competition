"""Record the group-B phi_pred (leverage_sealed.json and its sha256 in seal_log.txt), then build the group-B four
states as sector_force_balance_prepare.py (network at r_aff with centroid shift, A at P2, P2 lattice; same
INCAR/KPOINTS/POTCAR as the system's four-state inputs). The build step requires leverage_sealed.json.
Reads and writes LEVERAGE_DIR on the cluster (not deposited).
usage: python lev_build_group_b.py seal | build"""
import datetime, hashlib, json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lev_common import LEVERAGE_DIR, SYSTEMS, write_poscar
from lev_four_state_build import build_fourstate

GROUP_B_SYSTEMS = ("Ca2SnTiO6_rs", "Ca8Sn6Ti2O24")
if sys.argv[1] == "seal":
    assert not (LEVERAGE_DIR / "leverage_sealed.json").exists()

    def file_sha256(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    out = {"sealed_utc": datetime.datetime.utcnow().isoformat() + "Z",
           "INPUT_HASHES.json_sha256": file_sha256(LEVERAGE_DIR / "input_hashes.json"), "systems": {}}
    for name in GROUP_B_SYSTEMS:
        leverage = json.loads((LEVERAGE_DIR / "runs" / name / "leverage.json").read_text())
        out["systems"][name] = {k: leverage[k] for k in (
            "margin", "Lambda_meV", "TOTEN_eV", "g_b_meV_per_A", "g_c_meV_per_A", "dP_b_pred_meV", "dP_c_pred_meV",
            "dP_total_pred_meV", "phi_pred", "phi_pred_abs", "E_b_harm_meV", "E_c_harm_meV", "harmonic_energy_share_b")}
        out["systems"][name]["LEVERAGE.json_sha256"] = file_sha256(LEVERAGE_DIR / "runs" / name / "leverage.json")
        out["systems"][name]["outcar_sha256"] = {t: file_sha256(LEVERAGE_DIR / "runs" / name / t / "OUTCAR")
                                                 for t in ("bp", "bm", "cp", "cm")}
        out["systems"][name]["eigenval_sha256"] = {t: file_sha256(LEVERAGE_DIR / "runs" / name / t / "EIGENVAL")
                                                   for t in ("bp", "bm", "cp", "cm")}
    (LEVERAGE_DIR / "leverage_sealed.json").write_text(json.dumps(out, indent=1))
    line = (f"{datetime.datetime.utcnow().isoformat()}Z  sha256(leverage_sealed.json) = "
            f"{file_sha256(LEVERAGE_DIR / 'leverage_sealed.json')}\n")
    with open(LEVERAGE_DIR / "seal_log.txt", "a") as fh:
        fh.write(line)
    print(line, json.dumps(out, indent=1))
elif sys.argv[1] == "build":
    assert (LEVERAGE_DIR / "leverage_sealed.json").exists() and (LEVERAGE_DIR / "seal_log.txt").exists()
    for name in GROUP_B_SYSTEMS:
        _, work_dir, fourstate_dir, _ = SYSTEMS[name]
        meta = json.loads((work_dir / "meta.json").read_text())
        L2, states = build_fourstate(meta)
        for k, fr in states.items():
            d = LEVERAGE_DIR / "runs" / name / "fourstate" / k
            d.mkdir(parents=True, exist_ok=False)
            write_poscar(d / "POSCAR", f"{name} four-state {k} in P5 lattice", L2, meta["species"], meta["counts"], fr)
            shutil.copy(fourstate_dir / k / "INCAR", d / "INCAR")
            shutil.copy(fourstate_dir / k / "KPOINTS", d / "KPOINTS")
            shutil.copy(fourstate_dir / k / "POTCAR", d / "POTCAR")
        print(name, "built")
