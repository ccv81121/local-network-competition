"""Margins of the A-channel statics (same Lambda rule as lev_readout.py).
Reads the VASP outputs of the three-channel folder on the cluster ($WORK/spinel): not deposited; ../tc_readout.json is the result."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "four_state_leverage" / "scripts"))
from lev_common import Lambda

LEVERAGE_DIR = Path("$WORK/spinel/LEVERAGE_20261023")
OUTPUT_DIR = Path("$WORK/spinel/THREECHANNEL_20261024")
prepare_report = json.loads((LEVERAGE_DIR / "prepare_report.json").read_text())
out = {}
for name in ("k12_t45", "k12_05"):
    margin = prepare_report[name]["margin"]
    out[name] = {}
    for tag in ("ap", "am"):
        lam, toten = Lambda(OUTPUT_DIR / "runs" / name / tag, margin)
        out[name][tag] = {"Lambda_meV": lam, "TOTEN_eV": toten}
(OUTPUT_DIR / "tc_readout.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
