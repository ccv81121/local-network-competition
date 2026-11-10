"""D1 read-out: Lambda_C(X), X = P5 internal coordinates in the P4 lattice; rival fixed at the P5 best rival.
Reads the EIGENVAL of the P4 and P5 statics (server_outputs/PRL_TIGHT_SPINEL_20261015/static_k4 of
substitution_path_split) and of state_x next to this script (EIGENVAL files not deposited); writes order_result.json.
usage: python order_readout.py"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH_SPLIT_DIR = HERE.parents[1] / "substitution_path_split"
sys.argv = ["x", str(PATH_SPLIT_DIR / "server_outputs/PRL_MECHANISM_TEST_20261014/Path_Split/work")]
path_readout_source = (PATH_SPLIT_DIR / "path_readout.py").read_text(encoding="utf8")
namespace = {"__name__": "d1"}
# only eig, rivals, lam and same_k are needed
exec(path_readout_source[:path_readout_source.index("res = {")], namespace)
eig, rivals, lam, same_k = namespace["eig"], namespace["rivals"], namespace["lam"], namespace["same_k"]
TIGHT_STATIC_DIR = PATH_SPLIT_DIR / "server_outputs/PRL_TIGHT_SPINEL_20261015/static_k4"
P4, P5 = eig(TIGHT_STATIC_DIR / "P4"), eig(TIGHT_STATIC_DIR / "P5")
X = eig(HERE / "state_x")
kv, kc = rivals(P5)
out = {"rival_C": P5["K"][kc].tolist()}
for name, s in (("P4", P4), ("X", X), ("P5", P5)):
    m = same_k(s, P5)
    out[name] = {"fixed": [round(v, 2) for v in lam(s, m(kv), m(kc))], "own": [round(v, 2) for v in lam(s, *rivals(s))]}
dX = out["X"]["fixed"][1] - out["P4"]["fixed"][1]
out["D1a"] = {"d_int_reverse": round(dX, 2), "d_int_forward": 8.16, "diff": round(dX - 8.16, 2),
              "verdict": "holds" if abs(dX - 8.16) <= 3 else ("falsified" if abs(dX - 8.16) > 6 else "undetermined")}
out["D1b"] = {"Lambda_C_X": out["X"]["fixed"][1], "verdict": "holds" if out["X"]["fixed"][1] < 0 else "falsified"}
(HERE / "order_result.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
