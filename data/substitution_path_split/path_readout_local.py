"""Read-out of the basis-free path split.
Lambda_V = E_V(G) - max_{k!=G} E_V(k); Lambda_C = min_{k!=G} E_C(k) - E_C(G) on the 4x4x4 mesh (ISYM=0 statics).
Per step the rival k is fixed at the end state of the step and used for every state of that step.
As path_readout.py, with the pressure-path end states read from server_outputs/PRL_TIGHT_SPINEL_20261015/static_k4
next to this script. WORKROOT holds the EIGENVAL of every static: comp/S01..S14/end, comp/step_nn_mm/{chem,aff} and
press/step_Pa_Pc/aff. Writes path_split.json next to WORKROOT.
The EIGENVAL files of WORKROOT and of server_outputs/ are not deposited.
usage: python path_readout_local.py WORKROOT"""
import json
import sys
from pathlib import Path

import numpy as np

WORK_DIR = Path(sys.argv[1])
TIGHT_STATIC_DIR = Path(__file__).resolve().parent / "server_outputs" / "PRL_TIGHT_SPINEL_20261015" / "static_k4"
ORIG = {"S%02d" % n: None for n in range(1, 15)}


def eig(d):
    L = (Path(d) / "EIGENVAL").read_text().splitlines()
    ne, nk, nb = map(int, L[5].split())
    no = ne // 2
    i = 7
    K, V, C = [], [], []
    for _ in range(nk):
        while not L[i].strip():
            i += 1
        K.append(list(map(float, L[i].split()[:3])))
        i += 1
        b = [float(L[i + j].split()[1]) for j in range(nb)]
        i += nb
        V.append(b[no - 1])
        C.append(b[no])
    K = np.array(K)
    g = int(np.flatnonzero(np.all(np.abs(K - np.round(K)) < 1e-6, axis=1))[0])
    return {"K": K, "V": np.array(V), "C": np.array(C), "G": g}


def rivals(s):
    ng = [i for i in range(len(s["K"])) if i != s["G"]]
    return ng[int(np.argmax(s["V"][ng]))], ng[int(np.argmin(s["C"][ng]))]


def lam(s, kv, kc):
    return 1000 * (s["V"][s["G"]] - s["V"][kv]), 1000 * (s["C"][kc] - s["C"][s["G"]])


def same_k(a, b):
    """map index of a k-point of state b onto state a (meshes identical up to ordering/time reversal)"""
    def f(kv):
        q = b["K"][kv]
        for i, p in enumerate(a["K"]):
            if (np.allclose((p - q) - np.round(p - q), 0, atol=1e-6)
                    or np.allclose((p + q) - np.round(p + q), 0, atol=1e-6)):
                return i
    return f


def split(states):
    """states: ordered list of (name, eig); returns per-edge values with the rival fixed at the last state"""
    end = states[-1][1]
    kv, kc = rivals(end)
    out = {"rival_V": end["K"][kv].tolist(), "rival_C": end["K"][kc].tolist(), "Lambda_fixed": {}, "Lambda_own": {}}
    for name, s in states:
        m = same_k(s, end)
        out["Lambda_fixed"][name] = [round(x, 2) for x in lam(s, m(kv), m(kc))]
        out["Lambda_own"][name] = [round(x, 2) for x in lam(s, *rivals(s))]
    return out


res = {"composition": [], "pressure": []}
END_STATES = {n: eig(WORK_DIR / "comp" / f"S{n:02d}" / "end") for n in range(1, 15)}
res["composition_endpoints_own"] = {f"S{n:02d}": [round(x, 2) for x in lam(END_STATES[n], *rivals(END_STATES[n]))]
                                    for n in END_STATES}
for n in range(1, 14):
    b = WORK_DIR / "comp" / f"step_{n:02d}_{n + 1:02d}"
    split_result = split([("start", END_STATES[n]), ("chem", eig(b / "chem")), ("aff", eig(b / "aff")),
                          ("end", END_STATES[n + 1])])
    f = split_result["Lambda_fixed"]
    row = {"step": f"S{n:02d}->S{n + 1:02d}", **split_result}
    for e, j in (("V", 0), ("C", 1)):
        chem = f["chem"][j] - f["start"][j]
        lat = f["aff"][j] - f["chem"][j]
        intl = f["end"][j] - f["aff"][j]
        row[e] = {"d_total": round(f["end"][j] - f["start"][j], 2), "d_chem": round(chem, 2), "d_lat": round(lat, 2),
                  "d_int": round(intl, 2), "d_net": round(lat + intl, 2)}
    res["composition"].append(row)
PRESSURE_LABELS = ["0", "2", "3", "4", "5", "6"]
for a, c in zip(PRESSURE_LABELS[:-1], PRESSURE_LABELS[1:]):
    sa, sc = eig(TIGHT_STATIC_DIR / f"P{a}"), eig(TIGHT_STATIC_DIR / f"P{c}")
    split_result = split([("start", sa), ("aff", eig(WORK_DIR / "press" / f"step_P{a}_P{c}" / "aff")), ("end", sc)])
    f = split_result["Lambda_fixed"]
    row = {"step": f"P{a}->P{c}", **split_result}
    for e, j in (("V", 0), ("C", 1)):
        row[e] = {"d_total": round(f["end"][j] - f["start"][j], 2), "d_lat": round(f["aff"][j] - f["start"][j], 2),
                  "d_int": round(f["end"][j] - f["aff"][j], 2)}
    res["pressure"].append(row)


def opposite(a, b, thr=5.0):
    return None if (abs(a) < thr or abs(b) < thr) else bool(a * b < 0)


evaluation = {}
for e in "VC":
    o = [opposite(r[e]["d_chem"], r[e]["d_net"]) for r in res["composition"]]
    evaluation[f"F1_{e}_opposite_count"] = sum(x is True for x in o)
    evaluation[f"F1_{e}_negligible"] = sum(x is None for x in o)
for step, e in (("S05->S06", "V"), ("S11->S12", "C")):
    r = next(r for r in res["composition"] if r["step"] == step)[e]
    evaluation[f"F2_{step}_{e}"] = {"d_chem": r["d_chem"], "d_net": r["d_net"],
                                    "chem_dominates": abs(r["d_chem"]) > abs(r["d_net"])}
for e in "VC":
    o = [opposite(r[e]["d_lat"], r[e]["d_int"]) for r in res["pressure"]]
    evaluation[f"F3_{e}_opposite_count"] = sum(x is True for x in o)
    evaluation[f"F3_{e}_negligible"] = sum(x is None for x in o)
r45 = next(r for r in res["pressure"] if r["step"] == "P4->P5")["C"]
evaluation["F4_P4P5_C"] = {"d_lat": r45["d_lat"], "d_int": r45["d_int"],
                           "lat_dominates": abs(r45["d_lat"]) > abs(r45["d_int"])}
res["evaluation"] = evaluation
(WORK_DIR.parent / "path_split.json").write_text(
    json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(json.dumps(evaluation, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
for r in res["composition"]:
    print(r["step"], *(f"{e}: tot {r[e]['d_total']:7.1f} chem {r[e]['d_chem']:7.1f} lat {r[e]['d_lat']:7.1f} "
                       f"int {r[e]['d_int']:7.1f}" for e in "VC"))
for r in res["pressure"]:
    print(r["step"], *(f"{e}: tot {r[e]['d_total']:7.1f} lat {r[e]['d_lat']:7.1f} int {r[e]['d_int']:7.1f}"
                       for e in "VC"))
print("endpoints own-rival (V, C):", res["composition_endpoints_own"])
