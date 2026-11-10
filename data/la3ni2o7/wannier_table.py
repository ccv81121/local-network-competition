"""Tabulate the Wannier decomposition (wannier/wannier3_<i>.json): band-top part by matrix-element class (exact at
Gamma), with the E_F part taken as a whole from the VASP readouts of four_state/readout.json (12x12x3, tetrahedron).
Order-averaged channels: d_b = [(10-00) + (11-01)]/2, d_c = [(01-00) + (11-10)]/2. Writes wannier/wannier_table.md."""
import json
from pathlib import Path

HERE = Path(__file__).parent
READOUT = json.loads((HERE / "four_state/readout.json").read_text())
CLASS_GROUPS = {"t⊥ path: hop Ni-O1 (inner apical)": ["hop Ni-O1 (inner apical)"],
         "hop Ni-O2 (outer apical)": ["hop Ni-O2 (outer apical)"],
         "hop Ni-O in-plane": ["hop Ni-O in-plane"],
         "longer-range hops (Ni-O far, O-O, Ni-Ni)": ["hop Ni-O farther", "hop O-O", "hop Ni-Ni"],
         "onsite Ni d_z2": ["onsite Ni d_z2"], "onsite Ni other d": ["onsite Ni other d"],
         "onsite apical O (O1 + O2)": ["onsite O1_inner", "onsite O2_outer"], "onsite in-plane O": ["onsite O_inplane"]}
lines = ["# La3Ni2O7 第 1c 阶段：d_z² 成键带顶的 Wannier 分解（meV）", "",
         "带顶部分：Γ 点精确，按矩阵元类别分解（已去掉各次 VASP 计算的绝对参考平移，在位能变化相对平均值）。",
         "E_F 部分：总量取第 1b 阶段 VASP 读数（12×12×3，四面体）。R = 带顶 − E_F。", ""]
for interval in ("P1", "P2", "S1", "S2"):
    decomposition = json.loads((HERE / f"wannier/wannier3_{interval.lower()}.json").read_text())
    pairs, states = decomposition["pairs"], READOUT[f"runs/{interval}"]["states"]

    def top(p):
        t = pairs[p]["top_meV"]
        return ({g: sum(t[k] for k in ks) for g, ks in CLASS_GROUPS.items()},
                sum(t.values()) + 1e3 * pairs[p]["ref_shift_eV"])

    def vasp(a, b):
        return 1e3 * (states[b]["R"] - states[a]["R"]), 1e3 * (states[b]["efermi"] - states[a]["efermi"])

    cols = {"10-00": ("00", "10"), "11-01": ("01", "11"), "01-00": ("00", "01"), "11-10": ("10", "11"),
            "01ip-00": ("00", "01ip"), "01ap-00": ("00", "01ap")}
    table = {p: top(p) for p in cols}

    def avg(p, q):
        return ({g: (table[p][0][g] + table[q][0][g]) / 2 for g in CLASS_GROUPS}, (table[p][1] + table[q][1]) / 2)

    table["Δb"], table["Δc"] = avg("10-00", "11-01"), avg("01-00", "11-10")
    vasp_delta = {p: vasp(*ab) for p, ab in cols.items()}
    vasp_delta["Δb"] = tuple((vasp_delta["10-00"][i] + vasp_delta["11-01"][i]) / 2 for i in range(2))
    vasp_delta["Δc"] = tuple((vasp_delta["01-00"][i] + vasp_delta["11-10"][i]) / 2 for i in range(2))
    order = ["Δb", "Δc", "01ip-00", "01ap-00", "10-00", "11-01", "01-00", "11-10"]
    lines += [f"## {interval}（Γ 核对 {decomposition['gamma_check_meV']:.3f} meV，"
              f"6×6×2 核对 {decomposition['mesh_check_meV']:.3f} meV）", "",
              "| 带顶分解 | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for g in CLASS_GROUPS:
        lines.append(f"| {g} | " + " | ".join(f"{table[p][0][g]:+.1f}" for p in order) + " |")
    lines.append("| 参考平移（全体在位能） | " + " | ".join(
        f"{(table[p][1] - sum(table[p][0].values())):+.1f}" for p in order) + " |")
    lines.append("| **Δ带顶（模型）** | " + " | ".join(f"**{table[p][1]:+.1f}**" for p in order) + " |")
    lines.append("| Δ带顶（VASP） | " + " | ".join(f"{vasp_delta[p][0] + vasp_delta[p][1]:+.1f}" for p in order) + " |")
    lines.append("| ΔE_F（VASP） | " + " | ".join(f"{vasp_delta[p][1]:+.1f}" for p in order) + " |")
    lines.append("| **ΔR（VASP）** | " + " | ".join(f"**{vasp_delta[p][0]:+.1f}**" for p in order) + " |")
    if "00-start" in pairs:
        t, tot = top("00-start")
        lines += ["", f"仿射步（00 − 0 GPa 起点），带顶 {tot:+.1f}：" + "，".join(f"{g} {v:+.1f}" for g, v in t.items())]
    lines.append("")
(HERE / "wannier/wannier_table.md").write_text("\n".join(lines), encoding="utf8")
print("\n".join(lines))
