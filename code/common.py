"""Data paths and small drawing helpers shared by the figure modules."""
import json
from pathlib import Path
import numpy as np

from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
_D = ROOT / "data"

DATA = {
    "wannier_endpoint_audit": _D / "wannier_channel_response" / "hamiltonian_45_channel_responses.csv",
    "wpart_projected": _D / "spinel_wannier_soc" / "results" / "wannier_proj_converged" / "wpart.json",
    "wpart_mlwf": _D / "spinel_wannier_soc" / "results" / "wannier_mlwf_converged" / "wpart.json",
    "soc_margins": _D / "spinel_wannier_soc" / "results" / "soc_margins.csv",
    "tight_summary": _D / "spinel_pressure_path" / "tight_summary.json",
    "path_edges": _D / "spinel_pressure_path" / "path_edges.json",
    "fatband_p4": _D / "spinel_fatbands" / "fatband_p4.npz",
    "fatband_p5": _D / "spinel_fatbands" / "fatband_p5.npz",
    "wavefunction_gamma": _D / "rendered" / "cb_gamma_sc24.png",
    "wavefunction_r": _D / "rendered" / "cb_r_sb20.png",
    "leverage_prepare": _D / "four_state_leverage" / "prepare_report.json",
    "leverage_evaluation": _D / "four_state_leverage" / "evaluation.json",
    "three_channel": _D / "three_channel" / "tc_result.json",
    "three_channel_all": _D / "three_channel" / "all_network" / "tca_result.json",
    "polarization": _D / "polarization" / "berry_phase" / "polarization_result.json",
    "bending_amplitude": _D / "bending_amplitude" / "amplitude.json",
    "tilt_vs_pressure": _D / "tilt_pressure" / "tilt_vs_p.txt",
    "ordering_margins": _D / "spinel_relations" / "spinel_sc_sb_ordering_margins.csv",
    "path_split": _D / "substitution_path_split" / "path_split.json",
    "optical_pair": _D / "optics" / "optical_pair_aligned_common_grid.csv.gz",
    "feature_ablation": _D / "spinel_relations" / "sr_singlecoord_feature_ablation_mapped.csv",
    "group_neutralisation": _D / "relation_ablation" / "neutralisation_ablation.csv",
    "path_terms": _D / "relation_ablation" / "t2c_path_terms.csv",
    "dp_out_of_fold": _D / "perovskite_relations" / "dp_dual_oof_4state.csv",
    "la327_p0": _D / "la3ni2o7" / "drive_plane_grid" / "p0_contcar",
    "la327_p2": _D / "la3ni2o7" / "drive_plane_grid" / "p2_contcar",
    "la327_film": _D / "la3ni2o7" / "drive_plane_grid" / "film_contcar",
    "la327_orbital": _D / "la3ni2o7" / "orbital" / "slices.npz",
    "energy_bound": _D / "field_sectors" / "energy_bound.json",
}


def read_json(key):
    return json.loads(DATA[key].read_text())


def arrow(ax, p0, p1, color, lw=1.2, ms=8, ls="-", z=4):
    if ls in ("-", "solid"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                     shrinkA=0, shrinkB=0, zorder=z))
        return
    # dashed or dotted shaft with a solid head: a dash pattern on the head outline breaks the head
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-", color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0.6 * ms, zorder=z))
    p0, p1 = (np.asarray(p0, float), np.asarray(p1, float))
    ax.add_patch(FancyArrowPatch(p0 + 0.98 * (p1 - p0), p1, arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, shrinkA=0, shrinkB=0, zorder=z))
