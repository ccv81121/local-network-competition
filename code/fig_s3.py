"""Supplementary Fig. 3: near-edge absorption at 2 and 5 GPa and its integrated gain over 0-0.3 eV."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from common import DATA


def draw():
    p = pd.read_csv(DATA["optical_pair"])
    p = p[(p.profile == "c005") & (p.pair == "pressure_conduction") & (p.window_eV == 0.3) & (p.grid_phase == "primary")
          & (p.metric == "absorption")].sort_values("relative_energy_eV")
    e = p.relative_energy_eV.to_numpy()
    alpha_2gpa, alpha_5gpa = p.separated_value.to_numpy(), p.colocated_value.to_numpy()
    gain_pct = 100 * (np.trapezoid(alpha_5gpa, e) / np.trapezoid(alpha_2gpa, e) - 1)
    fig = plt.figure(figsize=(3.4, 2.4))
    ax = fig.add_axes([0.20, 0.19, 0.76, 0.76])
    scale = 1e-4                                  # in units of 10^4 cm^-1
    ax.plot(e, alpha_2gpa * scale, color="black", lw=1.1, ls=(0, (3, 1.6)))
    ax.plot(e, alpha_5gpa * scale, color="black", lw=1.3)
    ax.text(e[-1] - 0.005, alpha_5gpa[-1] * scale, "5 GPa", ha="right", va="bottom")
    i = np.searchsorted(e, 0.255)
    ax.annotate("2 GPa", (e[i], alpha_2gpa[i] * scale), xytext=(3, -5), textcoords="offset points", ha="left",
                va="top")
    ax.text(0.02, 0.96, r"$\mathrm{\Delta}I_\alpha$" + f" = +{gain_pct:.1f}% over 0–0.3 eV", transform=ax.transAxes,
            ha="left", va="top")
    ax.set_xlim(0, 0.3)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel(r"$\hbar\omega-E_g$ (eV)")
    ax.set_ylabel(r"Absorption (10$^{\mathrm{4}}$ cm$^{-\mathrm{1}}$)")
    return fig


def make(outdir):
    with style.use():
        style.save(draw(), outdir, "fig_s3")
