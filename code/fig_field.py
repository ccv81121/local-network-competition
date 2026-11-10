"""Fig. 3: the field of a Ti->Sn substitution in the 80-atom CaTiO3 cell, drawn atom by atom on the TiO2 layer through
Sn as continuous maps: a, force f; b, relaxed displacement u; c, its bond-preserving part. Colour: smoothed per-atom
in-plane magnitude; lines: streamlines where neighbouring atoms move together."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import render_fields
import style

# field, colour ramp, streamline colour, title, colour-bar label, ticks
PANELS = [("f", ["#ffffff", "#F3E2B8", "#D4A02F"], "#5a4210", "force $f$", r"$|f|$ (eV/Å)", [0, 1, 2]),
          ("u_act", ["#ffffff", "#D9D6D0", "#6B665F"], "#1F1D1B", "displacement $u$", r"$|u|$ (Å)", [0, 0.04, 0.08]),
          ("u_act_b", ["#ffffff", "#C9D7EA", "#2C64A4"], "#14325a", "bond-preserving part of $u$",
           r"$|u_b|$ (Å)", [0, 0.01, 0.02])]
S_U_PCT = 24.8      # relaxed-displacement share of the bond-preserving part (dispersion_result.json s_u_actual_pct)


def _nearest_o(field):
    """Full magnitude of `field` on the O atom nearest to Sn, and its in-plane position."""
    z = np.load(render_fields.SUBSTITUTION)
    fr, L, sp, sn = z["frac"], z["lattice"], z["species"], int(z["sn"])
    d = fr - fr[sn]
    d -= np.round(d)
    r = np.linalg.norm(d @ L, axis=1)
    r[(sp != "O") | (np.abs((d @ L)[:, 1]) > 1.0)] = 1e9          # O of the drawn layer (|y| < 1 A)
    j = int(np.argmin(r))
    return float(np.linalg.norm(z[field][j])), (d[j] @ L)[[0, 2]]


def draw(fig=None, H=2.78, y_shift=0.0):
    """The three field panels; `fig`, `H` and `y_shift` (inches) let another figure carry them as its top row."""
    from matplotlib.colors import LinearSegmentedColormap
    W = 7.0
    if fig is None:
        fig = plt.figure(figsize=(W, H))
    P, pitch, y0 = 2.05, 2.42, 0.52 + y_shift
    for k, (field, cols, line, title, clab, ticks) in enumerate(PANELS):
        x0 = 0.05 + pitch * k
        ax = fig.add_axes([x0 / W, y0 / H, P / W, P / H])
        im, vmax, frac = render_fields.field_continuous(ax, field, LinearSegmentedColormap.from_list(field, cols), line)
        print(f"fig_field panel {'abc'[k]} ({field}): max in-plane = {vmax:.5f}; streamline area {frac:.2f}")
        cw = 1.30                                                  # colour bar centred under its own panel
        cax = fig.add_axes([(x0 + (P - cw) / 2) / W, (0.36 + y_shift) / H, cw / W, 0.055 / H])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
        cb.outline.set_linewidth(0.4)
        cb.set_ticklabels([f"{t:g}" for t in ticks])
        cax.tick_params(length=1.5, width=0.4, labelsize=6.0, pad=1)
        cb.set_label(clab, fontsize=6.5, labelpad=1.5)
        if k == 0:                                                 # 2 A scale bar between the bottom Ti
            ax.plot([-3.75, -1.75], [-5.75, -5.75], color="#1F1D1B", lw=1.3, solid_capstyle="butt", zorder=6)
            ax.text(-2.75, -5.6, "2 Å", ha="center", va="bottom", fontsize=6.5, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
        if k == 2:
            title = title + f", $s_u$ = {S_U_PCT:.1f}%"
        ax.set_title(title, fontsize=style.TEXT_PT, pad=2)
        style.letter(fig, (0.0 + pitch * k) / W, (y0 + P + 0.17) / H, "abc"[k])
    return fig


def make(outdir):
    with style.use():
        style.save(draw(), outdir, "fig_field")
