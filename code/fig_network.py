"""Nature Physics network figure: a-c field of the Ti->Sn substitution (as fig_field), d symmetry sectors of CaTiO3
(Fig. 2d), e octahedral tilts under pressure (Fig. 2c)."""
import matplotlib.pyplot as plt

import fig_field
import style
import fig2
from fig2 import panel_sectors, panel_tilts

fig2.NP_LABELS = True


def draw():
    W, H = 7.0, 4.95
    low = 2.25                                      # height of the bottom row (inches); field panels sit above it
    fig = plt.figure(figsize=(W, H))
    fig_field.draw(fig=fig, H=H, y_shift=low)
    box = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    h, y = 1.62, 0.50
    axd = box(0.58, y, 2.62, h)
    axe = box(4.18, y, 2.62, h)
    panel_sectors(axd)
    panel_tilts(axe)
    for lab, x in (("d", 0.0), ("e", 3.62)):
        style.letter(fig, x / W, (y + h + 0.12) / H, lab)
    return fig


def make(outdir):
    with style.use():
        style.save(draw(), outdir, "fig_network")
