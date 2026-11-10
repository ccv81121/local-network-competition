"""Regenerate the figures of the Nature Physics manuscript and its Supplementary Information.

usage: python code/make_figures.py [FIGURE ...] [--outdir DIR]
FIGURE is Fig1 ... Fig4 or SFig1 ... SFig6 (default: all). Without --outdir, main-text figures go to paper/figures and
SI figures to paper/supplementary/figures, or to figures/ when there is no paper/ folder.

Figure (as published)      module             output file
Fig. 1                     fig1               fig1
Fig. 2                     fig_network        fig_network
Fig. 3                     fig4               fig4
Fig. 4                     fig3               fig3
Supplementary Fig. 1       fig_s5             fig_s5
Supplementary Fig. 2       fig_s1             fig_s1
Supplementary Fig. 3       fig_s4             fig_s4
Supplementary Fig. 4       fig_s2             fig_s2
Supplementary Fig. 5       fig5               fig_s6
Supplementary Fig. 6       fig_s3             fig_s3
"""
import argparse
import importlib
from pathlib import Path

from common import ROOT

FIGS = {  # published name: (module, function, main text or SI)
    "Fig1": ("fig1", "make", "main"), "Fig2": ("fig_network", "make", "main"),
    "Fig3": ("fig4", "make", "main"), "Fig4": ("fig3", "make", "main"),
    "SFig1": ("fig_s5", "make", "si"), "SFig2": ("fig_s1", "make", "si"), "SFig3": ("fig_s4", "make", "si"),
    "SFig4": ("fig_s2", "make", "si"), "SFig5": ("fig5", "make", "si"), "SFig6": ("fig_s3", "make", "si"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figures", nargs="*", choices=list(FIGS), metavar="FIGURE")
    parser.add_argument("--outdir", type=Path, help="write all figures into this directory")
    args = parser.parse_args()
    for name in args.figures or list(FIGS):
        module, func, part = FIGS[name]
        if args.outdir is not None:
            outdir = args.outdir
        elif (ROOT / "paper").is_dir():
            outdir = ROOT / "paper" / ("figures" if part == "main" else "supplementary/figures")
        else:
            outdir = ROOT / "figures"
        getattr(importlib.import_module(module), func)(outdir)
        print(f"{name} ({module}.{func}) -> {outdir}")


if __name__ == "__main__":
    main()
