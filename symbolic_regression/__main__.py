"""Command line: ``python -m symbolic_regression describe|prepare|run|replay|edges``."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

from .settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m symbolic_regression",
                                     description="Single-coordinate and two-edge symbolic "
                                                 "search and replay of saved formulas.")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="build the protocol and space directories "
                                             "from the descriptor tables of one family")
    prepare.add_argument("--family", choices=("spinel", "sp", "dp", "spinel-edges"),
                         required=True)
    prepare.add_argument("--raw", type=Path, required=True,
                         help="descriptor tables of the family (data/descriptor_tables/<family>, "
                              "spinel_edges for spinel-edges)")
    prepare.add_argument("--out", type=Path, required=True,
                         help="new directory; receives protocol/ and space/ (spinel-edges: "
                              "protocol/ and the band-edge, sample and prior tables)")
    prepare.add_argument("--edge", choices=("valence", "conduction"),
                         help="DP only: label each row by this band edge instead of the gap")
    prepare.add_argument("--edge-labels", type=Path,
                         help="table with A_V_exact / A_C_exact per sample_id "
                              "(data/perovskite_relations/dp_edge_targets.csv)")

    describe = sub.add_parser("describe", help="compute the descriptor table of one family "
                                               "from the structures")
    describe.add_argument("--family", choices=("spinel", "sp", "dp"), required=True)
    describe.add_argument("--structures", type=Path, required=True,
                          help="structure table (data/descriptor_tables/structures/"
                               "spinel_structures.csv or perovskite_structures.csv)")
    describe.add_argument("--elements", type=Path,
                          help="spinel: directory with elements_primary.csv and "
                               "elements_extended.csv (default: that of --structures)")
    describe.add_argument("--samples", type=Path,
                          help="CSV whose sample_id column lists the rows, in order "
                               "(default: every structure of the family)")
    describe.add_argument("--out", type=Path, required=True, help="descriptors.csv to write")

    def inputs(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", type=Path, required=True, help="settings file (configs/*.json)")
        p.add_argument("--protocol", type=Path, required=True,
                       help="directory with RF_PROTOCOL_BUNDLE/ and protocol_rows.csv")
        p.add_argument("--space", type=Path, required=True,
                       help="directory with the terminal, role and guidance registries")

    run = sub.add_parser("run", help="run, resume or finish a search")
    inputs(run)
    run.add_argument("--outdir", type=Path, required=True)
    run.add_argument("--jobs", type=int, help="worker processes (default from the config)")
    run.add_argument("--task-size", type=int, help="smallest batch handed to a worker")
    run.add_argument("--scratch", type=Path,
                     help="directory for temporary value files (default /dev/shm)")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="continue from the checkpoint")
    mode.add_argument("--overwrite", action="store_true", help="replace a non-empty outdir")
    run.add_argument("--stop-after-skeleton-generation", type=int, metavar="N",
                     help="checkpoint and stop after skeleton generation N")

    replay = sub.add_parser("replay", help="re-evaluate saved formulas and compare")
    inputs(replay)
    replay.add_argument("--formulas", type=Path, required=True,
                        help="formula record(s): best_formula.json, top30.csv, ...")
    replay.add_argument("--components", type=Path, required=True,
                        help="components.csv holding the components the formulas use")
    replay.add_argument("--scores", type=Path, help="write per-sample scores to this CSV")
    replay.add_argument("--predictions", type=Path,
                        help="grouped scoring: write per-sample out-of-fold predictions")
    replay.add_argument("--report", type=Path, help="write the comparison to this JSON file")

    edges = sub.add_parser("edges", help="two-edge (valence and conduction) search")
    esub = edges.add_subparsers(dest="edges_command", required=True)

    def edge_inputs(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", type=Path, required=True,
                       help="settings file (configs/spinel_edges.json)")
        p.add_argument("--protocol", type=Path, required=True,
                       help="directory with arrays/X_full_clean_float64.npy, "
                            "arrays/sample_id_unicode.npy and features/")
        p.add_argument("--samples", type=Path, required=True,
                       help="CSV with the sample_id of every discovery row, in order")
        p.add_argument("--kpoints", type=Path, required=True,
                       help="band-edge table: VBM/CBM k points (primitive cell for path "
                            "endpoints) and Γ-versus-non-Γ margins per sample")
        p.add_argument("--pi-s-formula", type=Path,
                       help="best_performance_formula.json of the single-coordinate relation")
        p.add_argument("--pi-s-scales", type=Path, help="its descriptor scales (.npy)")
        p.add_argument("--prior-valence", type=Path,
                       help="random-forest importances of the valence edge (feature, importance)")
        p.add_argument("--prior-conduction", type=Path,
                       help="random-forest importances of the conduction edge")

    erun = esub.add_parser("run", help="run or resume a two-edge search")
    edge_inputs(erun)
    erun.add_argument("--outdir", type=Path, required=True)
    erun.add_argument("--jobs", type=int, help="worker processes (default from the config)")
    emode = erun.add_mutually_exclusive_group()
    emode.add_argument("--resume", action="store_true", help="continue from the checkpoint")
    emode.add_argument("--overwrite", action="store_true", help="replace a non-empty outdir")
    erun.add_argument("--stop-after-generation", type=int, metavar="N",
                      help="checkpoint and stop after generation N")

    ereplay = esub.add_parser("replay", help="re-evaluate saved edge formulas and compare")
    edge_inputs(ereplay)
    ereplay.add_argument("--formulas", type=Path, required=True,
                         help="top30_valence.csv, recommended_formulas.json or a coordinate file")
    ereplay.add_argument("--components", type=Path,
                         help="components.json of the run (for formulas over components)")
    ereplay.add_argument("--scores", type=Path, help="write per-sample scores to this CSV")
    ereplay.add_argument("--report", type=Path, help="write the comparison to this JSON file")

    efin = esub.add_parser("finalize", help="selection, pruning, coefficients and freeze: "
                                            "from a finished search to the fixed coordinates")
    edge_inputs(efin)
    efin.add_argument("--finalize", type=Path, required=True,
                      help="post-search config (configs/spinel_edge_finalize.json)")
    efin.add_argument("--run", type=Path, required=True,
                      help="output directory of edges run (elite_*.csv, components.json)")
    efin.add_argument("--outdir", type=Path, required=True)

    states = esub.add_parser("states", help="four band-edge states from the out-of-fold "
                                            "predictions of a valence and a conduction relation")
    states.add_argument("--valence", type=Path, required=True,
                        help="out-of-fold predictions of the valence relation "
                             "(replay --predictions or top30_oof_predictions.csv)")
    states.add_argument("--conduction", type=Path, required=True)
    states.add_argument("--rank", type=int, default=1, help="formula rank to use (default 1)")
    states.add_argument("--out", type=Path, help="write the per-state table to this CSV")
    return parser


def _print_reports(reports: list[dict]) -> None:
    for r in reports:
        status = "reproduced" if r["all_equal"] else "DIFFERS"
        print(f"rank {r['rank']}: {status}  {r['formula_id']}")
        for field, check in r["checks"].items():
            if not check["equal"]:
                print(f"    {field}: stored {check['stored']!r}, replayed {check['replayed']!r}")


def edges_main(args: argparse.Namespace) -> int:
    from . import dual_edge as de
    if args.edges_command == "states":
        table, summary = de.four_states(args.valence, args.conduction, args.rank)
        if args.out:
            table.to_csv(args.out, index=False, float_format="%.17g")
        print(json.dumps(summary, indent=2))
        return 0
    inputs = {"protocol": args.protocol, "samples": args.samples, "kpoints": args.kpoints,
              "pi_s_formula": args.pi_s_formula, "pi_s_scales": args.pi_s_scales,
              "prior_valence": args.prior_valence, "prior_conduction": args.prior_conduction}
    if args.edges_command == "run":
        settings = de.load_edge_settings(args.config, jobs=args.jobs)
        record = de.run(settings, inputs, args.outdir, resume=args.resume,
                        overwrite=args.overwrite, stop_after_generation=args.stop_after_generation)
        return 0 if record["status"] in {"finalized", "stopped"} else 1
    settings = de.load_edge_settings(args.config)
    if args.edges_command == "finalize":
        from . import edge_finalize as ef
        # every post-search step uses Π_S with its offsets fitted on the discovery rows
        data = de.load_edge_data(args.protocol, args.samples, args.kpoints, settings,
                                 pi_s_formula=args.pi_s_formula, pi_s_scales=args.pi_s_scales,
                                 pi_s_fit_on_samples=True)
        summary = ef.finalize(ef.load_config(args.finalize), data, args.run, args.outdir)
        print(json.dumps({e: s["coefficients"] for e, s in summary.items()}, indent=2))
        return 0
    records, conventions = de.load_edge_formulas(args.formulas)
    data = de.load_edge_data(args.protocol, args.samples, args.kpoints, settings,
                             pi_s_formula=args.pi_s_formula, pi_s_scales=args.pi_s_scales,
                             pi_s_fit_on_samples=bool(conventions.get("pi_s_fit_on_samples")))
    components = (json.loads(args.components.read_text(encoding="utf-8"))
                  if args.components else None)
    reports, scores = de.replay(records, data, components)
    _print_reports(reports)
    if args.scores:
        scores.to_csv(args.scores, index=False, float_format="%.17g")
    if args.report:
        args.report.write_text(json.dumps(reports, indent=2, default=str) + "\n",
                               encoding="utf-8")
    return 0 if all(r["all_equal"] for r in reports) else 1


def describe_main(args: argparse.Namespace) -> int:
    import pandas as pd
    samples = (pd.read_csv(args.samples, usecols=["sample_id"]).sample_id.astype(str).tolist()
               if args.samples else None)
    if args.family == "spinel":
        from .inputs import spinel_descriptors as sd
        elements = args.elements or args.structures.parent
        table = sd.descriptor_table(
            sd.read_structures(args.structures),
            sd.load_elements(elements / "elements_primary.csv", elements / "elements_extended.csv"),
            samples)
    else:
        from .inputs import perovskite_descriptors as pv
        table = pv.descriptor_table(pd.read_csv(args.structures), args.family, samples)
    table.to_csv(args.out, index=False, encoding="utf-8", lineterminator="\n")
    print(json.dumps({"family": args.family, "rows": len(table),
                      "descriptors": table.shape[1] - 1, "out": str(args.out)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "edges":
        # The two-edge search keeps numpy's default warnings: overflowing
        # candidates become non-finite and are rejected by their metrics.
        return edges_main(args)
    if args.command == "describe":
        return describe_main(args)
    # Warnings are errors, as in the published single-coordinate runs: an
    # expression whose evaluation or diagnostics overflow is rejected.
    warnings.simplefilter("error")
    if args.command == "prepare":
        from .inputs import prepare
        print(json.dumps(prepare(args.family, args.raw, args.out, args.edge, args.edge_labels),
                         indent=2))
        return 0
    if args.command == "run":
        from .search import run
        settings = load_settings(args.config, jobs=args.jobs, task_size=args.task_size)
        record = run(settings, args.protocol, args.space, args.outdir, resume=args.resume,
                     overwrite=args.overwrite, scratch=args.scratch,
                     stop_after_skeleton_generation=args.stop_after_skeleton_generation)
        return 0 if record["status"] in {"finalized", "stopped"} else 1

    from .data import load_data
    from .replay import load_formulas, oof_predictions, replay
    settings = load_settings(args.config)
    data = load_data(args.protocol, args.space, settings.scoring)
    records = load_formulas(args.formulas)
    reports, scores = replay(records, args.components, data, settings)
    _print_reports(reports)
    if args.scores:
        scores.to_csv(args.scores, index=False, float_format="%.17g")
    if args.predictions:
        oof_predictions(records, reports, scores, data).to_csv(
            args.predictions, index=False, float_format="%.17g")
    if args.report:
        args.report.write_text(json.dumps(reports, indent=2, default=str) + "\n",
                               encoding="utf-8")
    return 0 if all(r["all_equal"] for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
