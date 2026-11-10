# symbolic_regression

Two-level component–skeleton coevolutionary symbolic search for one scalar
descriptor that separates two classes with a single threshold. It produced
the spinel relation Π_S and the single- and double-perovskite relations of
*Local–network competition in crystals*, and, with edge labels, the
double-perovskite valence and conduction relations. A separate two-edge search
(`dual_edge.py`) produced the spinel valence and conduction coordinates; see
[Edge-resolved relations](#edge-resolved-relations).

## What the search does

* **Components** are short typed expressions (at most three descriptors)
  over robust-scaled descriptors. They start from every descriptor, every
  descriptor power and random pairs, and evolve by mutation and crossover.
  Descriptors are drawn with a prior from the guidance registry (for spinel,
  the random-forest-selected descriptors form the most-drawn tier; uniform
  for the perovskites); the prior only shapes proposals. A component survives by its *context utility*: how much it
  raises the balanced accuracy of other components when combined with them.
* **Skeletons** are typed trees over components (at most eight leaves, depth
  seven, two division-like nodes). They evolve in a protected baseline
  lineage. Every few generations a feedback cycle mutates components inside
  good skeletons; the rebuilt skeletons evolve in a side branch and rejoin
  the population only if they beat their parent.
* A score becomes a classifier by one threshold and orientation:
  `hard_core` fits it on the hard-core rows and evaluates all rows (spinel);
  `grouped_oof` fits it on the training folds of a parent-grouped split and
  scores every row out of fold (perovskites).
* At the end the archive leaders are re-evaluated, single deletions that
  lose less than one sample's worth of accuracy are kept as simpler
  formulas, and the top 30 are ranked.

Results do not depend on the number of workers: every proposal has its own
seed drawn from the parent stream, and results are merged in proposal order.

## Requirements

Linux (fork-based workers, memory-mapped files in `/dev/shm`), Python 3.11,
numpy, pandas, and scikit-learn for `prepare` and the tests. The published settings used
32–96 worker processes; plan on about 16 GB of `/dev/shm` and a sparse
component value file of up to 16 GB (2,000,000 components × samples ×
8 bytes) in the output directory. Run with one BLAS thread
(`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`).

## Computing the descriptors

```bash
python -m symbolic_regression describe --family spinel \
    --structures data/descriptor_tables/structures/spinel_structures.csv \
    --samples data/descriptor_tables/spinel/samples.csv --out spinel_descriptors.csv
```

computes a descriptor table from the structures (`inputs/spinel_descriptors.py`
for the spinels, `inputs/perovskite_descriptors.py` for `sp` and `dp`, which
read `perovskite_structures.csv`); `--samples` picks the rows and their order.
The results are the `descriptors.csv` of `data/descriptor_tables/` (see its
README for the four commands and for how other numpy builds differ).

## Preparing the inputs

```bash
python -m symbolic_regression prepare --family spinel \
    --raw data/descriptor_tables/spinel --out prepared/spinel
```

writes `prepared/spinel/protocol` (descriptor matrix, labels, row masks,
folds, row table) and `prepared/spinel/space` (descriptor domains, channel
and role labels, sampling prior) from the descriptor tables of one family
(`spinel`, `sp` or `dp`). These are byte-identical to the inputs the
published runs read (`tests/test_inputs.py` checks their digests).

* **Spinel** (442 development structures, 214 descriptors): a row is
  hard-core when its ordering margin clears the class boundary by τ₊ (y = 1)
  or τ₋ (y = 0). The pair is chosen on a grid by random forests trained on
  the hard-core rows and scored out of bag and on the soft rows
  (τ₊ = 0.0025 eV, τ₋ = 0.03 eV, 330 hard-core rows). A forest on that split
  ranks the descriptors and a scan over the number of top-ranked descriptors
  selects 48; they form tier A of the sampling prior. Descriptor domains
  follow the minima over the 1003-structure database.
* **SP** (136 structures, 111 descriptors) and **DP** (85 structures; 109 of
  182 descriptors pass the numerical-resolution guard): every row is
  hard-core, five folds are stratified by class and grouped by parent
  compound, and the prior is uniform.

## Running a published setting

```bash
python -m symbolic_regression run --config symbolic_regression/configs/spinel.json \
    --protocol prepared/spinel/protocol --space prepared/spinel/space --outdir runs/spinel
```

`--protocol` is the directory with the descriptor matrix, labels and splits of
one family, and `--space` the directory with its descriptor registry and
sampling prior; `spinel.json`, `sp.json` and `dp.json` hold the published
settings of the three families.

`--jobs` and `--task-size` override the worker settings of the config;
`--scratch` moves the temporary value files out of `/dev/shm`. A run writes a
checkpoint after every generation; continue an interrupted run with
`--resume` (the settings must be unchanged). Outputs: `top30.csv`,
`best_formula.json`, `components.csv`, the archives, per-generation
histories, out-of-fold scores for grouped scoring, and `run_record.json`
(settings, seed, input checksums, software versions).

## Replaying a formula

```bash
python -m symbolic_regression replay --config symbolic_regression/configs/dp.json \
    --protocol <protocol> --space <space> \
    --formulas data/number_sources/dp/best_performance_formula.json \
    --components data/number_sources/dp/components.csv \
    --scores dp_scores.csv
```

Replay rebuilds the components from their expressions, evaluates the
skeleton, refits the threshold and compares every stored field (formula id,
descriptor count, operator nodes, orientation, threshold, BACC, AUC, ...).
`--formulas` also accepts a `top30.csv`. A `value_dtype` column with
`float32` in the components file rounds that component to single precision
first (see below).

## How the published runs went

A fresh run reproduces the published *setting*, not the exact published
trajectory. The spinel run started in single precision; after skeleton
generation 30 its component values were converted to double precision and
the run continued, so components 0–199,999 of its registry hold
single-precision values (five of them occur in its top 30, two in Π_S). The
single-perovskite run was stopped after skeleton generation 17 and finalized
from that checkpoint (`sp.json` reproduces this with
`final_skeleton_generation`); the double-perovskite run ended by the
saturation stop at generation 88.

## Edge-resolved relations

Two relations per family, one for the valence and one for the conduction
band edge, each with its own threshold.

### Double perovskites

A row is labelled 1 when the extremum of that edge lies at the momentum k_D
of the lowest direct transition (`A_V_exact` / `A_C_exact` in
`data/perovskite_relations/dp_edge_targets.csv`). Everything else is the DP
search: same descriptors, folds (still stratified by the gap label),
registries and `dp.json`.

```bash
python -m symbolic_regression prepare --family dp --raw data/descriptor_tables/dp \
    --edge valence --edge-labels data/perovskite_relations/dp_edge_targets.csv --out prepared/dp_valence
python -m symbolic_regression run --config symbolic_regression/configs/dp.json \
    --protocol prepared/dp_valence/protocol --space prepared/dp_valence/space --outdir runs/dp_valence
```

(and the same with `conduction`). `replay ... --predictions FILE` writes the
out-of-fold predictions of replayed formulas, and

```bash
python -m symbolic_regression edges states --valence v_predictions.csv \
    --conduction c_predictions.csv --out four_states.csv
```

pairs them into the four states (k_D, k_D), (q_V, k_D), (k_D, q_C),
(q_V, q_C) and counts the states with both edges right (79 of 85 for the
published pair; out-of-fold BACC 1.000 and 0.919). The published DP edge
runs evaluated `np.power` with numpy 1.26; under numpy 2.x the component
values differ in the last bits (≤ 2·10⁻¹² absolute), which moves the fitted
thresholds in the 16th digit but no prediction.

### Spinel valence and conduction coordinates (`edges`)

`dual_edge.py` is a separate two-head search. Its label is "extremum at Γ"
per edge (path endpoints take the primitive-cell k point), and a formula is
scored by its AUC, its Spearman agreement with the explicit Γ-versus-non-Γ
energy difference S_V or S_C, and the simplicity of its fully expanded
expression; the AUC along substitution paths is only a gate (≥ 0.65). V and
C evolve separate skeleton populations over one shared component library that
grows from the descriptors used by good formulas of both edges; Π_S (rescaled
by its IQR) is offered as a terminal, and formulas with and without it evolve
side by side. There is no cross-validation; all 440 development states are
used.

```bash
python -m symbolic_regression prepare --family spinel-edges \
    --raw data/descriptor_tables/spinel_edges --out prepared/spinel_edges
P=prepared/spinel_edges; R=data/spinel_relations/colocation_relation
EDGE_IN="--protocol $P/protocol --samples $P/samples.csv --kpoints $P/band_edges.csv \
    --pi-s-formula $R/best_performance_formula.json --pi-s-scales $R/feature_scales.npy \
    --prior-valence $P/prior_valence.csv --prior-conduction $P/prior_conduction.csv"
python -m symbolic_regression edges run --config symbolic_regression/configs/spinel_edges.json \
    $EDGE_IN --outdir runs/spinel_edges
```

`prepare` writes `protocol/` (`arrays/X_full_clean_float64.npy` and
`arrays/sample_id_unicode.npy` of all 681 retained states, since the Π_S
regularizers use every row, and `features/formula_full_feature_names.json`;
byte-identical to the files the published run read), the discovery rows in
search order (`samples.csv`), the band-edge table (`band_edges.csv`) and the
random-forest importances of the two edges. The run writes the top lists `top30_valence.csv`
and `top30_conduction.csv` (chosen independently), objective champions,
simple near-optimal formulas, the hall of fame, histories, the component
library, `pair_audit.csv` (joint states of all pairings, diagnostic only) and
the per-state scores of the recommended pair. With the published settings
and seed 56 it reproduces the published search run byte for byte.
`edges replay --formulas top30_valence.csv --components components.json`
re-evaluates saved formulas.

The published Π_V and Π_C are not raw search output. `edges finalize`
(`edge_finalize.py`) runs the steps after the search:

```bash
python -m symbolic_regression edges finalize --config symbolic_regression/configs/spinel_edges.json \
    $EDGE_IN --finalize symbolic_regression/configs/spinel_edge_finalize.json \
    --run runs/spinel_edges --outdir runs/spinel_edges_final
```

1. *Selection*: the numerically safe formulas of each hall of fame
   (`safe_pool_*.csv`) and four named ones (`key_formulas_*.csv`).
2. *Pruning*: a seed formula per edge, written over descriptors with Π_S as
   one terminal, loses single nodes while it keeps its AUC (−0.015), margin
   agreement (−0.08), safety and path gate (`pruning_*.csv`).
3. *Coefficients*: a fixed sum of terms from the pruning set gets its
   coefficients from a grid of simple fractions and a dense grid around the
   best (`coefficients_*.csv`, `coefficient_roles_*.csv`).
4. *Freeze*: the final coefficients give `edge_coordinates.json`, evaluated
   with protected-ratio offsets 10⁻⁶ × the development upper quartile of each
   denominator (the search uses 10⁻⁹) and Π_S with its offsets fitted on the
   development states (steps 1–3 use this Π_S as well).

Two choices were made by the authors, not by the steps, and are recorded as
such in `configs/spinel_edge_finalize.json`: the structures passed to the
coefficient step are the six-descriptor members of the pruning set, which keep
the terms −(1/6)·Var(I₁)·f²_Bd (V) and −(1/3)·Var(Z)² (C) that the pruning's
own recommendation drops; and Π_C keeps the coefficients (−1/3, 2, 1) of that
member instead of the scan's recommendation (−1/2, 3, 1). From the published
run this writes `edge_coordinates.json` byte-identical to
`configs/spinel_edge_coordinates.json`, which holds the two fixed expressions
and their stored metrics; `edges replay --formulas configs/spinel_edge_coordinates.json`
reproduces them (AUC 0.894 / 0.921, thresholds −9.038 / 42.36).
