"""Spinel: hard-core rows and the random-forest descriptor selection.

A row is hard-core when its ordering margin is clear of the class boundary:
``margin >= tau_positive`` for co-located (y = 1) and ``margin >= tau_negative``
for separated (y = 0) rows. The pair of margins is chosen by a random forest:
for every pair on the grid, a forest is trained on the hard-core rows and
scored out of bag on them and externally on the soft rows; the best-scoring
pair defines the split. A forest on that split then ranks the descriptors
(impurity screen, then permutation on the soft rows) and a scan over the
number of top-ranked descriptors fixes the selected set.

Neither step removes descriptors from the search: the split decides where
thresholds are fitted and the selection only marks tier-A descriptors of the
sampling prior. Every forest uses a fixed seed and one thread, so the result
does not depend on the machine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 56
TAU_POSITIVE = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)
TAU_NEGATIVE = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)
MIN_HARD_ROWS, MIN_HARD_PER_CLASS, MIN_HARD_COMPONENTS_PER_CLASS = 100, 20, 3
THRESHOLD_TREES, RANKING_TREES, COUNT_TREES = 96, 96, 256
PERMUTATION_REPEATS, PERMUTATION_CANDIDATES = 2, 128
FEATURE_COUNTS = (16, 24, 32, 48, 64, 96, 128)
COUNT_TOLERANCE = 0.003


@dataclass(frozen=True)
class Rows:
    X: np.ndarray                 # descriptors, shape (sample, descriptor)
    y: np.ndarray                 # int8 labels
    margin: np.ndarray            # ordering margin in eV
    curves: np.ndarray            # curve id per row
    components: np.ndarray        # relation component id per row


@dataclass(frozen=True)
class Split:
    tau_positive: float
    tau_negative: float
    hard: np.ndarray              # bool mask over the rows
    selected: np.ndarray          # selected descriptor indices, in selection order


def _forest(trees: int, seed: int, n_features: int):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=trees, random_state=seed, n_jobs=1,
        class_weight="balanced_subsample", bootstrap=True, oob_score=True,
        max_features=max(1, int(round(math.sqrt(n_features)))), min_samples_leaf=1)


def _positive_score(model, probabilities: np.ndarray) -> np.ndarray:
    return np.asarray(probabilities, dtype=float)[:, list(model.classes_).index(1)]


def _bacc_and_recalls(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    from sklearn.metrics import balanced_accuracy_score, recall_score
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    return (float(balanced_accuracy_score(y, pred)),
            float(recall_score(y, pred, pos_label=1, zero_division=0)),
            float(recall_score(y, pred, pos_label=0, zero_division=0)))


def _macro_accuracy(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> float:
    """Mean over groups (sorted) of the per-group accuracy."""
    frame = pd.DataFrame({"group": groups, "y": np.asarray(y, int), "pred": np.asarray(pred, int)})
    per_group = [float(np.mean(g.y.to_numpy() == g.pred.to_numpy()))
                 for _, g in frame.groupby("group", sort=True, dropna=False)]
    return float(pd.Series(per_group, dtype=float).mean())


def _crossfit(rows: Rows, hard: np.ndarray, features: np.ndarray, trees: int,
              seed: int) -> dict[str, float]:
    """Train on the hard rows; out-of-bag scores on them, external scores on the soft rows."""
    model = _forest(trees, seed, len(features))
    Xh, yh = rows.X[hard][:, features], rows.y[hard]
    model.fit(Xh, yh)
    pred = (_positive_score(model, model.predict_proba(rows.X[:, features])) >= 0.5).astype(np.int8)
    oob = (_positive_score(model, model.oob_decision_function_) >= 0.5).astype(np.int8)
    pred[hard] = oob
    train = (_positive_score(model, model.predict_proba(Xh)) >= 0.5).astype(np.int8)
    bacc, positive, negative = _bacc_and_recalls(rows.y, pred)
    train_bacc = _bacc_and_recalls(yh, train)[0]
    oob_bacc = _bacc_and_recalls(yh, oob)[0]
    return {"bacc": bacc, "worst_recall": min(positive, negative),
            "curve_macro": _macro_accuracy(rows.y, pred, rows.curves),
            "component_macro": _macro_accuracy(rows.y, pred, rows.components),
            "oob_bacc": oob_bacc, "train_oob_gap": float(train_bacc - oob_bacc)}


def hard_mask(rows: Rows, tau_positive: float, tau_negative: float) -> np.ndarray:
    return np.isfinite(rows.margin) & (((rows.y == 1) & (rows.margin >= tau_positive))
                                       | ((rows.y == 0) & (rows.margin >= tau_negative)))


def choose_split(rows: Rows) -> tuple[float, float, np.ndarray]:
    """Margin pair with the best cross-fitted BACC among pairs with enough hard rows.

    Ties go to the higher worst-class recall, the hard-core class balance
    closest to all rows, the higher component and curve macro accuracy, more
    hard rows, then the larger margins.
    """
    all_features = np.arange(rows.X.shape[1], dtype=np.int32)
    positive_fraction = float(np.mean(rows.y == 1))
    candidates = []
    for tp in TAU_POSITIVE:
        for tn in TAU_NEGATIVE:
            hard = hard_mask(rows, tp, tn)
            yh, comp = rows.y[hard], rows.components[hard]
            pos, neg = int(np.sum(yh == 1)), int(np.sum(yh == 0))
            if (hard.sum() < MIN_HARD_ROWS or pos < MIN_HARD_PER_CLASS or neg < MIN_HARD_PER_CLASS
                    or len(set(comp[yh == 1])) < MIN_HARD_COMPONENTS_PER_CLASS
                    or len(set(comp[yh == 0])) < MIN_HARD_COMPONENTS_PER_CLASS):
                continue
            balance = abs(pos / max(1, int(hard.sum())) - positive_fraction)
            m = _crossfit(rows, hard, all_features, THRESHOLD_TREES,
                          SEED + int(tp * 1e7) + int(tn * 1e6))
            candidates.append(((-m["bacc"], -m["worst_recall"], balance, -m["component_macro"],
                                -m["curve_macro"], -int(hard.sum()), -tp, -tn), tp, tn, hard))
    if not candidates:
        raise ValueError("no margin pair leaves enough hard-core rows")
    _, tp, tn, hard = min(candidates, key=lambda c: c[0])
    return tp, tn, hard


def _grouped_score(y: np.ndarray, pred: np.ndarray, curves: np.ndarray) -> float:
    bacc = _bacc_and_recalls(y, pred)[0]
    return float(0.7 * bacc + (1.0 - 0.7) * _macro_accuracy(y, pred, curves))


def rank_descriptors(rows: Rows, hard: np.ndarray) -> np.ndarray:
    """Descriptor order: the impurity top 128 by permutation drop on the soft rows, then the rest."""
    model = _forest(RANKING_TREES, SEED + 7000001, rows.X.shape[1])
    model.fit(rows.X[hard], rows.y[hard])
    impurity = np.asarray(model.feature_importances_, dtype=float)
    candidates = np.argsort(-impurity, kind="stable")[:PERMUTATION_CANDIDATES]
    Xs, ys, curves = rows.X[~hard], rows.y[~hard], rows.curves[~hard]

    def predict(X: np.ndarray) -> np.ndarray:
        return (_positive_score(model, model.predict_proba(X)) >= 0.5).astype(np.int8)

    base = _grouped_score(ys, predict(Xs), curves)
    screened = []
    for i in map(int, candidates):
        drops = []
        for r in range(PERMUTATION_REPEATS):
            rng = np.random.default_rng(SEED + 8000009 + i * 1009 + r * 100003)
            Xp = Xs.copy()
            Xp[:, i] = Xp[rng.permutation(len(Xs)), i]
            drops.append(base - _grouped_score(ys, predict(Xp), curves))
        mean, std = float(np.mean(drops)), float(np.std(drops, ddof=1))
        screened.append(((-mean, std, -impurity[i], i), i))
    screened_ids = set(candidates.tolist())
    rest = [((-impurity[i], i), i) for i in range(len(impurity)) if i not in screened_ids]
    return np.asarray([i for _, i in sorted(screened)] + [i for _, i in sorted(rest)],
                      dtype=np.int32)


def select_descriptors(rows: Rows, hard: np.ndarray, ordered: np.ndarray) -> np.ndarray:
    """Smallest robust prefix of the ranking: within 0.003 of the best BACC, then
    the highest worst-class recall, OOB BACC, smallest train-OOB gap and count."""
    counts = sorted({c for c in FEATURE_COUNTS if c <= len(ordered)} | {len(ordered)})
    scans = [(c, _crossfit(rows, hard, ordered[:c], COUNT_TREES, SEED + 9000011 + c * 1009))
             for c in counts]
    best = max(m["bacc"] for _, m in scans)
    near = [(c, m) for c, m in scans if m["bacc"] >= best - COUNT_TOLERANCE - 1e-15]
    count = min(near, key=lambda cm: (-cm[1]["worst_recall"], -cm[1]["oob_bacc"],
                                      cm[1]["train_oob_gap"], cm[0]))[0]
    return ordered[:count]


def split_and_select(rows: Rows) -> Split:
    tp, tn, hard = choose_split(rows)
    selected = select_descriptors(rows, hard, rank_descriptors(rows, hard))
    return Split(tp, tn, hard, selected)
