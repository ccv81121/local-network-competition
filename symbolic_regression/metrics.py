"""Threshold fitting and classification metrics for single-coordinate scores.

A candidate score is turned into a classifier by one threshold and one
orientation.  Two protocols are supported:

``hard_core``
    the threshold is fitted on the hard-core rows and the classifier is
    evaluated on all rows (spinel);
``grouped_oof``
    the threshold is fitted on the training folds of a parent-grouped
    split and every row is scored out of fold (single and double
    perovskites).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .grammar import MAX_ABS_OUTPUT

HARD_CORE = "hard_core"
GROUPED_OOF = "grouped_oof"
SCORING_MODES = (HARD_CORE, GROUPED_OOF)


def _lex_mask_argmax(metrics: Sequence[np.ndarray], valid: np.ndarray) -> np.ndarray:
    mask = valid.copy()
    for metric in metrics:
        masked = np.where(mask, metric, -np.inf)
        best = np.max(masked, axis=1)
        if np.issubdtype(metric.dtype, np.integer):
            mask &= metric == best[:, None]
        else:
            mask &= np.isclose(metric, best[:, None], rtol=0, atol=1e-15, equal_nan=False)
    return np.argmax(mask, axis=1)


def fit_threshold_batch(scores: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Best threshold and orientation for every row of ``scores``.

    Candidates are ranked by balanced accuracy, then worst-class recall, a
    non-degenerate cut, the smaller side, the margin to the neighbouring
    scores, a threshold close to zero and finally the earliest cut.
    """
    S = np.asarray(scores, dtype=np.float64)
    yt = np.asarray(y, dtype=np.int8)
    if S.ndim == 1:
        S = S[None, :]
    B, H = S.shape
    P = int(np.sum(yt == 1))
    N = int(np.sum(yt == 0))
    if P == 0 or N == 0:
        raise ValueError("threshold fitting needs both classes")
    order = np.argsort(S, axis=1, kind="stable")
    ss = np.take_along_axis(S, order, axis=1)
    sy = yt[order]
    pp = np.concatenate([np.zeros((B, 1), dtype=np.int32),
                         np.cumsum(sy == 1, axis=1, dtype=np.int32)], axis=1)
    k = np.arange(H + 1, dtype=np.int32)[None, :]
    pn = k - pp
    valid = np.ones((B, H + 1), dtype=bool)
    if H > 1:
        valid[:, 1:H] = ss[:, :-1] < ss[:, 1:]
    tp1 = P - pp
    tn1 = pn
    fp1 = (H - k) - tp1
    fn1 = pp
    tp2 = pp
    tn2 = N - pn
    fp2 = k - tp2
    fn2 = P - tp2

    def cat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.concatenate([a, b], axis=1)

    tp, tn, fp, fn = [cat(a, b).astype(np.int32)
                      for a, b in ((tp1, tp2), (tn1, tn2), (fp1, fp2), (fn1, fn2))]
    vv = cat(valid, valid)
    primary = tp.astype(np.int64) * N + tn.astype(np.int64) * P
    worst = np.minimum(tp.astype(np.int64) * N, tn.astype(np.int64) * P)
    inner = np.broadcast_to((k > 0) & (k < H), (B, H + 1))
    nondeg = cat(inner, inner).astype(np.int64)
    side = np.broadcast_to(np.minimum(k, H - k), (B, H + 1))
    minside = cat(side, side).astype(np.int64)
    th = np.empty((B, H + 1), dtype=float)
    th[:, 0] = np.nextafter(ss[:, 0], -np.inf)
    th[:, H] = np.nextafter(ss[:, -1], np.inf)
    margin = np.zeros_like(th)
    if H > 1:
        th[:, 1:H] = ss[:, :-1] + (ss[:, 1:] - ss[:, :-1]) / 2
        margin[:, 1:H] = ss[:, 1:] - ss[:, :-1]
    scale = np.quantile(np.abs(S), .75, axis=1)
    scale = np.where(scale > 1e-15, scale, np.maximum(np.max(np.abs(S), axis=1), 1.0))
    pref = -np.abs(th) / scale[:, None]
    th2, mg2, pref2 = cat(th, th), cat(margin, margin), cat(pref, pref)
    cuts = np.broadcast_to(k, (B, H + 1))
    cut = cat(cuts, cuts)
    idx = _lex_mask_argmax([primary, worst, nondeg, minside, mg2, pref2,
                            -cut.astype(float)], vv)
    r = np.arange(B)
    return {"threshold": th2[r, idx], "orientation": np.where(idx < H + 1, 1, -1).astype(np.int8),
            "tp": tp[r, idx], "tn": tn[r, idx], "fp": fp[r, idx], "fn": fn[r, idx],
            "bacc": primary[r, idx] / (2.0 * P * N), "worst_recall": worst[r, idx] / (P * N),
            "margin": mg2[r, idx]}


def evaluate_fixed_batch(scores: np.ndarray, y: np.ndarray, threshold: np.ndarray,
                         orientation: np.ndarray) -> dict[str, np.ndarray]:
    S = np.asarray(scores, dtype=float)
    yy = np.asarray(y, dtype=np.int8)
    if S.ndim == 1:
        S = S[None, :]
    pred = np.where(orientation[:, None] == 1, S >= threshold[:, None], S <= threshold[:, None])
    pos = yy == 1
    neg = yy == 0
    P = max(1, int(pos.sum()))
    N = max(1, int(neg.sum()))
    tp = np.sum(pred[:, pos], axis=1)
    tn = np.sum(~pred[:, neg], axis=1)
    fp = N - tn
    fn = P - tp
    pr = tp / P
    nr = tn / N
    bacc = .5 * (pr + nr)
    acc = (tp + tn) / (P + N)
    f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    den = np.sqrt(np.maximum(1, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / den
    return {"prediction": pred, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "positive_recall": pr, "negative_recall": nr, "worst_recall": np.minimum(pr, nr),
            "bacc": bacc, "acc": acc, "f1": f1, "mcc": mcc}


def safe_auc(y: np.ndarray, score: np.ndarray, orientation: int) -> float:
    """Binary ROC AUC from average ranks; NaN when undefined."""
    try:
        yy = np.asarray(y, dtype=np.int8)
        x = np.asarray(score, dtype=np.float64) * int(orientation)
        if not np.all(np.isfinite(x)):
            return float("nan")
        p = int(np.sum(yy == 1))
        n = int(np.sum(yy == 0))
        if p == 0 or n == 0:
            return float("nan")
        order = np.argsort(x, kind="stable")
        sx = x[order]
        ranks = np.empty(len(x), dtype=np.float64)
        start = 0
        while start < len(sx):
            end = start + 1
            while end < len(sx) and sx[end] == sx[start]:
                end += 1
            ranks[order[start:end]] = 0.5 * ((start + 1) + end)
            start = end
        rank_sum = float(np.sum(ranks[yy == 1]))
        return (rank_sum - p * (p + 1) / 2.0) / (p * n)
    except Exception:
        return float("nan")


def score_diagnostics(scores: np.ndarray) -> dict[str, np.ndarray]:
    S = np.asarray(scores, dtype=np.float64)
    if S.ndim == 1:
        S = S[None, :]
    finite = np.isfinite(S)
    safe = np.where(finite, S, 0.0)
    A = np.abs(safe)
    max_abs = np.max(A, axis=1)
    constant = np.empty(len(S), dtype=bool)
    for i, row in enumerate(safe):
        # ptp would overflow on extreme rows; they are unsafe anyway.
        if max_abs[i] >= MAX_ABS_OUTPUT:
            constant[i] = False
        else:
            spread = float(np.max(row)) - float(np.min(row))
            constant[i] = spread <= 1e-14 * max(1.0, float(max_abs[i]))
    q50 = np.quantile(A, .50, axis=1)
    q99 = np.quantile(A, .99, axis=1)
    ratio = np.full(len(S), np.inf)
    np.divide(q99, q50, out=ratio, where=q50 > np.finfo(float).tiny)
    ratio[(q50 <= np.finfo(float).tiny) & (q99 == 0)] = 0.0
    k = max(1, int(math.ceil(.01 * S.shape[1])))
    top = (np.partition(A, -k, axis=1)[:, -k:].sum(axis=1)
           / np.maximum(A.sum(axis=1), np.finfo(float).tiny))
    return {"finite_fraction": finite.mean(axis=1), "constant_or_near_constant": constant,
            "q99_to_q50": ratio, "top_1pct_absolute_share": top, "max_abs": max_abs}


def grouped_oof_metrics(scores: np.ndarray, y: np.ndarray,
                        folds: np.ndarray) -> dict[str, Any]:
    """Out-of-fold classification with thresholds fitted on the other folds.

    Out-of-fold margins are oriented and centred by the training-fold
    threshold, so one AUC can be computed across folds.
    """
    S = np.asarray(scores, dtype=np.float64)
    yy = np.asarray(y, dtype=np.int8)
    ff = np.asarray(folds, dtype=np.int16)
    if S.ndim == 1:
        S = S[None, :]
    if S.shape[1] != len(yy) or len(ff) != len(yy):
        raise ValueError("score, label and fold shapes differ")
    fold_values = sorted(set(map(int, ff)))
    B, N = S.shape
    pred = np.zeros((B, N), dtype=bool)
    margin = np.zeros((B, N), dtype=np.float64)
    thresholds = np.zeros((B, len(fold_values)), dtype=np.float64)
    orientations = np.zeros((B, len(fold_values)), dtype=np.int8)
    for j, f in enumerate(fold_values):
        tr = ff != f
        va = ff == f
        if not np.any(va) or len(np.unique(yy[tr])) < 2:
            raise ValueError(f"fold {f} cannot be held out")
        fit = fit_threshold_batch(S[:, tr], yy[tr])
        thresholds[:, j] = fit["threshold"]
        orientations[:, j] = fit["orientation"]
        pred[:, va] = np.where(fit["orientation"][:, None] == 1,
                               S[:, va] >= fit["threshold"][:, None],
                               S[:, va] <= fit["threshold"][:, None])
        margin[:, va] = fit["orientation"][:, None] * (S[:, va] - fit["threshold"][:, None])
    pos = yy == 1
    neg = yy == 0
    P = max(1, int(pos.sum()))
    Nn = max(1, int(neg.sum()))
    tp = np.sum(pred[:, pos], axis=1)
    tn = np.sum(~pred[:, neg], axis=1)
    fp = Nn - tn
    fn = P - tp
    pr = tp / P
    nr = tn / Nn
    bacc = .5 * (pr + nr)
    acc = (tp + tn) / (P + Nn)
    f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    den = np.sqrt(np.maximum(1, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / den
    auc = np.asarray([safe_auc(yy, margin[i], 1) for i in range(B)], dtype=np.float64)
    agreement = np.maximum(np.mean(orientations == 1, axis=1),
                           np.mean(orientations == -1, axis=1))
    scale = np.quantile(np.abs(S), .75, axis=1)
    scale = np.where(scale > 1e-12, scale, 1.0)
    th_std = np.std(thresholds, axis=1) / scale
    return {"prediction": pred, "margin": margin, "thresholds": thresholds,
            "orientations": orientations, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "positive_recall": pr, "negative_recall": nr, "worst_recall": np.minimum(pr, nr),
            "bacc": bacc, "acc": acc, "f1": f1, "mcc": mcc, "auc": auc,
            "orientation_agreement": agreement, "threshold_std_scaled": th_std}


def prediction_id(prediction: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(prediction, dtype=np.uint8).tobytes()).hexdigest()


@dataclass(frozen=True)
class Scoring:
    """Labels, row partitions and the threshold protocol of one data set."""

    mode: str
    y: np.ndarray
    hard_rows: np.ndarray
    all_rows: np.ndarray
    soft_rows: np.ndarray
    folds: np.ndarray | None = None

    def metrics(self, scores: np.ndarray) -> list[dict[str, Any]]:
        """One metric record per row of ``scores``."""
        if self.mode == HARD_CORE:
            return self._hard_core_metrics(scores)
        return self._grouped_metrics(scores)

    def context_bacc(self, scores: np.ndarray) -> np.ndarray:
        """Balanced accuracy only, for the many probe expressions of the context scan."""
        y = self.y
        if self.mode == HARD_CORE:
            fit = fit_threshold_batch(scores[:, self.hard_rows], y[self.hard_rows])
            return evaluate_fixed_batch(scores[:, self.all_rows], y[self.all_rows],
                                        fit["threshold"], fit["orientation"])["bacc"]
        return grouped_oof_metrics(scores[:, self.all_rows], y[self.all_rows],
                                   self.folds)["bacc"]

    def _hard_core_metrics(self, scores: np.ndarray) -> list[dict[str, Any]]:
        S = np.asarray(scores, dtype=np.float64)
        if S.ndim == 1:
            S = S[None, :]
        yy = np.asarray(self.y, dtype=np.int8)
        hard, all_rows, soft = self.hard_rows, self.all_rows, self.soft_rows
        fit = fit_threshold_batch(S[:, hard], yy[hard])
        allm = evaluate_fixed_batch(S[:, all_rows], yy[all_rows], fit["threshold"],
                                    fit["orientation"])
        softm = evaluate_fixed_batch(S[:, soft], yy[soft], fit["threshold"],
                                     fit["orientation"])
        diag = score_diagnostics(S)
        out = []
        for i in range(len(S)):
            orientation = int(fit["orientation"][i])
            out.append({
                "safe": _is_safe(diag, i),
                "threshold": float(fit["threshold"][i]), "orientation": orientation,
                "hard_bacc": float(fit["bacc"][i]),
                "hard_worst_recall": float(fit["worst_recall"][i]),
                "threshold_margin": float(fit["margin"][i]),
                "all_bacc": float(allm["bacc"][i]),
                "all_worst_recall": float(allm["worst_recall"][i]),
                "all_acc": float(allm["acc"][i]), "all_f1": float(allm["f1"][i]),
                "all_mcc": float(allm["mcc"][i]),
                "all_auc": safe_auc(yy[all_rows], S[i, all_rows], orientation),
                "soft_bacc": float(softm["bacc"][i]),
                "soft_worst_recall": float(softm["worst_recall"][i]),
                "soft_acc": float(softm["acc"][i]), "soft_f1": float(softm["f1"][i]),
                "soft_mcc": float(softm["mcc"][i]),
                "soft_auc": safe_auc(yy[soft], S[i, soft], orientation),
                **_diagnostic_fields(diag, i),
                "prediction_id": prediction_id(allm["prediction"][i]),
            })
        return out

    def _grouped_metrics(self, scores: np.ndarray) -> list[dict[str, Any]]:
        S = np.asarray(scores, dtype=np.float64)
        if S.ndim == 1:
            S = S[None, :]
        yy = np.asarray(self.y, dtype=np.int8)
        idx = np.asarray(self.all_rows, dtype=np.int32)
        oof = grouped_oof_metrics(S[:, idx], yy[idx], self.folds)
        # The full-data fit gives the reported threshold only; it never enters
        # the out-of-fold estimate.
        full = fit_threshold_batch(S[:, idx], yy[idx])
        diag = score_diagnostics(S)
        out = []
        for i in range(len(S)):
            record = {
                "safe": _is_safe(diag, i),
                "threshold": float(full["threshold"][i]),
                "orientation": int(full["orientation"][i]),
                "oof_bacc": float(oof["bacc"][i]),
                "oof_worst_recall": float(oof["worst_recall"][i]),
                "oof_acc": float(oof["acc"][i]), "oof_f1": float(oof["f1"][i]),
                "oof_mcc": float(oof["mcc"][i]), "oof_auc": float(oof["auc"][i]),
                "oof_orientation_agreement": float(oof["orientation_agreement"][i]),
                "oof_threshold_std_scaled": float(oof["threshold_std_scaled"][i]),
                "full_fit_bacc": float(full["bacc"][i]),
                "full_fit_worst_recall": float(full["worst_recall"][i]),
                "threshold_margin": float(full["margin"][i]),
                **_diagnostic_fields(diag, i),
                "prediction_id": prediction_id(oof["prediction"][i]),
            }
            # Selection and archives read the hard_*/all_*/soft_* fields; under
            # grouped folds all three are the out-of-fold estimate.
            for prefix, names in _OOF_ALIASES:
                for name in names:
                    record[f"{prefix}_{name}"] = record[f"oof_{name}"]
            out.append(record)
        return out


_OOF_ALIASES = (("hard", ("bacc", "worst_recall")),
                ("all", ("bacc", "worst_recall", "acc", "f1", "mcc", "auc")),
                ("soft", ("bacc", "worst_recall", "acc", "f1", "mcc", "auc")))


def _is_safe(diag: Mapping[str, np.ndarray], i: int) -> bool:
    return bool(diag["finite_fraction"][i] == 1.0
                and not diag["constant_or_near_constant"][i]
                and diag["max_abs"][i] < MAX_ABS_OUTPUT)


def _diagnostic_fields(diag: Mapping[str, np.ndarray], i: int) -> dict[str, Any]:
    return {"finite_fraction": float(diag["finite_fraction"][i]),
            "constant_or_near_constant": bool(diag["constant_or_near_constant"][i]),
            "q99_to_q50": float(diag["q99_to_q50"][i]),
            "top_1pct_absolute_share": float(diag["top_1pct_absolute_share"][i]),
            "max_abs": float(diag["max_abs"][i])}


def performance_key(record: Mapping[str, Any], mode: str) -> tuple[Any, ...]:
    """Archive order: safe first, then BACC, worst recall, a protocol-specific
    tie-breaker, expanded size and depth, and the formula id."""
    c = record.get("complexity", {})
    if isinstance(c, str):
        try:
            c = json.loads(c)
        except Exception:
            c = {}
    operators = float(record.get("expanded_operator_nodes", record.get(
        "operator_nodes", c.get("expanded_operator_nodes", 1e9))))
    depth = float(record.get("expanded_tree_depth", record.get(
        "tree_depth", c.get("expanded_tree_depth", 1e9))))
    if mode == HARD_CORE:
        tie_breaker = -float(record.get("hard_bacc", -1))
    else:
        tie_breaker = -float(record.get("oof_orientation_agreement", 0.0))
    return (0 if bool(record.get("safe")) else 1, -float(record.get("all_bacc", -1)),
            -float(record.get("all_worst_recall", -1)), tie_breaker, operators, depth,
            str(record.get("formula_id", "")))


def parameter_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Order among parameter variants of one structure."""
    return (-float(record.get("hard_bacc", -1)), -float(record.get("hard_worst_recall", -1)),
            -float(record.get("threshold_margin", -1)),
            float(record.get("q99_to_q50", np.inf)), str(record.get("formula_id", "")))


def natural_bacc_resolution(y: np.ndarray) -> float:
    yy = np.asarray(y, dtype=np.int8)
    p = max(1, int(np.sum(yy == 1)))
    n = max(1, int(np.sum(yy == 0)))
    return min(1.0 / (2.0 * p), 1.0 / (2.0 * n))


def grouped_accuracy(prediction: np.ndarray, y: np.ndarray,
                     codes: np.ndarray) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=bool)
    yy = np.asarray(y, dtype=np.int8)
    cc = np.asarray(codes, dtype=np.int32)
    acc = []
    exact = []
    for code in sorted(set(map(int, cc))):
        m = cc == code
        a = float(np.mean(pred[m] == yy[m]))
        acc.append(a)
        exact.append(float(a == 1.0))
    return {"macro_accuracy": float(np.mean(acc)) if acc else float("nan"),
            "worst_accuracy": float(np.min(acc)) if acc else float("nan"),
            "exact_rate": float(np.mean(exact)) if exact else float("nan")}
