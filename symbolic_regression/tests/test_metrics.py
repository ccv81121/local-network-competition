import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from symbolic_regression.metrics import (GROUPED_OOF, HARD_CORE, Scoring, fit_threshold_batch,
                                         grouped_oof_metrics, performance_key, safe_auc)


def test_separable_scores_are_classified_perfectly():
    rng = np.random.default_rng(0)
    x1, x2 = rng.normal(size=(2, 200))
    y = (x1 - x2 > 0).astype(np.int8)
    assert fit_threshold_batch((x1 - x2)[None, :], y)["bacc"][0] == 1.0
    assert fit_threshold_batch(x1[None, :], y)["bacc"][0] < 0.95


def test_orientation_follows_the_sign():
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    fit = fit_threshold_batch(np.array([[4.0, 3.0, 2.0, 1.0]]), y)
    assert fit["orientation"][0] == -1 and fit["bacc"][0] == 1.0


@pytest.mark.parametrize("mode", [HARD_CORE, GROUPED_OOF])
def test_batch_and_single_rows_agree(mode):
    rng = np.random.default_rng(1)
    n = 120
    y = (rng.random(n) < 0.4).astype(np.int8)
    scoring = Scoring(mode, y, np.arange(0, n, 2), np.arange(n), np.arange(1, n, 2),
                      (np.arange(n) % 5).astype(np.int16))
    scores = rng.normal(size=(9, n))
    batch = scoring.metrics(scores)
    for i in range(len(scores)):
        assert scoring.metrics(scores[i:i + 1])[0] == batch[i]


def test_rank_auc_matches_sklearn():
    y = np.array([0, 1, 0, 1, 1, 0, 1], dtype=np.int8)
    s = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 0.5, -1.0])
    assert safe_auc(y, s, 1) == pytest.approx(roc_auc_score(y, s), rel=1e-12)
    assert safe_auc(y, s, -1) == pytest.approx(roc_auc_score(y, -s), rel=1e-12)


def test_held_out_scores_do_not_move_their_threshold():
    rng = np.random.default_rng(2)
    n = 60
    y = (rng.random(n) < 0.5).astype(np.int8)
    folds = (np.arange(n) % 3).astype(np.int16)
    s = rng.normal(size=n) + y
    shifted = s.copy()
    shifted[folds == 0] += 100.0
    a = grouped_oof_metrics(s[None, :], y, folds)
    b = grouped_oof_metrics(shifted[None, :], y, folds)
    assert a["thresholds"][0, 0] == b["thresholds"][0, 0]


def test_safe_records_rank_first():
    good = {"safe": True, "all_bacc": 0.7, "formula_id": "b"}
    unsafe = {"safe": False, "all_bacc": 0.9, "formula_id": "a"}
    assert performance_key(good, HARD_CORE) < performance_key(unsafe, HARD_CORE)
