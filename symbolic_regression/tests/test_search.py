"""End-to-end runs on the synthetic data set (Linux: fork and memory-mapped files)."""
import json
import sys
import warnings

import pandas as pd
import pytest

from symbolic_regression.data import load_data
from symbolic_regression.replay import load_formulas, replay
from symbolic_regression.search import run
from symbolic_regression.settings import make_settings

from .conftest import TINY

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the search runs on Linux")

COMPARED = ["top30.csv", "components.csv", "archive_performance.csv", "skeleton_history.csv"]


@pytest.fixture(autouse=True)
def warnings_as_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def search(inputs, outdir, **changes):
    protocol, space = inputs
    settings = make_settings({**TINY, **changes})
    return run(settings, protocol, space, outdir, scratch=outdir.parent / "scratch")


def table(outdir, name):
    frame = pd.read_csv(outdir / name, dtype=str, keep_default_na=False)
    return frame.drop(columns=[c for c in frame if c == "seconds"])


@pytest.fixture(scope="module")
def reference(inputs, tmp_path_factory):
    outdir = tmp_path_factory.mktemp("reference") / "run"
    record = search(inputs, outdir, scoring="hard_core")
    return outdir, record


def test_run_finishes_with_a_ranked_list(reference):
    outdir, record = reference
    assert record["status"] == "finalized"
    top = pd.read_csv(outdir / "top30.csv")
    assert len(top) == 30 and list(top["rank"]) == list(range(1, 31))
    assert json.loads((outdir / "best_formula.json").read_text())["rank"] == 1
    assert record["inputs"]["sha256"]


def test_worker_count_does_not_change_results(inputs, reference, tmp_path):
    outdir = tmp_path / "run"
    search(inputs, outdir, scoring="hard_core", jobs=3, task_size=7)
    for name in COMPARED:
        pd.testing.assert_frame_equal(table(reference[0], name), table(outdir, name))


def test_resume_continues_exactly(inputs, reference, tmp_path):
    protocol, space = inputs
    outdir = tmp_path / "run"
    settings = make_settings({**TINY, "scoring": "hard_core"})
    first = run(settings, protocol, space, outdir, scratch=tmp_path / "scratch",
                stop_after_skeleton_generation=2)
    assert first["status"] == "stopped"
    run(settings, protocol, space, outdir, resume=True, scratch=tmp_path / "scratch")
    for name in COMPARED:
        pd.testing.assert_frame_equal(table(reference[0], name), table(outdir, name))


def test_resume_refuses_other_settings(inputs, reference):
    protocol, space = inputs
    settings = make_settings({**TINY, "scoring": "hard_core", "beam_width": 4})
    with pytest.raises(ValueError, match="beam_width"):
        run(settings, protocol, space, reference[0], resume=True)


def test_grouped_scoring_and_early_finalization(inputs, tmp_path):
    outdir = tmp_path / "run"
    record = search(inputs, outdir, scoring="grouped_oof", final_skeleton_generation=2)
    assert record["status"] == "finalized"
    assert len(pd.read_csv(outdir / "skeleton_history.csv")) == 2
    oof = pd.read_csv(outdir / "top30_oof_predictions.csv")
    assert set(oof["fold_id"]) == set(range(5))


def test_replay_reproduces_the_top_list(inputs, reference):
    protocol, space = inputs
    outdir = reference[0]
    settings = make_settings({**TINY, "scoring": "hard_core"})
    data = load_data(protocol, space, settings.scoring)
    reports, scores = replay(load_formulas(outdir / "top30.csv"), outdir / "components.csv",
                             data, settings)
    assert all(r["all_equal"] for r in reports)
    assert len(scores) == 30 * data.n_samples
