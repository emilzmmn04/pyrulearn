"""`pyrulearn.experiments.runner`/`stats`/`report`: offline (a tiny
synthetic dataset stands in for `CatalogEntry.load()` -- nothing here
hits the network), with `Pypper`/`PFossil` as fast, deterministic-enough
real learners rather than a hand-rolled fake one."""

import os
import time

import numpy as np
import pandas as pd
import pytest

from pyrulearn.experiments.catalog import CatalogEntry
from pyrulearn.experiments.report import render_results_table, render_setup_section
from pyrulearn.experiments.runner import TimeoutRunner, run_cv
from pyrulearn.experiments.stats import (
    critical_difference, critical_difference_diagram, critical_difference_groups, friedman_test, mean_rank,
    win_counts,
)
from pyrulearn.learners.seco import PFossil, Pypper


def _synthetic_entry(name: str, n: int = 60, seed: int = 0) -> CatalogEntry:
    """A `CatalogEntry` whose `load()` is monkeypatched to a small,
    in-memory, easily-learnable binary dataset -- no OpenML id, no
    network access. One numeric and one nominal feature, both
    informative, so `build_dataspec`/`binarize` exercise both code paths."""
    rng = np.random.default_rng(seed)
    num = rng.normal(size=n)
    nom = rng.choice(["p", "q"], size=n)
    noise = rng.normal(scale=0.2, size=n)
    y = np.where((num + np.where(nom == "p", 0.5, -0.5) + noise) > 0, "pos", "neg")
    df = pd.DataFrame({"num": num, "nom": nom, "target": y})
    entry = CatalogEntry(name=name)
    entry.load = lambda: (df, "target")
    return entry


LEARNERS = [Pypper(random_state=0), PFossil(random_state=0)]


def test_run_cv_produces_the_documented_columns():
    entries = [_synthetic_entry("synth_a", seed=0), _synthetic_entry("synth_b", n=50, seed=1)]
    results = run_cv(LEARNERS, entries, n_folds=3, fit_timeout=None, random_state=0, verbose=False)

    assert set(results["dataset"]) == {"synth_a", "synth_b"}
    assert set(results["learner"]) == {"Pypper", "PFossil"}
    assert len(results) == 2 * 3 * 2  # datasets x folds x learners
    for col in ("dataset", "fold", "learner", "fit_time", "error", "accuracy", "n_rules", "n_conditions"):
        assert col in results.columns
    # a clean, easily-separable synthetic dataset: no fit should fail
    assert results["error"].isna().all()
    assert results["accuracy"].between(0, 1).all()
    assert (results["fit_time"] >= 0).all()


def test_run_cv_verbose_prints_per_dataset_and_per_fold_progress(capsys):
    entries = [_synthetic_entry("synth_a"), _synthetic_entry("synth_b", n=50, seed=1)]
    run_cv(LEARNERS, entries, n_folds=2, fit_timeout=None, random_state=0, verbose=True)
    out = capsys.readouterr().out
    for name in ("synth_a", "synth_b"):
        assert f"{name} ..." in out
    for learner in LEARNERS:
        assert learner.display_name in out
    assert "fold 1/2" in out and "fold 2/2" in out
    assert "acc=" in out  # every fit succeeds on this dataset -- no "error" outcome to check instead


def test_run_cv_verbose_false_prints_nothing(capsys):
    entries = [_synthetic_entry("synth_a")]
    run_cv(LEARNERS, entries, n_folds=2, fit_timeout=None, random_state=0, verbose=False)
    assert capsys.readouterr().out == ""


def test_run_cv_verbose_marks_cached_rows(tmp_path, capsys):
    entries = [_synthetic_entry("synth_a")]
    cache_dir = str(tmp_path / "cache")
    run_cv(LEARNERS, entries, n_folds=2, fit_timeout=None, random_state=0,
          cache_dir=cache_dir, verbose=False)
    capsys.readouterr()  # discard the first (uncached) run's output, if any
    run_cv(LEARNERS, entries, n_folds=2, fit_timeout=None, random_state=0,
          cache_dir=cache_dir, verbose=True)
    assert "(cached)" in capsys.readouterr().out


def test_run_cv_measure_fn_adds_extra_columns():
    entries = [_synthetic_entry("synth_a")]

    def extra(model, test_rep):
        return {"n_predicted_pos": int((model.predict(test_rep) == "pos").sum())}

    results = run_cv(LEARNERS, entries, n_folds=2, fit_timeout=None, random_state=0,
                     measure_fn=extra, verbose=False)
    assert "n_predicted_pos" in results.columns
    assert results["n_predicted_pos"].notna().all()


def test_run_cv_caches_finished_folds(tmp_path):
    calls = {"n": 0}

    class CountingPypper(Pypper):
        # `_fit_default` (not `fit`) -- `fit(data)` with no `model=` calls
        # `_fit_default`, which itself calls back into `self.fit(data,
        # model=...)`; counting in `fit` directly would double-count.
        def _fit_default(self, data):
            calls["n"] += 1
            return super()._fit_default(data)

    entries = [_synthetic_entry("synth_a"), _synthetic_entry("synth_b", n=50, seed=1)]
    cache_dir = str(tmp_path / "cv_cache")
    learner = CountingPypper(random_state=0)

    first = run_cv([learner], entries, n_folds=3, fit_timeout=None, random_state=0,
                   cache_dir=cache_dir, verbose=False)
    assert calls["n"] == 2 * 3  # datasets x folds, once each

    second = run_cv([learner], entries, n_folds=3, fit_timeout=None, random_state=0,
                    cache_dir=cache_dir, verbose=False)
    assert calls["n"] == 2 * 3  # every row came from the cache -- no new fits
    pd.testing.assert_frame_equal(
        first.sort_values(["dataset", "fold"]).reset_index(drop=True),
        second.sort_values(["dataset", "fold"]).reset_index(drop=True),
    )


def test_run_cv_does_not_load_or_encode_what_is_cached(tmp_path, monkeypatch):
    import pyrulearn.experiments.runner as runner

    loads, encodings = {"n": 0}, {"n": 0}
    entry = _synthetic_entry("synth_a")
    load = entry.load

    def counting_load():
        loads["n"] += 1
        return load()

    entry.load = counting_load
    encode = runner.encode

    def counting_encode(*args, **kwargs):
        encodings["n"] += 1
        return encode(*args, **kwargs)

    monkeypatch.setattr(runner, "encode", counting_encode)
    cache_dir = str(tmp_path / "cache")
    kwargs = dict(n_folds=3, fit_timeout=None, random_state=0, cache_dir=cache_dir, verbose=False)
    first = run_cv([Pypper(random_state=0)], [entry], **kwargs)
    assert (loads["n"], encodings["n"]) == (1, 2 * 3)  # train and test, per fold

    # everything cached: the dataset is not loaded, no fold encoded -- also in parallel
    for n_jobs in (1, 2):
        again = run_cv([Pypper(random_state=0)], [entry], n_jobs=n_jobs, **kwargs)
        assert (loads["n"], encodings["n"]) == (1, 2 * 3)
        pd.testing.assert_frame_equal(first, again)

    # a new learner: its folds are not cached, so they are encoded again
    both = run_cv([Pypper(random_state=0), PFossil(random_state=0)], [entry], **kwargs)
    assert (loads["n"], encodings["n"]) == (2, 4 * 3)
    assert len(both) == 2 * 3 and both["error"].isna().all()


def test_cache_key_is_stable_for_learners_built_from_components():
    # regression test: a SeCo learner's parameters include its
    # SingleRuleLearner object, whose default str() carries a memory
    # address -- two equal learners built separately (as in two runs) got
    # different keys, so their cached folds were never reused
    from pyrulearn.data import HorizontalBitmapRepresentation, NListRepresentation
    from pyrulearn.experiments.runner import _cache_key

    entry = _synthetic_entry("synth_a")
    key = lambda learner, rep=NListRepresentation: _cache_key(entry, 3, 0, 0, learner, rep)  # noqa: E731
    assert key(PFossil(random_state=0)) == key(PFossil(random_state=0))
    assert key(PFossil(random_state=0)) != key(PFossil(random_state=0, correlation_threshold=0.2))
    assert key(PFossil(random_state=0)) != key(PFossil(random_state=0), rep=HorizontalBitmapRepresentation)


def _sleep_then_return(seconds: float) -> str:
    """Module-level (picklable by reference) -- see `TimeoutRunner`'s
    docstring on why that's required."""
    time.sleep(seconds)
    return "done"


def test_timeout_runner_records_a_timeout_instead_of_raising():
    # timeout=15.0 leaves generous headroom for the replacement worker's
    # own spawn -- a fresh `python -m multiprocessing.spawn` reimporting
    # this test module and its dependencies, measured at ~1.3s in
    # isolation but seen taking several times that under load (a smaller
    # budget here made this test measurably flaky).
    runner = TimeoutRunner(timeout=15.0)
    try:
        result, error = runner.run(_sleep_then_return, 60.0)
        assert result is None
        assert error == "timeout"
        # the worker was replaced -- a fast call afterwards still works
        result, error = runner.run(_sleep_then_return, 0.0)
        assert result == "done" and error is None
    finally:
        runner.close()


def test_timeout_runner_exits_cleanly_after_killing_a_worker_that_never_read_its_task():
    """A worker killed before reading its task (here: 8 MB, not read within the timeout) used to leave the
    queue's feeder thread blocked on a full pipe, and Python joins that thread at exit --
    the process hung. Run in a subprocess, so a regression fails the test instead of
    hanging it."""
    import os
    import subprocess
    import sys
    code = ("from pyrulearn.experiments.runner import TimeoutRunner\n"
            "runner = TimeoutRunner(timeout=0.001)\n"
            "print(runner.run(len, b'x' * (8 << 20)))\n"
            "runner.close()\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "timeout" in proc.stdout


def test_timeout_runner_does_not_count_a_new_workers_start_up():
    """A new worker takes a second or more to start (a fresh interpreter importing numpy,
    scikit-learn and pyrulearn). The timeout counts from when it is up, so a fast call on a
    fresh worker -- the first one, or the replacement after a timeout -- doesn't time out.
    (`operator.add` and `time.sleep`: unpickling them imports nothing new in the worker.)"""
    import operator
    runner = TimeoutRunner(timeout=0.5)
    try:
        assert runner.run(operator.add, 1, 2) == (3, None)           # the first worker
        assert runner.run(time.sleep, 60.0) == (None, "timeout")
        assert runner.run(operator.add, 1, 2) == (3, None)           # its replacement
    finally:
        runner.close()


def _start_a_child_then_hang(pid_file: str) -> None:
    """Module-level (picklable by reference): starts a child process, as a fitter's own
    pool or an external learner's JVM would, writes its pid to `pid_file`, and hangs."""
    import subprocess
    import sys
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    with open(pid_file, "w") as f:
        f.write(str(child.pid))
    time.sleep(120)


def test_timeout_runner_stops_the_child_processes_of_a_timed_out_call(tmp_path):
    psutil = pytest.importorskip("psutil")
    pid_file = tmp_path / "child.pid"
    runner = TimeoutRunner(timeout=30.0)
    try:
        assert runner.run(_sleep_then_return, 0.0) == ("done", None)  # the worker imports this module
        assert runner.run(_start_a_child_then_hang, str(pid_file), timeout=3.0) == (None, "timeout")
    finally:
        runner.close()
    try:
        child = psutil.Process(int(pid_file.read_text()))
    except psutil.NoSuchProcess:
        return  # already gone
    try:
        gone, _ = psutil.wait_procs([child], timeout=10)
        assert gone or child.status() == psutil.STATUS_ZOMBIE, "the timed-out call's child is still running"
    except psutil.NoSuchProcess:
        pass
    finally:
        try:
            child.kill()  # don't leave it behind if the test failed
        except psutil.Error:
            pass


def test_run_cv_records_a_failed_fit_without_raising():
    class AlwaysFails(Pypper):
        def fit(self, data, model=None, **model_kwargs):
            raise RuntimeError("synthetic failure")

    entries = [_synthetic_entry("synth_a")]
    results = run_cv([AlwaysFails()], entries, n_folds=2, fit_timeout=None, random_state=0, verbose=False)
    assert (results["error"] == "RuntimeError: synthetic failure").all()
    assert results["accuracy"].isna().all()


class _AlwaysFailsLearner(Pypper):
    """Module-level, so that it can be sent to a worker process (a class defined
    inside a test function can't be)."""
    def fit(self, data, model=None, **model_kwargs):
        raise RuntimeError("synthetic failure")


def _cells(results):
    return results.drop(columns=["fit_time"]).reset_index(drop=True)


def test_run_cv_n_jobs_gives_the_same_table_as_sequential():
    entries = [_synthetic_entry("synth_a", seed=0), _synthetic_entry("synth_b", n=50, seed=1)]
    kwargs = dict(n_folds=3, fit_timeout=60.0, random_state=0, verbose=False)
    sequential = run_cv(LEARNERS, entries, **kwargs)
    parallel = run_cv(LEARNERS, entries, n_jobs=2, **kwargs)

    # same rows, same order, same measurements; only the timings may differ
    pd.testing.assert_frame_equal(_cells(parallel), _cells(sequential))
    assert (parallel["fit_time"] >= 0).all()


def test_run_cv_n_jobs_uses_and_fills_the_cache(tmp_path, capsys):
    entries = [_synthetic_entry("synth_a")]
    kwargs = dict(n_folds=2, fit_timeout=60.0, random_state=0, cache_dir=str(tmp_path), n_jobs=2)
    first = run_cv(LEARNERS, entries, verbose=False, **kwargs)
    assert len(list(tmp_path.glob("*.json"))) == len(first) == 4

    second = run_cv(LEARNERS, entries, verbose=True, **kwargs)
    out = capsys.readouterr().out
    assert out.count("(cached)") == 4
    pd.testing.assert_frame_equal(second.reset_index(drop=True), first.reset_index(drop=True))


def test_run_cv_n_jobs_records_a_failed_fit_without_raising():
    entries = [_synthetic_entry("synth_a")]
    results = run_cv([_AlwaysFailsLearner(), PFossil(random_state=0)], entries, n_folds=2,
                     fit_timeout=60.0, random_state=0, verbose=False, n_jobs=2)
    assert len(results) == 4
    failed = results["error"].notna()
    assert failed.sum() == 2 and results.loc[~failed, "learner"].eq("PFossil").all()
    assert (results.loc[failed, "error"] == "RuntimeError: synthetic failure").all()
    assert results["fold"].tolist() == [0, 0, 1, 1]  # still in sequential order


def test_run_cv_rejects_a_bad_n_jobs():
    entries = [_synthetic_entry("synth_a")]
    for bad in (0, -1, 1.5, None, True, "", "x", "-2", "3!!", "!"):
        with pytest.raises(ValueError):
            run_cv(LEARNERS, entries, n_folds=2, fit_timeout=None, verbose=False, n_jobs=bad)


def test_n_jobs_above_the_available_cores_is_capped_with_a_warning(monkeypatch):
    from pyrulearn.experiments import runner

    monkeypatch.setattr(runner, "_available_cpus", lambda: 2)
    with pytest.warns(RuntimeWarning, match=r"more than the 2 available.*'10!'"):
        assert runner._resolve_n_jobs(10) == 2
    with pytest.warns(RuntimeWarning):
        assert runner._resolve_n_jobs("10") == 2


def test_n_jobs_within_the_available_cores_is_left_alone(monkeypatch, recwarn):
    from pyrulearn.experiments import runner

    monkeypatch.setattr(runner, "_available_cpus", lambda: 4)
    assert runner._resolve_n_jobs(4) == 4
    assert runner._resolve_n_jobs(1) == 1
    assert runner._resolve_n_jobs("3") == 3
    assert not recwarn.list


def test_n_jobs_with_an_exclamation_mark_is_not_capped(monkeypatch, recwarn):
    from pyrulearn.experiments import runner

    monkeypatch.setattr(runner, "_available_cpus", lambda: 1)
    assert runner._resolve_n_jobs("10!") == 10
    assert runner._resolve_n_jobs(" 3! ") == 3
    assert not recwarn.list


def test_run_cv_starts_the_forced_number_of_workers(monkeypatch):
    from pyrulearn.experiments import runner

    monkeypatch.setattr(runner, "_available_cpus", lambda: 1)
    started = []
    real = runner.TimeoutRunner
    monkeypatch.setattr(runner, "TimeoutRunner", lambda *a, **k: started.append(1) or real(*a, **k))
    entries = [_synthetic_entry("synth_a")]
    results = run_cv(LEARNERS, entries, n_folds=2, fit_timeout=60.0, random_state=0, verbose=False,
                     n_jobs="3!")
    assert len(started) == 3 and len(results) == 4


class _SlowOnSize(Pypper):
    """Module-level, so a worker process can unpickle it: sleeps far past any test's
    `fit_timeout` on a training set of exactly `slow_size` rows."""
    def __init__(self, slow_size: int, **kwargs):
        super().__init__(**kwargs)
        self.slow_size = slow_size

    def fit(self, data, model=None, **model_kwargs):
        if data.n_samples == self.slow_size:
            time.sleep(60)
        return super().fit(data, model, **model_kwargs)


def _fold_train_sizes(entry, n_folds):
    from pyrulearn.experiments.runner import _stratified_folds
    df, target = entry.load()
    return [len(tr) for tr, _ in _stratified_folds(df[target].to_numpy(), n_folds, 0)]


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_run_cv_stop_on_timeout_fails_the_whole_cross_validation(tmp_path, n_jobs):
    entry = _synthetic_entry("synth_a", n=61)
    sizes = _fold_train_sizes(entry, 3)
    slow = next(s for s in sizes[1:] if s != sizes[0])        # fold 0 finishes, a later one hangs
    learners = [_SlowOnSize(slow, random_state=0), PFossil(random_state=0)]
    results = run_cv(learners, [entry], n_folds=3, fit_timeout=3.0, random_state=0, verbose=False,
                     cache_dir=str(tmp_path), n_jobs=n_jobs)
    name = learners[0].display_name
    slow_rows = results[results["learner"] == name]
    assert len(slow_rows) == 3
    assert (slow_rows["error"] == "timeout").all() and slow_rows["accuracy"].isna().all()
    other = results[results["learner"] == "PFossil"]
    assert len(other) == 3 and other["error"].isna().all()  # other learners unaffected
    # the cache keeps what really happened: fold 0 finished
    import json
    cached = [json.load(open(f)) for f in tmp_path.glob("*.json")]
    fold0 = [r for r in cached if r["learner"] == name and r["fold"] == 0]
    assert fold0 and fold0[0]["error"] is None
    from pyrulearn.experiments.runner import timed_out
    assert timed_out(results) == {("synth_a", name)}
    if n_jobs == 1:   # sequentially, the folds after the timed-out one are not run at all
        first_slow = sizes.index(slow)
        later = slow_rows[slow_rows["fold"] > first_slow]
        assert later["fit_time"].isna().all()


def test_a_cached_timeout_is_tried_again_with_a_longer_fit_timeout(tmp_path):
    from pyrulearn.experiments.runner import _cache_load_valid, _cache_save
    _cache_save(str(tmp_path), "t", {"error": "timeout", "fit_time": 300.4})
    _cache_save(str(tmp_path), "ok", {"error": None, "fit_time": 1.0})
    assert _cache_load_valid(str(tmp_path), "t", 300.0) is not None    # the same timeout: still timed out
    assert _cache_load_valid(str(tmp_path), "t", 1800.0) is None       # a longer one: run it again
    assert _cache_load_valid(str(tmp_path), "t", None) is None         # no limit at all: run it again
    assert _cache_load_valid(str(tmp_path), "ok", 0.5) is not None     # a finished fold is always kept


def test_run_cv_cached_only_reports_the_cache_and_fits_nothing(tmp_path):
    entries = [_synthetic_entry("synth_a"), _synthetic_entry("synth_b")]
    cache = str(tmp_path)
    full = run_cv(LEARNERS, entries, n_folds=2, fit_timeout=None, random_state=0, cache_dir=cache, verbose=False)
    cached = run_cv(LEARNERS, entries, n_folds=2, fit_timeout=None, random_state=0, cache_dir=cache,
                    cached_only=True)
    pd.testing.assert_frame_equal(cached.drop(columns="error"), full.drop(columns="error"))

    os.remove(next(os.path.join(cache, f) for f in sorted(os.listdir(cache))))
    missing = run_cv(LEARNERS, entries, n_folds=2, fit_timeout=None, random_state=0, cache_dir=cache,
                     cached_only=True)
    assert len(missing) == len(full) - 1    # left out, not fitted
    assert len(os.listdir(cache)) == len(full) - 1

    assert run_cv(LEARNERS, entries, n_folds=2, fit_timeout=None, random_state=0,
                  cache_dir=str(tmp_path / "empty"), cached_only=True).empty
    with pytest.raises(ValueError, match="cache_dir"):
        run_cv(LEARNERS, entries, n_folds=2, cached_only=True)


def test_run_cv_cached_only_reads_a_skipped_fold_as_timed_out(tmp_path):
    from pyrulearn.experiments.runner import _cache_key, _cache_save, timed_out
    from pyrulearn.data.representation import default_representation
    entry = _synthetic_entry("synth_a")
    learner, other = LEARNERS[0], LEARNERS[1]
    key = lambda fold, l: _cache_key(entry, 3, fold, 0, l, default_representation())
    base = {"dataset": entry.name, "accuracy": 0.9, "n_rules": 2, "n_conditions": 3, "fit_time": 1.0, "error": None}
    _cache_save(str(tmp_path), key(0, learner), {**base, "fold": 0, "learner": learner.display_name})
    _cache_save(str(tmp_path), key(1, learner), {**base, "fold": 1, "learner": learner.display_name,
                                                 "accuracy": None, "error": "timeout", "fit_time": 300.5})
    _cache_save(str(tmp_path), key(0, other), {**base, "fold": 0, "learner": other.display_name})
    # fold 2 of `learner` was skipped, so not cached; `other` simply hasn't run its later folds
    results = run_cv([learner, other], [entry], n_folds=3, fit_timeout=300.0, random_state=0,
                     cache_dir=str(tmp_path), cached_only=True)
    rows = results[results["learner"] == learner.display_name]
    assert list(rows["fold"]) == [0, 1, 2]
    assert (rows["error"] == "timeout").all() and rows["accuracy"].isna().all()
    assert list(results[results["learner"] == other.display_name]["fold"]) == [0]
    assert timed_out(results) == {(entry.name, learner.display_name)}


def test_timed_out_on_a_table_without_timeouts_and_the_closing_line(capsys):
    from pyrulearn.experiments.runner import timed_out
    entries = [_synthetic_entry("synth_a")]
    results = run_cv(LEARNERS, entries, n_folds=2, fit_timeout=None, random_state=0, verbose=True)
    assert timed_out(results) == set()
    assert "timed out:" not in capsys.readouterr().out
    assert timed_out(pd.DataFrame()) == set()


def test_run_cv_names_the_timed_out_pairs_at_the_end(capsys):
    entry = _synthetic_entry("synth_a", n=61)
    sizes = _fold_train_sizes(entry, 3)
    slow = next(s for s in sizes[1:] if s != sizes[0])
    learner = _SlowOnSize(slow, random_state=0)
    run_cv([learner], [entry], n_folds=3, fit_timeout=3.0, random_state=0, verbose=True)
    assert f"timed out: synth_a/{learner.display_name}" in capsys.readouterr().out


def test_run_cv_without_stop_on_timeout_counts_folds_on_their_own():
    entry = _synthetic_entry("synth_a", n=61)
    sizes = _fold_train_sizes(entry, 3)
    slow = next(s for s in sizes[1:] if s != sizes[0])
    results = run_cv([_SlowOnSize(slow, random_state=0)], [entry], n_folds=3, fit_timeout=3.0,
                     random_state=0, verbose=False, stop_on_timeout=False)
    by_fold = results.set_index("fold")
    assert by_fold.loc[0, "error"] is None or by_fold.loc[0, "error"] != by_fold.loc[0, "error"]
    assert by_fold.loc[0, "accuracy"] == by_fold.loc[0, "accuracy"]          # a real number
    assert (results["error"] == "timeout").sum() == sum(s == slow for s in sizes)


# ------------------------------------------------------------------- stats

@pytest.fixture(scope="module")
def cv_results():
    entries = [_synthetic_entry(f"synth_{i}", n=60, seed=i) for i in range(4)]
    return run_cv(LEARNERS, entries, n_folds=3, fit_timeout=None, random_state=0, verbose=False)


def test_mean_rank_ranks_every_learner(cv_results):
    ranks = mean_rank(cv_results, "accuracy", higher_is_better=True)
    assert set(ranks.index) == {"Pypper", "PFossil"}
    assert ranks.between(1, 2).all()
    assert list(ranks.index) == list(ranks.sort_values().index)  # best-first


def test_mean_rank_ties_a_failed_pair_for_last():
    results = pd.DataFrame({
        "dataset": ["d1", "d1", "d2", "d2"],
        "learner": ["A", "B", "A", "B"],
        "accuracy": [0.9, 0.8, np.nan, 0.7],
    })
    ranks = mean_rank(results, "accuracy", higher_is_better=True)
    # d1: A=1, B=2. d2: B=1 (A failed, tied for "last" among 2 -> rank 2)
    assert ranks["A"] == pytest.approx((1 + 2) / 2)
    assert ranks["B"] == pytest.approx((2 + 1) / 2)


def test_win_counts_counts_best_per_dataset_ties_split():
    results = pd.DataFrame({
        "dataset": ["d1", "d1", "d2", "d2", "d3", "d3"],
        "learner": ["A", "B", "A", "B", "A", "B"],
        "accuracy": [0.9, 0.8, 0.7, 0.9, 0.6, 0.6],  # d1: A wins, d2: B wins, d3: tie
    })
    wins = win_counts(results, "accuracy", higher_is_better=True)
    assert wins["A"] == pytest.approx(1.5)
    assert wins["B"] == pytest.approx(1.5)
    assert list(wins.index) == list(wins.sort_values(ascending=False).index)  # most wins first


def test_win_counts_a_dataset_where_everyone_failed_awards_nothing():
    results = pd.DataFrame({
        "dataset": ["d1", "d1", "d2", "d2"],
        "learner": ["A", "B", "A", "B"],
        "accuracy": [0.9, 0.8, np.nan, np.nan],
    })
    wins = win_counts(results, "accuracy")
    assert wins["A"] == pytest.approx(1.0)
    assert wins["B"] == pytest.approx(0.0)
    assert wins.sum() == pytest.approx(1.0)  # d2 contributed nothing


def test_friedman_test_runs_on_a_complete_matrix():
    # friedmanchisquare needs >=3 learners (treatments) -- cv_results only
    # has Pypper/PFossil, so this one uses a synthetic 3-learner table.
    results = pd.DataFrame({
        "dataset": [d for d in ("d1", "d2", "d3", "d4", "d5") for _ in range(3)],
        "learner": ["A", "B", "C"] * 5,
        "accuracy": [0.9, 0.8, 0.7, 0.85, 0.75, 0.65, 0.95, 0.9, 0.6, 0.8, 0.82, 0.7, 0.7, 0.9, 0.75],
    })
    result = friedman_test(results, "accuracy")
    assert result.n_datasets == 5
    assert result.n_learners == 3
    assert 0.0 <= result.pvalue <= 1.0


def test_friedman_test_needs_at_least_three_learners(cv_results):
    with pytest.raises(ValueError, match="needs >=3 learners"):
        friedman_test(cv_results, "accuracy")  # only Pypper/PFossil


def test_critical_difference_matches_demsar_table():
    # Demsar (2006), Table 5(b), alpha=0.05 -- q_alpha per k: 2->1.960,
    # 3->2.343, 10->3.164 (checked to 3 decimals; our q_alpha comes from
    # scipy's studentized_range, more precise than the published table).
    for k, q_alpha in ((2, 1.960), (3, 2.343), (10, 3.164)):
        cd = critical_difference(k, n_datasets=10, alpha=0.05)
        assert cd == pytest.approx(q_alpha * np.sqrt(k * (k + 1) / 60), abs=1e-3)
    cd_k2 = critical_difference(2, 10, alpha=0.05)
    cd_k10 = critical_difference(10, 10, alpha=0.05)
    assert cd_k10 > cd_k2  # more learners compared -> wider critical difference


def test_critical_difference_diagram_returns_an_axes(cv_results):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    ax = critical_difference_diagram(cv_results, "accuracy")
    assert ax.get_xlabel() == "mean rank (accuracy)"


def test_critical_difference_groups_are_maximal_and_may_overlap():
    # each learner joined to every other within one CD, by the longest runs only
    assert critical_difference_groups([3.68, 4.04, 4.15, 4.22, 4.53, 4.70, 5.31, 5.37], 1.24) == [(0, 5), (2, 7)]
    assert critical_difference_groups([2.76, 3.16, 3.22, 3.60, 3.79, 4.47], 0.66) == [(0, 2), (1, 4)]
    assert critical_difference_groups([1.0, 1.2, 1.4], 1.0) == [(0, 2)]
    assert critical_difference_groups([1.0, 2.0, 3.0], 0.5) == []


# ------------------------------------------------------------------ report

def test_render_setup_section_wraps_the_description():
    out = render_setup_section("  Some experiment description.  ")
    assert out == "## Setup\n\nSome experiment description.\n\n"


def test_render_results_table_has_an_overall_row_and_marks_failures():
    results = pd.DataFrame({
        "dataset": ["d1", "d1", "d2"],
        "learner": ["A", "A", "A"],
        "accuracy": [0.8, np.nan, 0.6],
    })
    out = render_results_table(results, ["accuracy"])
    assert "| dataset | accuracy |" in out
    assert "| overall |" in out
    assert "n/a" not in out  # d1's two rows average to a real number, not NaN

    all_nan = pd.DataFrame({"dataset": ["d1"], "learner": ["A"], "accuracy": [np.nan]})
    assert "n/a" in render_results_table(all_nan, ["accuracy"])


def test_render_results_table_group_by_list_breaks_out_every_column():
    results = pd.DataFrame({
        "dataset": ["d1", "d1", "d2", "d2"],
        "learner": ["A", "B", "A", "B"],
        "accuracy": [0.8, 0.6, 0.9, 0.7],
    })
    out = render_results_table(results, ["accuracy"], group_by=["dataset", "learner"])
    assert "| dataset | learner | accuracy |" in out
    assert "| d1 | A | 80.00% |" in out
    assert "| d1 | B | 60.00% |" in out
    assert "| d2 | A | 90.00% |" in out
    assert "| d2 | B | 70.00% |" in out
    assert "| overall |  | 75.00% |" in out  # padded blank for the second group_by column


def test_report_number_formats():
    from pyrulearn.experiments.report import format_accuracy, format_mean_count
    assert format_accuracy(0.86314) == "86.31%"
    assert format_accuracy(1.0) == "100.00%"
    assert format_accuracy(float("nan")) == "n/a"
    assert format_mean_count(12.345) == "12.3"
    assert format_mean_count(float("nan")) == "n/a"


def test_render_results_table_formats_counts_and_other_columns():
    results = pd.DataFrame({"dataset": ["d1", "d1"], "accuracy": [0.8, 0.9], "n_rules": [3, 4],
                            "fit_time": [0.25, 0.75]})
    out = render_results_table(results, ["accuracy", "n_rules", "fit_time"], include_overall=False)
    assert "| d1 | 85.00% | 3.5 | 0.500 |" in out


def test_render_results_table_include_overall_false_omits_the_row():
    results = pd.DataFrame({
        "dataset": ["d1", "d1"],
        "learner": ["A", "B"],
        "accuracy": [0.8, 0.6],
    })
    out = render_results_table(results, ["accuracy"], group_by="learner", include_overall=False)
    assert "| A | 80.00% |" in out
    assert "| B | 60.00% |" in out
    assert "overall" not in out
