"""
demos/deep_restarts.py
======================

Restarts of the stochastic deep rule learners (experimental branch).

A run of `DeepLocalSearch`, `DeepGenetic`, `DeepNeuroFuzzy` or
`DeepBlameRevision` ends in a local optimum, and runs differ. With
``restarts=9`` each fit makes ten runs, each from a new start of `init`,
and keeps the best network by the training objective (errors plus
`LAMBDA` per backbone edge). Here: one run against ten, per learner and
start -- a random one (the empty network for blame revision), Pypper's
rules (`RuleSetInit(Pypper())`), FOIL's (`FoilInit`: `PFoil` for the
network's class, at most one rule per first-layer node) -- on a network
``[32, 16, 8]``: the parity datasets (`parity_groups` with groups of two
and of three, parity-4), a few real ones, and KRK illegality. Pypper and
PFoil on their own are the references. The time limit per fit
(`MAX_SECONDS`) is long enough for ten runs; each fit's trace gives test
accuracy against seconds, the best network so far across the runs.

Run: `python demos/deep_restarts.py` (quick: two datasets, two folds,
short limits) or ``--full --n-jobs N``; writes
``deep_restarts_report.md`` and ``deep_restarts_plots/``. ``--report-only``
fits nothing (the report from the cache).
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

from pyrulearn.datasets import KRKIllegal, Parity, ParityGroups
from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.report import render_setup_section
from pyrulearn.experiments.runner import run_cv
from pyrulearn.learners.base import NativeRuleLearner
from pyrulearn.learners.deep import (DeepBlameRevision, DeepGenetic, DeepInit, DeepLocalSearch, DeepNeuroFuzzy,
                                     RuleSetInit)
from pyrulearn.learners.seco import PFoil, Pypper

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "deep_restarts"
CACHE_DIR = os.path.join(HERE, f"_{NAME}_cache")
PLOTS_DIR = os.path.join(HERE, f"{NAME}_plots")

LAYERS = [32, 16, 8]
LAMBDA = 1.0
RESTARTS = 9  # ten runs
FOLDS, QUICK_FOLDS = 5, 2
#: seconds per fit: enough for ten runs (one run of local search or the GA on KRK takes ~250 s)
MAX_SECONDS, QUICK_MAX_SECONDS = 3600, 20
FIT_TIMEOUT = 4000.0
DATASETS = [ParityGroups(4, 2, 2, n_irrelevant=4, n=1000), ParityGroups(4, 3, 2, n=1000), Parity(4, 6),
            "tic-tac-toe", "vote", "breast-cancer", "SPECT", KRKIllegal(1000)]
QUICK_DATASETS = [ParityGroups(4, 2, 2, n_irrelevant=4, n=1000), "tic-tac-toe"]
ALGORITHMS = ["local search", "genetic", "neuro-fuzzy", "blame revision"]
STARTS = ["random", "Pypper", "FOIL"]
REFERENCES = ["Pypper", "PFoil"]
COLORS = {"local search": "#2a78d6", "genetic": "#eb6834", "neuro-fuzzy": "#1baf7a", "blame revision": "#c2338a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
N_JOBS = 1
CACHED_ONLY = False


class FoilInit(DeepInit):
    """FOIL's rules as the start: `PFoil` for the class the network's
    (single) output learns, at most as many rules as the first layer has
    nodes (the first ones cover most), put in by `RuleSetInit`."""

    def weights(self, data, labels, shapes, rng, first="and"):
        if len(labels) != 1:
            raise ValueError("FoilInit is for one output (two classes)")
        rules = PFoil(target_class=labels[0], max_rules=shapes[0][0]).fit(data)
        return RuleSetInit(rules).weights(data, labels, shapes, rng, first)


def make(algorithm: str, start: str, restarts: int, seconds: float):
    kw = dict(layers=LAYERS, size_penalty=LAMBDA, max_seconds=seconds, restarts=restarts, trace=True)
    if start == "Pypper":
        kw["init"] = RuleSetInit(Pypper())
    elif start == "FOIL":
        kw["init"] = FoilInit()
    if algorithm == "local search":
        return DeepLocalSearch(**kw)
    if algorithm == "genetic":
        return DeepGenetic(**kw)
    if algorithm == "neuro-fuzzy":
        return DeepNeuroFuzzy(**kw)
    return DeepBlameRevision(batch_size=256, soft_weight=0.25, **kw)


def _label(algorithm: str, start: str, restarts: int) -> str:
    return f"{algorithm} from {start} ({'10 runs' if restarts else 'one run'})"


class Variant(NativeRuleLearner):
    """`algorithm` from `start` with ``restarts`` (0: one run), or a
    reference learner (``algorithm`` "Pypper" or "PFoil")."""

    def __init__(self, algorithm: str, start: str = "random", restarts: int = 0, seconds: float = MAX_SECONDS):
        self.algorithm, self.start, self.restarts, self.seconds = algorithm, start, restarts, seconds

    @property
    def display_name(self) -> str:
        if self.algorithm in REFERENCES:
            return self.algorithm
        return _label(self.algorithm, self.start, self.restarts)

    def _provenance_params(self):
        if self.algorithm in REFERENCES:
            return {"algorithm": self.algorithm}
        return {"algorithm": self.algorithm, "start": self.start, "restarts": self.restarts,
                "seconds": self.seconds, "layers": LAYERS, "lambda": LAMBDA}

    def fit(self, data):
        if self.algorithm == "Pypper":
            return Pypper().fit(data)
        if self.algorithm == "PFoil":
            return PFoil().fit(data)
        return make(self.algorithm, self.start, self.restarts, self.seconds).fit(data)


def measure(model, test_rep) -> dict:
    """Every trace point: seconds, test accuracy and backbone edges of the
    best network so far."""
    trace = getattr(model, "trace", None)
    if not trace:
        return {}
    points = []
    for i, p in enumerate(trace):
        net = model.trace_model(i)
        points.append([p.seconds, float((net.predict(test_rep) == test_rep.y).mean()), net.pruned().n_edges])
    return {"trace": points, "edges": model.pruned().n_edges}


def run(quick: bool) -> pd.DataFrame:
    catalog = Catalog.default()
    datasets = [catalog.select(names=[d])[0] if isinstance(d, str) else d
                for d in (QUICK_DATASETS if quick else DATASETS)]
    seconds = QUICK_MAX_SECONDS if quick else MAX_SECONDS
    learners = [Variant(r) for r in REFERENCES] + [
        Variant(a, s_, r, seconds) for a in ALGORITHMS for s_ in STARTS for r in (0, RESTARTS)]
    return run_cv(learners, datasets, n_folds=QUICK_FOLDS if quick else FOLDS, fit_timeout=FIT_TIMEOUT,
                  random_state=0, cache_dir=CACHE_DIR, measure_fn=measure, n_jobs=N_JOBS, cached_only=CACHED_ONLY)


# -- the report -----------------------------------------------------------------

def _plot(res: pd.DataFrame, quick: bool) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    lines = []
    for ds in dict.fromkeys(res["dataset"]):
        sub = res[res["dataset"] == ds]
        t_max = max((pt[0] for tr in sub["trace"].dropna() for pt in tr), default=1.0) if "trace" in sub else 1.0
        grid = np.geomspace(0.01, max(t_max, 0.02) * 1.05, 300)
        fig, axes = plt.subplots(1, len(STARTS), figsize=(5.2 * len(STARTS) + 2.0, 4.6), sharey=True, facecolor=SURF)
        refs = {r: sub[(sub["learner"] == r) & sub["error"].isna()]["accuracy"].mean() for r in REFERENCES}
        for ax, start in zip(axes, STARTS):
            ax.set_facecolor(SURF)
            for algorithm in ALGORITHMS:
                for restarts, style in ((0, "--"), (RESTARTS, "-")):
                    rows = sub[(sub["learner"] == _label(algorithm, start, restarts)) & sub["error"].isna()]
                    curves = []
                    for tr in rows.get("trace", pd.Series(dtype=object)):
                        if not isinstance(tr, list) or not tr:
                            continue
                        t = np.array([p[0] for p in tr])
                        v = np.array([p[1] for p in tr])
                        idx = np.searchsorted(t, grid, side="right") - 1
                        curves.append(np.where(idx >= 0, v[np.clip(idx, 0, None)], np.nan))
                    if not curves:
                        continue
                    stack = np.vstack(curves)
                    enough = np.sum(~np.isnan(stack), axis=0) >= max(1, len(curves) // 2 + 1)
                    with np.errstate(all="ignore"):
                        mean = np.where(enough, np.nanmean(stack, axis=0), np.nan)
                    ax.step(grid, mean, where="post", color=COLORS[algorithm], lw=2, ls=style,
                            label=f"{algorithm} ({'10 runs' if restarts else 'one run'})")
            for k, (r, value) in enumerate(refs.items()):
                if not np.isnan(value):
                    ax.axhline(value, color=INK2, lw=1, ls=("-", ":")[k])
                    ax.annotate(f"{r} {value:.3f}", (0.0, value), xytext=(4, (4, -12)[k]), xycoords=("axes fraction", "data"),
                                textcoords="offset points", color=INK2, fontsize=8.5)
            ax.set_xscale("log")
            ax.set_xlabel("seconds (log scale)", color=INK2, fontsize=9)
            ax.set_title(f"from {start}" + (" (empty for blame revision)" if start == "random" else ""),
                         color=INK, fontsize=10, loc="left")
            ax.grid(True, axis="y", color=GRID, lw=1)
            ax.set_axisbelow(True)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            for s_ in ("left", "bottom"):
                ax.spines[s_].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=8.5)
        handles, labels = max((a.get_legend_handles_labels() for a in axes), key=lambda hl: len(hl[0]))
        fig.legend(handles, labels, frameon=False, fontsize=8, loc="center right")
        fig.suptitle(f"{ds}: test accuracy of the best network so far, mean over folds "
                     f"(solid: ten runs, dashed: one run; {LAYERS}, lambda {LAMBDA:g})",
                     x=0.01, ha="left", color=INK, fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1 - 2.0 / (5.2 * len(STARTS) + 2.0), 0.93))
        path = os.path.join(PLOTS_DIR, f"{NAME}{'_quick' if quick else ''}_{ds}.png")
        fig.savefig(path, dpi=130, facecolor=SURF)
        plt.close(fig)
        lines.append(f"![{ds}]({os.path.basename(PLOTS_DIR)}/{os.path.basename(path)})\n")
    return lines


def write_report(res: pd.DataFrame, quick: bool, path: str) -> None:
    folds = QUICK_FOLDS if quick else FOLDS
    seconds = QUICK_MAX_SECONDS if quick else MAX_SECONDS
    description = f"""
Each learner as one run and with `restarts={RESTARTS}` (ten runs, each from a new start, the best
network by the training objective kept), from three starts: random (`RandomInit`; the empty
network for blame revision), Pypper's rules (`RuleSetInit(Pypper())`) and FOIL's (`FoilInit`:
`PFoil` for the network's class, at most one rule per first-layer node). Learners:
`DeepLocalSearch`, `DeepGenetic`, `DeepNeuroFuzzy` (their default settings) and
`DeepBlameRevision` (batch 256, soft weight 0.25). Network `{LAYERS}`, `size_penalty={LAMBDA:g}`
(errors plus {LAMBDA:g} per edge), at most {seconds} s per fit (`max_seconds`; the runs end early
otherwise), the fit capped at {FIT_TIMEOUT:.0f} s. Stratified {folds}-fold cross-validation;
Pypper and PFoil (its defaults: rules for both classes) on their own as references. Times are
wall-clock.
"""
    ok = res[res["error"].isna()]
    acc = ok.groupby(["dataset", "learner"])["accuracy"].mean()
    edges = ok.groupby(["dataset", "learner"])["edges"].mean() if "edges" in ok else pd.Series(dtype=float)
    times = ok.groupby(["dataset", "learner"])["fit_time"].mean()
    datasets = list(dict.fromkeys(res["dataset"]))
    columns = [(s_, r) for s_ in STARTS for r in (0, RESTARTS)]

    def cell(ds, name, with_edges=True):
        if (ds, name) not in acc.index:
            return "failed" if ((res["dataset"] == ds) & (res["learner"] == name)).any() else "-"
        e = edges.get((ds, name), np.nan) if with_edges else np.nan
        return f"{100 * acc[(ds, name)]:.2f}" + ("" if pd.isna(e) else f" ({e:.0f})")

    head = " | ".join(f"from {s_}: {'10' if r else '1'}" for s_, r in columns)
    L = [f"# Restarts of the deep rule learners{' (quick run)' if quick else ''}\n",
         render_setup_section(description),
         "## Test accuracy\n",
         "Mean test accuracy (%), the backbone's edges in brackets; per start one run (1), then ten "
         "(10). The first row of a dataset: the references on their own.\n",
         f"| dataset | learner | {head} |", "|---|---|" + "--:|" * len(columns)]
    for ds in datasets:
        L.append(f"| {ds} | *references* | " + " | ".join(
            [f"Pypper {cell(ds, 'Pypper', False)}", f"PFoil {cell(ds, 'PFoil', False)}"]
            + [""] * (len(columns) - 2)) + " |")
        for a in ALGORITHMS:
            L.append(f"| | {a} | " + " | ".join(cell(ds, _label(a, s_, r)) for s_, r in columns) + " |")
    L += ["", "Mean over the datasets:\n", f"| learner | {head} |", "|---|" + "--:|" * len(columns)]
    for a in ALGORITHMS:
        vals = []
        for s_, r in columns:
            v = [acc[(ds, _label(a, s_, r))] for ds in datasets if (ds, _label(a, s_, r)) in acc.index]
            vals.append(f"{100 * np.mean(v):.2f}" if len(v) == len(datasets) else "-")
        L.append(f"| {a} | " + " | ".join(vals) + " |")
    ref = [f"{r} {100 * np.mean([acc[(ds, r)] for ds in datasets if (ds, r) in acc.index]):.2f}" for r in REFERENCES]
    L += ["", f"References: {', '.join(ref)}.\n",
          "## Fit time\n", "Mean seconds per fit (ten runs end early where their search does).\n",
          f"| dataset | learner | {head} |", "|---|---|" + "--:|" * len(columns)]
    for ds in datasets:
        for k, a in enumerate(ALGORITHMS):
            L.append(f"| {ds if k == 0 else ''} | {a} | " + " | ".join(
                f"{times[(ds, _label(a, s_, r))]:.0f}" if (ds, _label(a, s_, r)) in times.index else "-"
                for s_, r in columns) + " |")
    L += ["", "## Over time\n",
          "The test accuracy of the best network so far (by the training objective, across the runs), "
          "mean over the folds; a panel per start.\n"]
    L += _plot(res, quick)
    failed = res[res["error"].notna()].groupby(["dataset", "learner"]).size()
    if len(failed):
        L.append("Failed folds: " + "; ".join(f"{d} / {l}: {k}" for (d, l), k in failed.items()) + ".\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Report -> {path}")


def main(quick: bool) -> None:
    res = run(quick)
    write_report(res, quick, os.path.join(HERE, f"{NAME}_{'quick_' if quick else ''}report.md"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="every dataset, five folds, the full time limits")
    parser.add_argument("--n-jobs", type=int, default=1, help="fits to run at the same time")
    parser.add_argument("--report-only", action="store_true", help="fit nothing: the report from the cache")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    CACHED_ONLY = args.report_only
    main(quick=not args.full)
