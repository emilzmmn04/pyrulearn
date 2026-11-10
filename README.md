# pyrulearn

[![CI](https://github.com/juffif/pyrulearn/actions/workflows/ci.yml/badge.svg)](https://github.com/juffif/pyrulearn/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

A Python library for **representing, learning, combining and analyzing
propositional rule models** over Boolean data. It provides a common model
hierarchy (rule sets, decision lists, concept sets, ensembles), a shared
data representation, pluggable rule-evaluation heuristics and combiners,
[native implementations of a range of rule learners](#natively-implemented) (a
configurable separate-and-conquer (SeCo) framework and the algorithms that
are instantiations of it, such as CN2, AQR, PFOIL, FOSSIL and Pypper, a
RIPPER re-implementation, with removal or weighted covering; plus locally
optimal rules, the associative classifiers CBA, CMAR, CPAR and IDS, the rule
ensembles SLIPPER, LRI, ENDER, BOOMER and optimal rule boosting, and a
RuleFit-style linear model), and
[interfaces to external learners](#interfaced-external-implementations) (scikit-learn, wittgenstein,
imodels, Weka, LORD, pyarc, mlrl-boomer, realkd), so that native and external
algorithms can be run and compared through one API.

> **Status: early development (alpha).** The APIs are still changing, and
> some demos need revising. Changes between versions are listed in the
> [changelog](CHANGELOG.md).

New here? [`tutorial/tutorial.ipynb`](tutorial/tutorial.ipynb) (also as a
[PDF](tutorial/tutorial.pdf)) walks through loading your own data from a
CSV, fitting a native learner, inspecting the rules it found, and comparing
two learners across a few benchmark datasets.

## Contents

- [Install](#install)
- [Algorithms](#algorithms)
  - [Natively implemented](#natively-implemented)
  - [Interfaced (external implementations)](#interfaced-external-implementations)
- [Module map](#module-map)
- [Data representation](#data-representation)
  - [Row weights](#row-weights)
  - [Typed attributes](#typed-attributes)
  - [Merging DataSpecs](#merging-dataspecs)
    - [Rebasing rules onto a different DataSpec](#rebasing-rules-onto-a-different-dataspec)
  - [Reading ARFF / CSV data](#reading-arff--csv-data)
  - [Missing values](#missing-values)
- [Rule models](#rule-models)
  - [Printing a model](#printing-a-model)
  - [Converting between model types](#converting-between-model-types)
  - [Conflict resolution](#conflict-resolution)
  - [Default predictions](#default-predictions)
  - [Explaining a prediction](#explaining-a-prediction)
  - [Statistics](#statistics)
  - [Rule-evaluation heuristics](#rule-evaluation-heuristics)
  - [Coverage space](#coverage-space)
    - [Isometrics: pencil, parallel and curved](#isometrics-pencil-parallel-and-curved)
    - [CoverageSpace: layering heuristics, rulesets, and refinement paths](#coveragespace-layering-heuristics-rulesets-and-refinement-paths)
- [Learning algorithms](#learning-algorithms)
  - [Native learning algorithms](#native-learning-algorithms)
    - [Separate-and-conquer (SeCo)](#separate-and-conquer-seco)
    - [Weighted covering](#weighted-covering)
    - [Boosting](#boosting)
  - [Interfaced learning algorithms](#interfaced-learning-algorithms)
    - [What each tool receives](#what-each-tool-receives)
    - [Importer base classes](#importer-base-classes)
    - [Text formats: `StringRuleImporter`](#text-formats-stringruleimporter)
    - [scikit-learn: decision trees and random forests](#scikit-learn-decision-trees-and-random-forests)
    - [wittgenstein: IREP and RIPPER](#wittgenstein-irep-and-ripper)
    - [imodels: Bayesian rule lists, rule sets, RuleFit and Slipper](#imodels-bayesian-rule-lists-rule-sets-rulefit-and-slipper)
    - [weka: JRip, PART and J48](#weka-jrip-part-and-j48)
    - [LORD: the Java reference implementation](#lord-the-java-reference-implementation)
    - [pyarc: CBA](#pyarc-cba)
    - [BOOMER and realkd: boosted rule ensembles](#boomer-and-realkd-boosted-rule-ensembles)
    - [Provenance of imported rules](#provenance-of-imported-rules)
  - [Multiclass classification](#multiclass-classification)
- [Experiments](#experiments)
  - [Benchmark datasets](#benchmark-datasets)
  - [Cross-validation runner](#cross-validation-runner)
  - [Statistics and reports](#statistics-and-reports)
  - [Comparisons built on it](#comparisons-built-on-it)
- [Try it](#try-it)
- [Not yet implemented (left as clear extension points)](#not-yet-implemented-left-as-clear-extension-points)
- [Authors and contributions](#authors-and-contributions)
- [How to cite](#how-to-cite)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Install

```
pip install git+https://github.com/juffif/pyrulearn.git
```

Requires Python ≥ 3.10, `numpy` and `scikit-learn` (the latter because
`RuleSetClassifier` inherits from `sklearn.base.BaseEstimator`, hence
`import pyrulearn` itself needs it). Optional extras:

| Extra | Installs | Needed for |
|---|---|---|
| `data` | `pandas`, `scipy` | `pyrulearn.data.io` (reading ARFF/CSV; `scipy` for ARFF and sparse representations) |
| `plot` | `matplotlib`, `networkx` | coverage-space and refinement-graph plotting in `pyrulearn.evaluation` |
| `experiments` | `pandas`, `scipy`, `matplotlib`, `psutil` | `pyrulearn.experiments` (the dataset catalog, cross-validation runner, ranking statistics and report helpers behind the comparison scripts in `demos/`; `psutil` to stop a timed-out fit's child processes on Linux and macOS) |
| `wittgenstein` | `wittgenstein` | `pyrulearn.interfaces.wittgenstein` (IREP, RIPPER) |
| `imodels` | `imodels` | `pyrulearn.interfaces.imodels` (Bayesian rule lists / sets, RuleFit, Slipper) |
| `pyarc` | `pyarc` | `pyrulearn.interfaces.pyarc` (CBA); `pyarc` itself also needs Borgelt's `pyfim` C extension, no Windows wheels but builds there with MinGW-w64 -- see the module docstring |
| `boomer` | `mlrl-boomer` | `pyrulearn.interfaces.boomer` (BOOMER); its declared scikit-learn pin isn't enforced at runtime -- install with `--no-deps`, see the module docstring |
| `realkd` | `realkd` | `pyrulearn.interfaces.realkd` (optimal rule boosting); needs `sortednp`, no Windows wheel but builds there with MinGW-w64 -- see the module docstring |
| `notebook` | `jupyter`, `nbconvert`, `ipykernel` | only to *regenerate* `tutorial/tutorial.ipynb`/`.pdf` -- reading the committed files needs nothing extra. PDF export also needs a LaTeX install (e.g. MiKTeX on Windows) and `pandoc`. |
| `test` | `pytest` | running the test suite |
| `all` | all of the above except `pyarc`, `boomer` and `realkd` | |

For example `pip install "pyrulearn[data,plot] @ git+https://github.com/juffif/pyrulearn.git"`.
For development, clone the repository and run `pip install -e ".[all]"`, then `pytest`.
The Weka, LORD and pyarc interfaces additionally need those external
tools (Java and `weka.jar`, LORD's Java implementation, `pyarc` with `pyfim`);
their tests skip themselves when the tool is not available.

## Algorithms

All learners below share the `learner.fit(data)` entry point (see
*Learning algorithms*); multiclass decompositions
(`model=ConceptSet | ConceptCascade | PairwiseModel`) work for the native
and most of the external learners. Importers on their own convert an
already-fitted external model (or its text output) into a `RuleModel`.

### Natively implemented

| Algorithm | Class (in `pyrulearn.learners`) | Notes | Reference |
|---|---|---|---|
| **SeCo framework** | `seco.SeCo` | the separate-and-conquer (covering) engine that the next five entries are instantiations of: a per-class covering loop around composable building blocks (search, heuristics, pruning, stopping, optimization, covering: removal or weighted) | Fürnkranz, Gamberger & Lavrač 2012; Fürnkranz & Flach 2005 |
| **CN2** | `seco.CN2` | a `SeCo` instantiation: Laplace heuristic and likelihood-ratio significance test; unlike the rest of the family, its default `ConceptSet` resolves a clash between firing rules by summing their covered-class distributions (`MicroVoteCombiner`), not the family's generic `HeuristicMaxCombiner()`, matching Clark & Boswell's own unordered-CN2 | Clark & Niblett 1989; Clark & Boswell 1991 |
| **AQR** | `seco.AQR` | a `SeCo` instantiation: Clark & Niblett's reimplementation of Michalski's AQ; the literal *star* search is approximated by a seed-restricted beam search | Clark & Niblett 1989 |
| **PFOIL** | `seco.PFoil` | a `SeCo` instantiation: propositional FOIL with information gain, hill climbing, MDL-based encoding-length restriction | Mooney 1995; Quinlan 1990 |
| **FOSSIL** | `seco.PFossil` | a `SeCo` instantiation: correlation heuristic with a quality threshold | Fürnkranz 1994 |
| **Pypper** | `seco.Pypper` | a `SeCo` instantiation: a re-implementation of RIPPER, not a port of Cohen's code. IREP\* growth and pruning plus the replace/revise optimization phase, per class, least-frequent class first. It differs from the original in places: the covering loop stops on FOIL's MDL restriction or IREP's precision below 0.5 instead of Cohen's 64-bit description-length rule, there is no residual IREP\* pass after optimization, and pruning keeps the prefix with the best Laplace estimate rather than Cohen's (p-n)/(p+n) -- as Weka's JRip does, and more accurate on average | Cohen 1995; Fürnkranz & Widmer 1994 |
| **SLIPPER** | `boosting.Slipper` | confidence-rated boosting of rules: each round a rule is grown with `SlipperZ` and pruned on a held-out split, gets a confidence, and reweights the examples (`AdaBoostReweighting`); the model is a `LinearRuleModel`. Fixed number of rounds instead of the original's internal cross-validation | Cohen & Singer 1999 |
| **BOOMER** (single-label) | `boosting.Boomer` | `ENDER` with BOOMER's settings: logistic loss, L2-regularized Newton steps (λ = 1), shrinkage 0.3, up to 1000 rules; predicts like the original on binary data (see *Interfaced*); multi-label BOOMER is not done. `boosting.DenseBoomer`: the faster dense-matrix version, same models | Rapp et al. 2020 |
| **Optimal rule boosting** | `boosting.ORB`, `boosting.DenseORB` | gradient boosting of rules that each maximize the XGBoost-style gain `(Σg)² / (λ + Σh)` (the `XGBGain` objective), found by any rule search -- a beam search by default, exactly by branch-and-bound with the paper's prefix/suffix bound; `DenseORB` is the dense-matrix specialization; both cross-checked rule-for-rule against `realkd` | Boley, Teshuva, Le Bodic & Webb 2021 |
| **ENDER** / MLRules | `boosting.ENDER` | boosting of rules by forward stagewise loss minimization: each rule, grown greedily on a subsample to minimize an impurity derived from the loss (constant-step, gradient descent, gradient boosting, simultaneous minimization, or MLRules' Newton criterion), votes for one class with a shrunk weight computed on all rows; pluggable loss (`LogisticLoss`, default, multiclass; `ExponentialLoss`; `SigmoidLoss`) and impurity criterion (`ImpurityCriterion`); a `LinearRuleModel`; runs on every data representation. `boosting.DenseENDER`: the faster dense-matrix version, same models | Dembczyński, Kotłowski & Słowiński 2008, 2010 |
| **LRI** | `lri.LRI` | Lightweight Rule Induction: the same number of unweighted DNF rules per class, each grown term by term minimizing the weighted error `FP + k·FN` without pruning, with cases reweighted by the rules' cumulative errors (`LRIReweighting`); the class with the most satisfied rules wins | Weiss & Indurkhya 2000 |
| **LORD** (simplified, `PyLORD`) | `pylord.PyLORD` | locally optimal rules, built from the `SeCo` building blocks but not a covering loop: every training example seeds a rule search. A simplified reimplementation, not the reference one (see *Interfaced* for that) | Huynh, Fürnkranz & Beck 2023 |
| **OPUS** | `opus.Opus`, `opus.OpusTopK` | optimal rules by exhaustive branch-and-bound search (`seco.BranchAndBoundSearch`): `Opus` covers with the best rule by the heuristic in every step (a `SeCo` configuration); `OpusTopK` finds the `k` best rules per class (k-optimal rule discovery in the manner of Magnum Opus) | Webb 1995 |
| **Class association rule mining** | `associative.CARMiner` | Apriori-style CBA-RG; returns a compact, lazily materialized `PooledRuleSet` | Liu et al. 1998; Agrawal & Srikant 1994 |
| **CBA** | `associative.CBA` | CBA-CB (M1) classifier building on top of a rule pool; cross-checked rule-for-rule against `pyarc` | Liu et al. 1998 |
| **CMAR** | `associative.CMAR` | simplified: chi-square significance filter, per-class coverage pruning, weighted chi-square voting | Li et al. 2001 |
| **CPAR** | `cpar.CPAR` | FOIL-gain rule growing on weighted examples that also follows every nearly-as-good condition (several rules per search), covered positives decayed instead of removed; predicts by the mean expected accuracy of each class's best `k` covering rules (`TopKMeanCombiner`). Built from exchangeable components; `cpar.DenseCPAR` is a faster, Boolean-only implementation learning the same models | Yin & Han 2003 |
| **IDS** | `ids.IDS` | interpretable decision sets: submodular objective, `SmoothLocalSearch` or `GreedySelection` as `optimizer=`, optional coordinate-ascent tuning of the weights | Lakkaraju et al. 2016 |
| **RuleFit** (distiller) | `rulefit.RuleFit` | a sparse (L1 / elastic-net) logistic regression over a rule pool's coverage, multinomial for multiclass; returns a `LinearRuleModel`. Only RuleFit's fitting step: candidates come from the pool, not from a tree ensemble | Friedman & Popescu 2008 |
| **Bagging** | `bagging.Bagging` | members of any learner, each from a bootstrap sample (the drawn rows weighted by their numbers of draws); an `EnsembleModel` voting by any rule combiner over the members' deciding rules, members abstaining where their rules don't cover an example | Breiman 1996 |
| **AdaBoost** (SAMME) | `adaboost.AdaBoost` | members of any learner, each from the examples reweighted by its predecessors' errors, voting with SAMME's weights; `early_stopping=False` restarts from a bootstrap sample where SAMME would stop | Freund & Schapire 1997; Zhu et al. 2009 |
| **Multiclass decomposition** | `multiclass.OneVsRest`, `OrderedOneVsRest`, `Pairwise` | one-vs-rest, ordered (peeling) and round-robin decomposition for any binary-capable learner | Fürnkranz 2002 |

`CBA`, `CMAR`, `IDS` and `RuleFit` are *rule distillers*: each consumes any pool of
rules given as a `FlatRuleSet` (`rules=`) — mined by
`CARMiner` by default, but equally one extracted from a
random forest — and returns its own, much smaller model.

### Interfaced (external implementations)

| Algorithm | Class(es) (in `pyrulearn.interfaces`) | Requires | Reference |
|---|---|---|---|
| **Decision tree** (CART) | `sklearn.SKLDecisionTree`, `SklearnTreeImporter` | `scikit-learn` | Breiman et al. 1984 |
| **Random forest** | `sklearn.SKLRandomForest`, `RandomForestImporter` | `scikit-learn` | Breiman 2001 |
| **IREP** | `wittgenstein.IREP`, `IREPImporter` | `wittgenstein` | Fürnkranz & Widmer 1994 |
| **RIPPER** | `wittgenstein.WittRIPPER`, `RIPPERImporter` | `wittgenstein` | Cohen 1995 |
| **Bayesian Rule Lists** | `imodels.IModBayesianRuleList`, `BayesianRuleListImporter` | `imodels` | Letham et al. 2015 |
| **Bayesian Rule Sets** | `imodels.IModBayesianRuleSet`, `BayesianRuleSetImporter` | `imodels` | Wang et al. 2017 |
| **RuleFit** | `imodels.IModRuleFit`, `RuleFitImporter` | `imodels` | Friedman & Popescu 2008 |
| **"SLIPPER"** (AdaBoost over imodels' rule learner) | `imodels.IModSlipper`, `SlipperImporter` | `imodels` | Cohen & Singer 1999 (loosely) |
| **JRip** (Weka's RIPPER) | `weka.WekaJRip`, `JRipImporter` | Java, `weka.jar` (`$WEKA_JAR`) | Cohen 1995 |
| **PART** | `weka.WekaPART`, `PARTImporter` | Java, `weka.jar` | Frank & Witten 1998 |
| **J48** (Weka's C4.5) | `weka.WekaJ48`, `J48Importer` | Java, `weka.jar` | Quinlan 1993 |
| **LORD** (reference implementation) | `lord.JavaLORD`, `LORDImporter` | LORD's Java implementation ([vqphuynh/LORD](https://github.com/vqphuynh/LORD)) | Huynh, Fürnkranz & Beck 2023 |
| **CBA** | `pyarc.PArcCBA`, `PyarcCBAImporter` | `pyarc` and Borgelt's `pyfim` | Liu et al. 1998 |
| **BOOMER** (binary) | `boomer.MLRLBoomer`, `BoomerImporter` | `mlrl-boomer` | Rapp et al. 2020 |
| **Optimal rule boosting** (reference) | `realkd.RKDRuleBoosting`, `RealkdImporter` | `realkd` | Boley et al. 2021 |

Many of the ideas behind this library, such as the separate-and-conquer
algorithms and rule-evaluation heuristics, are described in Fürnkranz,
Gamberger & Lavrač, *Foundations of Rule Learning* (Springer, 2012). Complete bibliographic entries are in
[`references.bib`](references.bib).

## Module map

One row per package or top-level module, in alphabetical order; submodules
are named inline. The module docstrings and the sections below carry the
details.

| Module (in `pyrulearn`) | What it's for |
|---|---|
| `combiners` | `RuleCombiner`: how a `RuleSet` resolves an example covered by several rules. List order, plain majority vote, heuristic-scored max or vote, and per-class-distribution combiners (`MacroVoteCombiner` reproduces scikit-learn's soft voting). |
| `data` | Everything about data. **Four base representations**, all behind the same `coverage(rule)` / `features_of(row)` interface, so every rule learner runs on any of them and finds identical rules: `VerticalBitmapRepresentation` (one bit-vector per feature over the examples; the fastest for most native learners and as small as a bit matrix gets, so first in `REPRESENTATION_PREFERENCE_ORDER`, which decides what `data.io`'s loaders and `run_cv` build by default), `HorizontalBitmapRepresentation` (the same bits packed per example, the simplest to inspect directly), `SparseDataRepresentation` (scipy CSR/CSC, Eclat-style tid-lists) and `NListRepresentation` (the PPC-tree / N-list index of LORD, stored instead of the table; `PrePostNListRepresentation` is a variant). Submodules: `data.attributes` (typed attributes -- boolean, binary, nominal, numeric, set, hierarchical, relational -- and the derived Boolean features they generate, `color=red`, `age>=30`, ...; also the **constraints** among those features, `ExactlyOne`, `ThresholdChain`, `MutuallyExclusive`, `Implies`, which record what is impossible or already implied: rule search uses them to skip contradictory refinements and to drop features an added condition already determines, which **reduces the search space**, and they let a rule check its own consistency; plus `evaluate_feature` and `MissingStrategy`), `data.spec` (`DataSpec`, `DataSpecBuilder`, `merge_dataspecs`: the feature space, no data), `data.representation` (the representations above) `data.io` (ARFF/CSV reading and writing, `encode`/`binarize`, `build_dataspec`; needs `pandas`). |
| `datasets` | Generated datasets, defined by a rule instead of downloaded: `krk_illegal(n, random_state)`, the KRK illegality task of the ILP literature (white king, white rook, black king, white to move; six nominal coordinates, ~1/3 illegal), and `KRKIllegal` -- the same as a dataset for `run_cv`. Learning it from the coordinates alone needs relations such as "same file" or "adjacent", the ILP systems' background knowledge, built from the features; `relations=True` adds that background knowledge as 24 relational features (`eq`, `adj`, `lt` of two pieces' files or ranks), which give an exact theory, and `coordinates=False` leaves out the coordinates. `parity(n_relevant, n_irrelevant, n, noise)`: the parity of a few relevant Boolean features among irrelevant ones (a flat DNF needs 2^(k-1) rules, a deep network small XOR/XNOR parts); `Parity` for `run_cv`. |
| `evaluation` | Measured statistics (`RuleStats`, `ConfusionMatrix`, `ModelStats`), `sort_rules`, `summarize`, and coverage-space plotting (`CoverageSpace`, `coverage_space_plot`, `coverage_space_auc`, `rule_refinement_plot`, `build_refinement_graph`). |
| `experiments` | Shared infrastructure for demos and other cross-validation-style experiments: `experiments.catalog` (the catalog of benchmark datasets, see *Benchmark datasets*), `experiments.runner` (`run_cv`: per-fold encoding into `representation=` (default: the preference order's first entry, `VerticalBitmapRepresentation`), a per-fit timeout via `TimeoutRunner`, uniform measurement, fold caching), `experiments.stats` (optional `mean_rank`, `friedman_test`, `critical_difference_diagram`, never run automatically) and `experiments.report` (small Markdown-writing building blocks). Needs `pandas`/`scipy`/`matplotlib` (the `experiments` extra). |
| `heuristics` | `RuleHeuristic`: pluggable rule-evaluation heuristics (`Precision`, `Laplace`, `MEstimate`, `WRAcc`, `FoilGain`, `Correlation`, `Entropy`, `LikelihoodRatio`, ...), the composable `LEF`, and `plot_isometrics` for drawing a heuristic into a `CoverageSpace`. |
| `interfaces` | Bringing external rule models in. `interfaces.base` has the shared `RuleImporter` machinery (`ObjectRuleImporter`, `StringRuleImporter`, the importer registry, `PatternStringImporter`); each external tool then has its own submodule, pairing an importer with a learner wrapper: `interfaces.sklearn` (decision trees, random forests, and `RuleSetClassifier`, which wraps any `RuleModel` as a scikit-learn estimator), `interfaces.wittgenstein` (IREP, RIPPER), `interfaces.imodels` (Bayesian rule lists and sets, RuleFit, Slipper), `interfaces.weka` (JRip, PART, J48), `interfaces.lord` (the reference LORD implementation), `interfaces.pyarc` (CBA), `interfaces.boomer` (BOOMER) and `interfaces.realkd` (optimal rule boosting). |
| `learners` | Turning data into rules through one `fit(data, model=None) -> RuleModel`. `learners.base` has the shared `RuleLearner` classes, including the `DecomposingLearner` multiclass switcher. Native algorithms: `learners.seco` (the `SeCo` framework and `CN2`, `AQR`, `PFoil`, `PFossil`, `Pypper`), `learners.pylord` (`PyLORD`), `learners.associative` (`CARMiner`, the `RuleDistiller` mixin, and the `CBA` and `CMAR` classifiers built on it), `learners.ids` (`IDS`), `learners.rulefit` (`RuleFit`), `learners.boosting` (`Slipper`, `ENDER`, `DenseENDER`, `Boomer`, `DenseBoomer`, `ORB`, `DenseORB`), `learners.lri` (`LRI`, `DenseLRI`), `learners.cpar` (`CPAR`, `DenseCPAR`), `learners.opus` (`Opus`, `OpusTopK`), `learners.deep` (`DeepLocalSearch`, `DeepGenetic`, `DeepSAT`, `DeepNeuroFuzzy`), and `learners.multiclass` (`OneVsRest`, `OrderedOneVsRest`, `Pairwise`). |
| `models` | The `RuleModel` hierarchy, organised by how a prediction is resolved: `RuleSet` (`FlatRuleSet`, `ConceptModel`, `ConceptSet`, `DisjointRuleSet`, and the memory-compact `PooledRuleSet` that `CARMiner` returns), `RuleList` (`DecisionList`, `ConceptCascade`), `CompositeModel` (`EnsembleModel`, `PairwiseModel`, `DeepModel`) and `SingleRule`. Also the `default_prediction` policy, per-model `stats`, `Provenance`, `annotate_rules`, and the model-to-model converters. |
| `pruning` | `PrePruningCriterion`: one per-candidate test (`ThresholdPrePruning`, `EncodingLengthRestriction`, ...) that a search can use as a filter, as a stopping trigger, or that the covering loop can use as its stop condition. |
| `rule` | `Rule`: a conjunction of Boolean literals, with optional condition order, several output formats and constraint-aware consistency checks. No dependencies beyond numpy. |

## Data representation

A `DataSpec` (pure schema: feature names, typed attributes, constraints,
missing-value policy) defines the Boolean feature space; a
`DataRepresentation` binds actual data to one. There are **four base
representations**, all implementing the same `coverage(rule)` /
`features_of(row)` interface, so every native rule learner
(`pyrulearn.learners.seco`'s `CN2`, `PFoil`, `PFossil`, `AQR` and `Pypper`,
`pyrulearn.learners.pylord.PyLORD`, ...) runs on any of them unchanged and
produces byte-identical rules. Two are bitmaps, one bit per example and
feature, packed in either direction; two keep, per feature, the list of
examples where it is true:

- `VerticalBitmapRepresentation` -- the default representation (first in
  the preference order below): one bit-vector per *feature*, over the
  examples, in 64-bit words (the vertical bitmap layout of the MAFIA and
  SPAM frequent-pattern miners). Covering a rule ANDs its features'
  bit-vectors, counting is a popcount against a class's bit-vector, and a
  search's batch counting ANDs the rule's cover into every open feature's
  words at once, over just the words the cover still touches.
- `HorizontalBitmapRepresentation` -- the same bit matrix packed per
  *example* (`numpy.packbits` over each row): a rule's coverage checks
  every example's row against the rule's features. The simplest to
  construct by hand, and `rep.X` is its own storage (the others build `X`
  on demand). Until 0.3.4 it was called `BooleanDataRepresentation`.
- `SparseDataRepresentation` -- `scipy` CSR/CSC; a rule's coverage is
  the intersection of its features' column index-sets (Eclat's vertical
  tid-lists), rarest feature first. This is the N-list *without* the
  prefix tree.
- `NListRepresentation` -- the FP-tree / N-list vertical index Huynh,
  Fürnkranz & Beck's LORD builds. Each row's true-feature set is inserted
  (most-frequent-first) into a prefix trie with shared prefixes, and the
  tree is stored *instead of* the table: per node, the bitmask of the
  features on its path and the range of rows passing through it (the rows
  are kept in tree order, so that range is contiguous); per row, its place
  in that order and the node it ends at. A rule's coverage is a vectorized
  `uint64` word-AND over one item's N-list plus the matching nodes' row
  ranges; `features_of` and `X` are decoded from the tree.
  `PrePostNListRepresentation` builds the same tree but keeps the nodes'
  pre/post visit codes and parents instead of the path bitmasks, as the
  N-lists of PrePost: its search tests ancestors by binary search, and
  `features_of` walks up the tree. It holds less memory (on `adult`, 2.4
  times less with negation), and its fits are a little slower.

**Which one to use.** Every learner learns the same rules on all of
them; only time and memory differ (`demos/representations_report.md`,
which measures both on real data, over density, and over the number of
features):

- *The default:* `VerticalBitmapRepresentation`. On real data the
  separate-and-conquer learners and PyLORD fit fastest on it (`adult`,
  32,000 rows, with negation features: Pypper 2.0 s against 6.2 s on
  N-lists, 17.9 s on Sparse and 33.4 s on the horizontal bitmap; CN2
  121 s against 155, 198 and 226 s; PyLORD finishes 32,000 rows in
  505 s only on the vertical bitmap, and needs more than 600 s on the
  others), and ENDER is close to the fastest (23.0 s against 20.6 s on
  N-lists). It holds one bit per cell, like the horizontal bitmap (1.1 MiB
  for `adult`, against 38-42 MiB for N-lists and Sparse), and a fit
  allocates little on top: a CN2 fit needs 5.5 MiB in total, about as
  much as on the horizontal bitmap (6.2 MiB), against 72-99 MiB on
  N-lists and Sparse.
- *For sparse data* -- many features, few of them true per example
  (one-hot encodings of many-valued attributes, text), built without
  negation features: `SparseDataRepresentation`. About 10 bytes per true
  cell against 1/8 byte per cell for the bitmaps, so it is the smaller one
  below a density of about 1/80 (1,000 examples, 20,000 features, about
  18 true per example: 257 KiB against 2.4 MiB for the bitmaps and
  38.7 MiB for N-lists). The separate-and-conquer learners are about as
  fast on it as on the vertical bitmap there (CN2 3.3 s against 3.7 s, and
  14.3 s on the horizontal bitmap); ENDER is faster on the list-based
  representations on such data (1.5 s on Sparse, 1.6 s on N-lists, 7.8 s
  on the vertical bitmap).
- Negation features make any data dense (every test or its negation
  holds), so with them, the vertical bitmap.

`pyrulearn.data.REPRESENTATION_PREFERENCE_ORDER` ranks them --
`VerticalBitmapRepresentation`, `NListRepresentation`,
`PrePostNListRepresentation`, `HorizontalBitmapRepresentation`,
`SparseDataRepresentation` -- and its first entry
(`default_representation()`) is what `read_arff`/`read_csv`/`encode`
and `run_cv` build when no `representation=` is given; reassigning
`pyrulearn.data.representation.REPRESENTATION_PREFERENCE_ORDER` changes
that library-wide. A learner that needs one representation's own storage
declares it (`NATIVE_REPRESENTATIONS`) and converts anything else on its
own (`RuleLearner.ensure_representation`, the order breaking ties):
`DenseCPAR`/`DenseENDER` (bounded by `max_auto_convert_cells`), `CARMiner`
(to N-lists), and every external learner (`ExternalRuleLearner`: the
tool takes a dense matrix, converted without a bound) -- so every learner
accepts every representation.

Coverage dispatches on the `DataRepresentation` subclass:
`Rule.covers_data` / `Rule.covers_data_packed` both forward to
`data.coverage(rule)`, so a `Rule` needs to know nothing about how the data
is stored.

Build any of the others from a horizontal bitmap with `.from_boolean(bitmap)` (or `from_xy` /
`from_dataframe`); see `demos/representations.py`, which also runs
both feature encodings (with and without negation features). Any
`DataRepresentation` can switch between those encodings in place:
`rep.without_negations()` drops every `not f` / `x != v` feature (a
column slice; regenerates the constraints; numeric attributes keep both
`x >= t` and `x < t`), and `rep.with_negations()` adds the missing ones back
(synthesized as complements -- exact for data with no missing-value
routing). `DataSpec.without_negations()` / `.with_negations()` are the
schema-only versions. `rep.select_rows(mask)` gives a same-type
representation over a row subset (feature space unchanged, so learned
rules still apply to the full data) -- rebuilt from scratch, used by
`OrderedOneVsRest` to train each stage on the not-yet-peeled classes.

### Row weights

Every representation can carry non-negative row weights: how much each
row counts, e.g. how often it was observed. Pass them to the constructor
(`HorizontalBitmapRepresentation(spec, X, y, weights=w)`) or use
`rep.with_weights(w)`, a cheap copy that shares the storage (so an
N-list isn't rebuilt). `select_rows`, `relabel`, the negation toggles and
`from_boolean` keep them.

With weights, every count becomes a sum of weights: the searches'
coverage counts, `RuleStats.from_rule`, Pypper's description length,
`evaluate`, and so the training stats rules store and print
(`% (12.5/3)`). Integer weights give exactly the results of duplicating
rows. Without weights, all counts stay integers as before.

The heuristics are plain arithmetic on tp/fp/fn/tn and work unchanged
with weighted counts. Only the meaning of a few shifts: the pseudo-counts
of `Laplace`/`MEstimate` are in weight units, and the significance tests
(`LikelihoodRatio`, `ChiSquare`) and description-length criteria assume
whole-example counts, so with weights they are approximations -- as
usual in weighted rule learning (CN2-SD, Slipper).

Not yet weighted: mined rule pools (`CARMiner` counts rows) and the
coverage-space analyses.

### Typed attributes

Beyond plain Boolean features, a `DataSpec` can know the **typed
attributes** its features derive from, via `DataSpecBuilder`:

```python
from pyrulearn import DataSpecBuilder

b = DataSpecBuilder()
b.add_boolean("smoker")
b.add_binary("sex", ["male", "female"])
b.add_nominal("color", ["red", "green", "blue"])
b.add_numeric("age", [20, 30, 40])
b.add_ordinal("size", ["small", "medium", "large"])
b.add_set("tags", ["urgent", "billing", "bug"])
b.add_hierarchical("region", {"Europe": {"France": {"Paris": {}}}, "Asia": {}})
b.add_relational("income_gt_age_scaled", ["income", "age"], expression="income > age * 1000")
ds = b.build()
```

Nominal/numeric/ordinal/set/hierarchical attributes each generate
derived Boolean features per value/threshold/node; relational attributes register
a single feature derived from several source attributes (the value
itself is still supplied externally, via a `HorizontalBitmapRepresentation`).
The
constraints implied among derived features (mutual exclusion, threshold
chains, hierarchy sibling-exclusion + upward implication) are tracked
automatically, which is what lets a `Rule` ask whether it's internally
consistent (`Rule.is_consistent`) or what it implies beyond its explicit
conditions (`Rule.implied_conditions`).

Three of these types differ in how many values they test and whether
the list of values is complete, which decides what negation features
they need:

- **Boolean** (`add_boolean("smoker")`) -- one tested value, a presence
  flag: the feature `smoker`, plus `not smoker` when negation is on. In
  data-mining terms an *asymmetric* binary attribute (an item in a
  basket), which is why it stays a single feature by default.
- **Binary** (`add_binary("sex", ["male", "female"])`) -- a *closed* set
  of exactly two values: always both features `sex=male` and
  `sex=female`, each the other's negation, and never `!=` columns. A
  value outside the two counts as missing (both features False). A
  *symmetric* binary attribute.
- **Nominal** (`add_nominal`) -- a value list that may be incomplete: with
  negation on it gets `color!=v` columns even for two values, since an
  unknown value has to satisfy every `color!=v` (`x!=a` is not `x=b`
  there).

`build_dataspec`/`read_csv`/`read_arff` infer an attribute with exactly
two values (declared in an ARFF header, else observed in the data) as
binary. Importers that infer a `DataSpec` from a rule set (Weka, LORD,
wittgenstein) always use nominal: rules need not mention every value.

The two ordered types differ in what they test:

- **Numeric** (`add_numeric("age", [20, 30, 40])`) -- a continuous
  quantity, tested at chosen thresholds: `age>=20` and its complement
  `age<20`, and so on for every threshold. Both directions always,
  whatever the negation setting: one direction alone would let a rule
  bound an attribute from one side only. (Until 0.3.4, `age<20` existed
  only with negation on.)
- **Ordinal** (`add_ordinal("size", ["small", "medium", "large"])`) --
  a few ordered values, or a small set of numbers (a count, a rating):
  `size=v` for every value, `size<=v` and its complement `size>v` for
  every value but the largest, and with negation on `size!=v` -- the
  equality and inequality tests that thresholds can't express (`x!=v`
  would be a disjunction). Values compare by their position in the
  given order, or numerically if they are all numbers (then an unseen
  value still satisfies the right `<=`/`>` tests); one `OrdinalGroup`
  constraint propagates all of them, so `size=medium` implies
  `size<=medium` and `size>small`. `read_csv` (and `run_cv`) infer a
  numeric column with three to `max_ordinal_values` distinct values
  (default 10, independent of the discretizer) as ordinal (two make a
  binary attribute); a column declared numeric in an ARFF header stays
  numeric.

`add_numeric`'s `ge_thresholds=` generates ``>=``/``<`` features, as above; a
separate `le_thresholds=` argument (default: none, a complete no-op)
generates ``<=``/``>`` features instead -- a second, independent
`NumericGroup`, not a reinterpretation of the first. The two aren't
the same family in disguise: ``x<=t`` is the exact logical complement of
``x>t``, *not* of ``x>=t`` (they disagree exactly at ``x==t``), so
treating a `<=` condition as `NOT(x>=t)` would get that boundary wrong.
Both directions binarize straight from the same raw numeric column with
no extra step either way, and giving the same value to both arguments
(`add_numeric("age", [30], le_thresholds=[30])`) is how to represent
`age==30` as a conjunction of the two resulting literals. The two
families aren't cross-checked against each other, though (deliberately
not built): nothing stops a rule from combining `age>=30` and `age<=25`
into an impossible-to-satisfy conjunction without `Rule.is_consistent`
flagging it -- such a rule simply never covers any real example, which
is a sufficient outcome for rules read in from an external source (see
`pyrulearn.interfaces.weka.JRipImporter`, below, for exactly this
use case).

### Merging DataSpecs

`merge_dataspecs(a, b)` (in `pyrulearn.data`) merges two `DataSpec`s
describing the same underlying attributes -- e.g. two different
discretizations of the same numeric attribute -- into one
`DataSpecBuilder` with the union of nominal categories, numeric
thresholds, and set values for attributes present in both; attributes
present in only one side are carried over as-is. It's deliberately
strict about anything that isn't a straightforward union: a type
mismatch, a different `Hierarchy`, or (for relational attributes)
different sources/expression all raise rather than guessing -- reconcile
such conflicts by hand before merging. It returns a builder (spec only
-- `DataSpec` never carries data to begin with); existing `Rule`s built
against `a`/`b` don't automatically carry over to the merged feature
space -- see `remap` below for that.

#### Rebasing rules onto a different DataSpec

`Rule.remap(new_dataspec)` rebuilds a rule's conditions against a
different, compatible `DataSpec`, translating each condition's feature
index **by name** (`old_idx -> self.dataspec.feature_name(old_idx) ->
new_dataspec.feature_index(name)`) -- the same name-based identity
`merge_dataspecs` already relies on. It's a pure structural translation
(no data touched, nothing recomputed) and raises rather than silently
dropping a condition if a name has no match in `new_dataspec`.
`RuleModel.remap(new_dataspec)` applies it to every rule and carries the
`default_prediction` policy (and `.provenance`) over unchanged, and
keeps every rule's stats (the default rule's too): a rebased rule covers
exactly the same rows. It returns a new instance of the same concrete
class.

This is the fix for reading in several rule-based models over "the same"
dataset when each was imported against its own per-model `DataSpec`
(different attribute/threshold choices, different feature order): build
the shared target with `merge_dataspecs`, `remap` each model onto it,
then binarize the real dataset **once** against that shared `DataSpec`
and score/compare/combine all the remapped models against it --

```python
from pyrulearn.data import HorizontalBitmapRepresentation

shared = merge_dataspecs(model1.rules[0].dataspec, model2.rules[0].dataspec).build()
model1 = model1.remap(shared)
model2 = model2.remap(shared)
shared_rep = HorizontalBitmapRepresentation(shared, binarize(shared, df))
```

Only `shared` ever needs a full Boolean data matrix built for it; the
per-model DataSpecs used during import don't (`remap` needs nothing but
their feature *names*), which keeps memory to one dataset's worth
regardless of how many models get combined. The rules keep their
training stats through `remap`; `model.evaluate(shared_rep)` measures a
model on `shared_rep` without storing anything, and `reset_stats` replaces
a rule's training stats deliberately (see *Statistics*).

### Reading ARFF / CSV data

```python
from pyrulearn.data.io import read_arff, read_csv

# infer a DataSpec from the file's own column types, discretizing numeric
# columns against `target` with a decision tree (TreeDiscretizer(): at most
# max_intervals - 1 = 7 thresholds -- a full 3-level binary tree)
rep = read_arff("weather.arff", target="play")
from pyrulearn.data import BoundaryCorrection, EqualFrequencyDiscretizer, TreeDiscretizer
rep = read_csv("data.csv", target="label", discretizer=TreeDiscretizer(max_intervals=4))
rep = read_csv("data.csv", target="label",       # equal-frequency cuts, each moved
               discretizer=BoundaryCorrection(EqualFrequencyDiscretizer(8)))  # to where the class changes
# rep.spec is the inferred DataSpec; rep is a VerticalBitmapRepresentation
# (representation=, default default_representation() -- pass
# HorizontalBitmapRepresentation for the dense matrix instead)

# or encode against a DataSpec you already have (thresholds and all)
from pyrulearn.data import HorizontalBitmapRepresentation
rep = read_csv("data.csv", dataspec=my_dataspec, target="label",
               representation=HorizontalBitmapRepresentation)
# rep.spec is my_dataspec; rep.X / rep.y hold the dense data
```

Numeric columns without pre-given thresholds are discretized by a
`Discretizer` object (`pyrulearn.data.discretization`), passed as
`discretizer=` to `read_csv`, `read_arff`, `build_dataspec` and `run_cv`:

- `TreeDiscretizer(max_intervals=8)` (the default) -- the split points of
  a single-feature decision tree against the class (scikit-learn's).
  `max_intervals` is a ceiling: a column with little signal gets fewer
  thresholds. `max_intervals=None` grows the full tree: a threshold at
  every point where the class changes.
- `FusinterDiscretizer(alpha=0.975, lam=1.0, max_intervals=None)` --
  FUSINTER (Zighed, Rabaseda & Rakotomalala, 1998), supervised and
  bottom-up, native: starts from the class-homogeneous intervals and
  merges adjacent ones while a quadratic entropy with a penalty on small
  intervals improves. It decides the number of intervals itself;
  `max_intervals` caps it, by going on merging the least harmful pair.
- `EqualFrequencyDiscretizer(max_intervals=8)` -- unsupervised
  equal-frequency ("quantile") binning, native: each threshold in the
  gap between two observed values nearest to an equal-frequency
  position.
- `EqualWidthDiscretizer(max_intervals=8)` -- unsupervised equal-width
  binning, native; empty intervals merge with their neighbour.
- `BoundaryCorrection(discretizer)` -- moves each of another
  discretizer's thresholds to the nearest *boundary point* (counted in
  examples): a gap between two adjacent values where the class changes.
  Thresholds that land on the same boundary merge, so a column may get
  a few fewer thresholds.
  A threshold between two values whose examples all share one class is
  never the best place for one, for any heuristic that rewards covered
  positives and penalizes covered negatives (Fayyad & Irani, 1993).
- `PerClass(discretizer)` -- runs a supervised discretizer once per
  class, on that class against all others, and keeps the union of the
  thresholds, so every class gets cuts placed for it, e.g.
  `PerClass(TreeDiscretizer(3))` (up to two thresholds per class).

A supervised discretizer (the tree, FUSINTER, or one wrapped in `BoundaryCorrection` or `PerClass`)
needs a `target` column. `validate_dataspec(dataspec, df)` checks an existing `DataSpec`
against a file's header without reading data (missing attributes, type
mismatches, unknown nominal categories); `strict=True` (the default on
`read_arff`/`read_csv`) raises on anything `validate_dataspec` reports
rather than binarizing against a `DataSpec` that doesn't actually match.
`read_arff` reads ARFF's missing marker `?` as a missing value (not as a
category) and takes each nominal attribute's values from the header's
declared list, so an attribute declared with two values is binary even
if only one of them occurs.

Not yet handled: set-valued/hierarchical attributes (ARFF/CSV headers
can't declare them, so they're never inferred). Relational attributes
can't be inferred either, but a frame can declare them:
`df.attrs["relational"] = {column: {"sources": [...], "expression":
"..."}}` makes each such precomputed Boolean column a RELATIONAL
attribute in `build_dataspec` (with its negation), and `encode` reads
its value from the column -- `pyrulearn.datasets.krk_illegal(...,
relations=True)` is an example.

### Missing values

A missing raw value (`None`/NaN, or an attribute's declared
`missing_values`, e.g. `("?",)` for the common ARFF/UCI convention) is
handled per `pyrulearn.data.attributes.MissingStrategy`, resolved (explicit
`binarize`/`read_arff`/`read_csv` argument > the `DataSpec`'s own
`missing_strategy` > `DataSpec.DEFAULT_MISSING_STRATEGY`) the same way
`Rule.to_string`'s `fmt` resolves:

- **`NEVER_COVERS`** (the default) -- every feature derived from the
  missing attribute is False for that example, so it never satisfies
  *any* literal conditioning on it, positive or negated. This is a
  closed-world "we don't know" reading, not "the negation holds" -- the
  cheaper alternative to genuinely tracking a third "unknown" state
  through `Rule`'s coverage checks, which isn't implemented.
- **`MAJORITY`** -- impute the raw value before evaluating: the column
  median for a NUMERIC attribute, the lower median value for an ORDINAL
  one (always one of its values), the most frequent value (mode)
  otherwise.
- **`RANDOM`** -- impute with another, uniformly randomly chosen
  non-missing example's value from the same column (`random_state=`
  seeds this).
- **`SEPARATE`** -- route into a dedicated feature, treating "missing"
  as its own value alongside the attribute's declared domain. Requires
  the attribute to have been built with `missing_name=`
  (`add_nominal`/`add_numeric`/`add_ordinal`) -- raises if none was
  declared. For an ORDINAL attribute it is one more position, outside
  the order. For a
  NOMINAL attribute the dedicated feature rides in the same exhaustive
  `ExactlyOne` group as the declared domain (missing is just one more
  category); for a NUMERIC attribute it's a standalone feature outside
  the `ThresholdChain`, deliberately with no constraint linking the two
  (a numeric value can be simultaneously "missing" and fail every `>=`
  test).

```python
b = DataSpecBuilder()
b.add_nominal("color", ["red", "green", "blue"], missing_name="<missing>", missing_values=("?",))
b.add_numeric("age", [20, 30, 40], missing_name="<missing>")
ds = b.build(missing_strategy=MissingStrategy.SEPARATE)
```

`add_boolean`/`add_binary`/`add_set`/`add_hierarchical` accept `missing_values=` too
(for sentinel recognition) but not `missing_name=` -- `SEPARATE` isn't
supported for those attribute types.

## Rule models

A `Rule` is a conjunction of Boolean literals over a fixed feature space
(`Literal(feature)` -- a literal is just a feature index; there are no
negative literals). "Feature absent" is expressed by conditioning on a
separate *negation feature* (`not f`, `color!=red`), paired with its
positive counterpart by a `MutuallyExclusive` / `NominalGroup` /
`OrdinalGroup` constraint -- `DataSpecBuilder` generates these by
default (`negation=False` opts out). A binary attribute (`sex=male` /
`sex=female`, see *Typed attributes*) needs no extra columns: its two
value features are each other's negation; neither does a numeric one,
which always has `age>=30` and its complement `age<30`. A rule reads back out as one tuple
of feature indices (`pos`, the features that must be True) plus a single
integer bitmask, so coverage of one example is a genuine O(1)-ish subset
check:

```python
covers = (x_bits & mask) == mask
```

`Rule.from_pos_neg(pos=..., neg=...)` is still the convenient
constructor: each `neg` index is translated to its negation feature via
`DataSpec.negation_of`, so a negation-enabled `dataspec` is required
when `neg` is non-empty.

A rule can optionally carry the order its conditions were added/tested
in (`ordered=True`), and renders via `to_string(fmt=...)` as logic
notation, a Prolog-style clause, a positional pattern string, or a bare
condition list. `fmt` is optional: omitted, `to_string` (and `__repr__`,
so this is also what you see printed) falls back to that rule's own
`default_fmt` if set at construction (`Rule(..., default_fmt="logic")`),
else the class-wide `Rule.DEFAULT_FORMAT` (`"prolog"`) -- an explicit
`fmt=` always overrides both. Plain printing shows every condition;
`pretty=True` (on a rule or a whole model) also compresses the conditions
on one attribute: two or more `≠` on a nominal attribute become one value
set, in the shorter of its two forms -- `sibsp ∉ {4, 5, 8}`, or `color ∈
{green, white}` when fewer values remain than are excluded
(`\+ member(V, [...])`/`member(V, [...])` in Prolog) -- and two or more
thresholds on a numeric attribute one interval, `20 <= age < 40`. A
single condition stays as it is. Only the display changes: the rule
keeps its conditions and `length()` counts each one. `reorder`/`generalize` carry a rule's
`default_fmt` over to the result. A bare `Rule` carries no weight or
free-form metadata; measured per-rule statistics live on `SingleRule.stats()`,
and what built a rule on `SingleRule.provenance`.

`WeightedRule` is the `Rule` subclass that adds a weight: one declarative
number that is *part of the model* (e.g. a ProbLog-style probability), as
opposed to statistics measured against a dataset. Weights are reserved for
values that are fitted or set explicitly and can't be recomputed from a
rule's stats. So far the one model that predicts from them is
`LinearRuleModel` (below).

`RuleModel` (in `pyrulearn.models`) is the shared abstract base for
collections of `Rule`s that make predictions together. The hierarchy is
organised by **resolution** -- how a prediction is decided when several
rules apply:

- **`SingleRule`** -- one rule, wrapped so it carries its own `stats`/
  `provenance` and predicts on its own; the hierarchy's base case, and
  what every container actually stores (`.rules` is a list of
  `SingleRule`s, never bare `Rule`s).
- **`RuleSet`** (abstract) -- *unordered*; covering rules are reconciled
  by `self.resolution` (a `Combine` over a `combiner`, or an `Exclusive`
  disjointness assumption). Concrete subclasses:
  - `FlatRuleSet` -- a plain bag of mixed-head rules plus one `combiner`
    (the generic, directly-instantiable case). `PooledRuleSet` is its
    memory-compact variant for huge mined rule pools.
  - `ConceptModel` -- rules that all share one head (one **concept**);
    `predict` returns that label where covered, `default_prediction`
    otherwise.
  - `ConceptSet` -- one `ConceptModel` per label, `combiner`-resolved
    where several concepts cover the same row.
  - `LinearRuleModel` -- a linear model over rules (RuleFit): each rule's
    signed weight counts for its head, every class competes on every row
    (a class no covering rule predicts scores 0), an empty-body rule is a
    class's intercept, and the highest sum wins. `scores(data)` gives the
    sums. `learners.rulefit.RuleFit` fits one.
  - `DisjointRuleSet` -- rules assumed pairwise mutually exclusive --
    e.g. a decision tree's leaves, whose path conditions partition the
    feature space by construction, so there's never actually a tie to
    break. `interfaces.sklearn.from_sklearn_tree` returns one
    directly.
- **`RuleList`** (abstract) -- *ordered*; the first matching rule wins
  (RIPPER/CN2-style sequential covering). This rule-level order is
  unrelated to `Rule.ordered`, which is about condition order *within*
  one rule. Concrete subclasses:
  - `DecisionList` -- a linear list of rules; the first that matches
    decides.
  - `ConceptCascade` -- ordered `ConceptModel`s, "peeling" one-vs-rest:
    the first concept that fires decides, and concept k only sees the
    rows concepts 1..k-1 didn't take.
- **`CompositeModel`** (abstract) -- members that are themselves
  `RuleModel`s, combined at prediction time. Concrete subclasses:
  - `EnsembleModel` -- a flat, optionally weighted vote over independent
    members (bagging / boosting-style ensembles, e.g. one member per
    random-forest tree), or, with `combiner=`, the members' deciding rules
    resolved by any rule combiner. With `abstain` (the default), a member
    votes only where its rules cover an example. `staged(n)` keeps the
    first `n` members.
  - `PairwiseModel` -- round robin: one binary member per label pair,
    each voting for one of its two labels.
  - `DeepModel(input_spec, labels, layers, first="and")` -- a formula in
    negation normal form: a fully connected network of alternating AND and
    OR layers with 0/1 edge weights. The inputs are the features, the
    output layer has one node per label, `layers` are the sizes of the
    layers in between, like a neural network's, any number of them;
    `first` is the type of the layer above the inputs (`"and"` or `"or"`),
    the others alternate, so the output's type follows (`[32, 16, 10, 5,
    2]` from AND ends in an OR output, `[6, 6]` in an AND output; by
    default one AND layer of `ceil(sqrt(n))` nodes for `n` features, a DNF,
    or from OR a CNF); the weights start at 0, or come from `weights=`.
    The weights are the model and predict by matrix products; its rules
    are a view of them (`to_concepts()`): every OR node is a concept (a
    `ConceptModel` with one rule per incoming edge -- the AND node on it,
    or a single input literal), so is an AND output (one rule, its
    inputs), a hidden AND node is a rule body, and a concept is a feature
    for the layers above. By default the
    view removes the pass-through nodes: a concept with a single rule is
    inlined, a rule whose body is a single concept replaced by that
    concept's rules.

### Printing a model

`to_string(fmt=None, ascii=False, show_stats=True)` renders a whole model,
applying one resolved format (explicit `fmt=`, else `Rule.DEFAULT_FORMAT`)
to *every* rule, regardless of any individual rule's own `default_fmt`.
This is what keeps a printed model internally consistent even when its
rules weren't all built with the same `default_fmt`. Structure differs by
family:

- `RuleSet` (`FlatRuleSet`/`ConceptModel`/`DisjointRuleSet`/`ConceptSet`)
  groups rules by target label, each section headed by `% class:
  <target>`. For `fmt="logic"`, a label's rules collapse into one DNF
  expression -- each rule's conjunction parenthesized on its own line,
  `∨`-prefixed after the first, e.g. `(age_gt_30 ∧ smoker)` /
  `∨ (high_bp)` / `→ high_risk`. `fmt="prolog"` doesn't need that
  treatment: consecutive clauses sharing a head already read as a
  disjunction, so each rule just stays its own clause under the header.
  `"pattern"`/`"conditions"` have no natural merged form, so rules are
  listed under the label header unmerged. The exception is a rule set
  whose own combiner is a `ListCombiner` (first covering rule wins) and whose
  rules have more than one head: list order decides its predictions, so
  it prints like a `DecisionList`, in that order.
- `RuleList` (`DecisionList`/`ConceptCascade`) doesn't group by label --
  order, not shared target, is what decision-list semantics depend on.
  `fmt="logic"` renders as if/elif/else pseudocode (order = priority,
  matching `predict`'s own first-match-wins semantics exactly); other
  formats list rules sequentially in list order.
- `default_rule` (when the policy has a constant label) gets its own
  trailing section: a `% default` header for `RuleSet`/non-logic
  `RuleList` output, or the pseudocode's own `else` line for `RuleList` +
  `fmt="logic"`.

The `"conditions"` and `"pattern"` formats render only a rule's body. Inside
a model the class is still shown: a decision list prefixes each rule with
it (`z: ¬f0, ¬f1`, `z: 0 1 0 1`), and every model's default section names
it (`% default: x`); a rule set's `% class:` headers already cover its
rules.

`pretty=True` prints each rule's conditions on separate lines, indented
under the head, with its coverage comment on a line of its own above the
head (Prolog format; the other formats are unaffected so far):

```prolog
% (57/21)
good(X) :-
    checking_status(X, no_checking),
    duration(X, le_24).
```

`print(model)` shows the default rendering (`to_string()`); a model's
`repr` is a short identification, e.g. `FlatRuleSet(3 rules)` or
`SingleRule(2 conditions)`.

A `WeightedRule` prints its weight as part of the rule: in front of it in
the default Prolog format (`0.8::head :- body`), and appended in the other
formats (`[0.8]` for `"logic"`, `% 0.8` otherwise). `weight_format=`, a
Python format spec, formats the weights -- `to_string(weight_format="6.2f")`
gives every weight the same width, so the heads line up even with signs and
differing magnitudes:

```prolog
% conflict resolution: sum of rule weights per class, highest wins
% class: good
 -3.02::good(X) :- true.  % (101/299)
  5.00::good(X) :- f0(X), f1(X).  % (97/2)
```

Every rule that carries
training stats is suffixed with a coverage comment read from those stats --
exactly the numbers the model holds and its combiner scores from, never
recomputed against other data (`show_stats=False` prints the bare rules):

- Ordinarily `% (tp/fp)` -- covered training rows that are, or aren't,
  actually this rule's own target (the classic C4.5/RIPPER rule-quality
  notation; 0 `fp` reads as a perfect rule).
- For a model whose own `combiner` is genuinely a `DistributionCombiner`
  (`MicroVoteCombiner`/`MacroVoteCombiner`/`MicroMaxCombiner`/`MacroMaxCombiner`) *and* has more
  than two classes, the full per-class breakdown instead -- `% [n0, n1, ...]`
  -- everything that combiner's own `resolve()` reads, nothing it doesn't.
  A one-line `% classes: [...]` legend, printed once above the rest of the
  output (never repeated per rule), gives that vector's order. With exactly
  two classes this vector would just be `(tp/fp)` reordered, so it's
  suppressed in favor of the plain form.

Both are choices, not hard rules -- `to_string`'s `show_distribution`/
`show_classes` (`None` by default) force either one independently, for any
model, any class count, any combiner: the per-class counts are always in a
rule's stats, whether or not a given model's own resolution actually
consults them. `show_distribution=True`/`False` forces the vector
on or off outright (e.g. showing it for a plain `HeuristicMaxCombiner()` model, or
suppressing it for a genuine `DistributionCombiner`); `show_classes=True`/
`False` independently forces the legend on or off, regardless of whether any
rule ends up showing a vector at all -- useful for naming a model's relevant
classes as a label on its own.

`DeepModel.to_string` prints its hidden concepts, each headed
`% concept: <label>`, then its output concepts as a `ConceptSet` prints them;
by default without the pass-through nodes (`simplify=False` prints every
node).

`EnsembleModel`/`PairwiseModel` also have `to_string`. Both render every member/pair
in turn plus a top-level `% classes: [...]` naming the model's own `labels`
(from its own declared structure, not `data` -- see `_container_legend`).
`EnsembleModel` headers each member `% member <k>`, with `(weight: ...)`
appended where `member_weights` is set -- exactly the number `predict`'s
plurality vote weighs that member's verdict by, so the one thing beyond each
member's own rules a reader needs to redo the vote by hand; `show_stats`/
`show_distribution`/`show_classes` pass through unchanged to every member,
since they all cover the same overall multiclass problem. `PairwiseModel`
headers each pair `% pair: a vs b`, with `(member weight: ...)` appended for
`AccuracyWeightedVote` specifically -- `WeightedVote`'s own per-row deciding-
rule weight is its heuristic (named in the model's conflict-resolution line)
on that rule's printed training stats, so nothing extra is needed for that
combiner. Each pair's rules carry stats measured on that
pair's own two classes' rows (where they were fitted), and, unlike
`EnsembleModel`, its own `show_classes` defaults to forced-on (`None` here means "force", not
"auto") -- a sub-model's rules may only ever explicitly predict *one* of its
two classes (the other only ever surfacing as its own `default_prediction`),
so without this a reader may have no way to tell which two classes a given
pair is even about.

### Converting between model types

`pyrulearn.models.convert(model, dst)` converts a model to another type when
a direct converter exists (`can_convert(src, dst)` checks; there is no
transitive closure, so chain converters explicitly). The registered ones:

- `FlatRuleSet` to `DecisionList` -- adopts the insertion order, so
  prediction changes from combiner resolution to first-match.
- `FlatRuleSet` to `ConceptSet` -- groups the rules by head; with an
  order-independent combiner, prediction is unchanged.
- `ConceptSet` to `FlatRuleSet` -- drops the per-concept structure.
- `ConceptCascade` to `DecisionList` -- prediction is preserved.
- `EnsembleModel` to `FlatRuleSet` -- lossy: pools every member's rules under
  one combiner (default `CountVoteCombiner()`), dropping the per-member grouping and any
  member weights.

Any `RuleSet` also has `to_rulelist(key=None, reverse=True)`, which orders
its rules with `pyrulearn.evaluation.sort_rules` (default: descending
Laplace-on-measured-stats -- pass `key=`/`reverse=` for another order) into
a `DecisionList`.

### Conflict resolution

When several rules cover the same row, a `RuleSet` reconciles them with its
`combiner` (a `RuleCombiner` in `pyrulearn.combiners`, or a string shortcut).
Pass a `RuleCombiner` object to the constructor -- like every component,
combiners are objects, not names -- or override it per call via
`predict(data, combiner=...)`.
Omitted, `predict` falls back to the set's own `self.combiner` (settable at
construction, itself defaulting to `HeuristicMaxCombiner()`), so a `RuleSet` built with a
particular strategy keeps it without every `predict` call re-passing it.

A printed model whose rules predict more than one class names its conflict
resolution in its first line -- the combiner's `describe()`, e.g.
`% conflict resolution: max Laplace`, `... micro distribution voting (sum of
covered class counts)`,
or `... first matching rule` for a decision list (`to_string(...,
show_resolution=False)` omits it). Together with each rule's printed
training stats, that is everything needed to work out a prediction by hand.

**Ties** never depend on where a rule sits in the model. When a combiner's
own criterion leaves several classes level, `max` first lets the tied
top-scoring rules vote (the class with the most of them wins); any tie
left after that, for every combiner, goes to the class that is more
frequent in the training data, then to the one that sorts first. This
convention is documented rather than printed. Only `ListCombiner` is order-based,
by definition. A `LinearRuleModel` doesn't use a combiner: its resolution
is the weighted sum, printed as `% conflict resolution: sum of rule weights
per class, highest wins`, and its (rare, exact) ties follow the same
convention.

`RuleCombiner.resolve(rules, covering)` is the one abstract method
(`covering`: non-empty indices into `rules`). `ListCombiner` and
`CountVoteCombiner` stand alone; everything else falls under one of two
intermediate bases, matching two different kinds of per-rule information a
combiner can use:

- `ListCombiner`: rule position, i.e. list order.
- `CountVoteCombiner`: plain, unweighted majority vote.
- `HeuristicCombiner` -- scores a `RuleHeuristic` against each rule's own
  measured stats (`SingleRule.stats()`'s `ConfusionMatrix`, rotated to the
  rule's own target), computed fresh at combine time, not baked into the
  model beforehand. All take a `heuristic=` (default `Laplace()`):
  - `HeuristicMaxCombiner`: the classic ensemble "max rule",
    which picks the single covering rule with the highest heuristic score.
  - `HeuristicVoteCombiner`: majority vote across every covering rule, each
    vote weighted by that rule's heuristic score.
  - `TopKMeanCombiner(heuristic, k=5)`: for each class, the mean score of
    its best `k` covering rules (all, if fewer); the highest mean wins
    (CPAR's prediction). `k=1` is the max rule.
- `DistributionCombiner` -- for rules scored by a full per-class breakdown
  rather than one scalar: each rule's own measured stats
  (`ConfusionMatrix.predicted_as(rule.target)`, the true-label distribution
  among the rows it actually fired on). Two independent axes give four
  concrete combiners. *Which* per-rule numbers get used: `Micro` pools every
  covering rule's raw counts (larger leaves count for more), `Macro`
  normalizes each rule's own counts to proportions first (every rule counts
  equally regardless of leaf size) -- the same micro/macro distinction used
  for multi-class F1 scores. *How* the rules' numbers are combined: `Vote`
  sums them (total support per class), `Max` takes the highest per class
  across covering rules (the classic ensemble "max rule" in its full
  per-class form, distinct from `HeuristicMaxCombiner`, which picks one whole
  rule rather than comparing per class).
  - `MicroVoteCombiner`.
  - `MacroVoteCombiner`. **This is the one that matches
    `sklearn.ensemble.RandomForestClassifier.predict()`'s own mechanism**:
    each tree contributes one class-probability vector (its own leaf counts
    normalized), averaged (equivalently, summed) across trees -- unlike every
    `HeuristicCombiner`/`CountVoteCombiner`, which collapses each rule down
    to a single hard vote/score before combining, discarding how confident a
    leaf actually was.
  - `MicroMaxCombiner`.
  - `MacroMaxCombiner`.

The distribution combiners are the ones to reach for when importing a whole
random forest as one `FlatRuleSet`
(`pyrulearn.interfaces.sklearn.RandomForestImporter`): each tree contributes
one always-firing rule per example, so a k-tree forest always has exactly k
covering rules, and combining across all of them is what actually reproduces
the forest's own prediction (`MacroVoteCombiner` in particular matches
sklearn's own soft-voting mechanism almost exactly).

`HeuristicMaxCombiner`, `HeuristicVoteCombiner` and `DistributionCombiner`
all **require** every covering rule to have measured stats when the covering
rules genuinely disagree in target, and raise `ValueError` otherwise (an
unmeasured rule contributing a silently wrong score is exactly the kind of
quiet wrong answer this stats-based design exists to rule out). Annotate
first: `pyrulearn.models.annotate_rules` (see *Statistics*, below), or a
`fit()`/importer `data=` call, which already do. A *unanimous* covering block
(every covering rule already agreeing on the target) never needs stats at all,
as there is no actual disagreement to resolve. `DistributionCombiner` also
raises if a covering rule has neither measured stats nor `class_counts` --
there is no further fallback.

Whether conflicts can occur at all is a check you can run: `is_disjoint(data)`
is available on any `RuleModel`, not just `DisjointRuleSet`, so a plain
`FlatRuleSet` can be checked *before* deciding whether it's safe to
convert, not only verified after the fact. Disjointness isn't enforced at
construction, the same way `Rule.is_consistent()` is a check you call, not a
built-in guarantee.

### Default predictions

A model isn't necessarily *exhaustive*: some rows may match no rule.
`is_exhaustive(data)` checks that against real data (like `is_disjoint`, on
any `RuleModel`).

Every `RuleModel` has a `default_prediction` (settable at construction or
afterward), the *policy* for rows no rule in `self.rules` decides. It is one
of:

- a bare label -- `predict` returns it;
- `None` -- abstain, `predict` returns `None`;
- a `DefaultPrediction` object -- consulted per uncovered row via
  `predict(rules, data, example_idx)`. This is not vectorized, so different
  rows can genuinely get different fallbacks. `MajorityClass` (the built-in
  one) resolves a fixed majority label at construction from a chosen row
  subset; "predict via whichever rule is most similar to this row" would be
  another.

`filter`/`to_rulelist`/`remap` all carry the policy over unchanged, and so
does `.provenance` (what built this model -- a `RuleLearner`/`RuleImporter`
stamps it; narrowing/rebasing/repackaging doesn't change who built the
rules, so it survives all three).

`default_rule` is a read-only, lazily-materialized `SingleRule` view of that
policy: its `target` is the policy's constant label (a bare label, or a
`DefaultPrediction`'s `constant_target`), or `None` for a `None` policy or a
non-constant strategy. Being a real `SingleRule`, it carries its own
`stats`/`provenance` like any other rule. Reassigning `default_prediction`
discards the materialized rule and its stats.

### Explaining a prediction

`RuleModel.covered_by(data, by=None)` returns, per row, every rule whose
body holds, ordered as an explanation of the prediction: the predicted
label's block first, other labels' blocks after (each ordered by its
strongest rule), rules within a block sorted by `by` (a per-rule key
callable; default: descending Laplace on the rule's own *measured* stats --
raises if a genuine multi-rule block has a rule with none). Uncovered rows
yield `[default_rule]` or `[]`. For a richer heuristic-based ranking
(needing `data=` to score against), use
`pyrulearn.evaluation.sort_rules(rules, by=<a RuleHeuristic>, data=data)`
directly on `model.covered_by(data)`'s output, or as `covered_by`'s own
`by=` argument.

### Statistics

Only rules store measurements. Each rule (`SingleRule`) holds its
**training stats**, `rule.stats()`: a `pyrulearn.evaluation.ModelStats` (a
`ConfusionMatrix` of the rule's own predictions vs the training labels,
plus `n_rules`/`n_conditions`), or `None` if it has none. They are set
where the rule is produced -- a learner's `fit`, an importer's `data=`,
or `pyrulearn.models.annotate_rules(rules, data)`, which wraps a plain
rule list and sets each rule's stats in one call -- so
combiners/`sort_rules`/`covered_by` have real measured stats to score
from.

Those stats are **frozen**: they are part of the model -- combiners score
from them and `to_string` prints them -- so setting them a second time
(`rule.set_stats(other_data)`, `annotate_rules(rules, other_data)`) raises
instead of silently changing what the model predicts. Replace them
deliberately with `rule.reset_stats(data)` (or `annotate_rules(...,
reset=True)`). `annotate_rules(..., copy=True)` annotates fresh copies of
the rules and leaves the given ones untouched -- what the rule
distillers (`CBA`, `IDS`) do with the pool they select from. `remap`,
`filter` and model conversions keep the stats.

Measuring on other data is `model.evaluate(data)`, available on every
model and every rule: it returns a `ModelStats` (the model's own
`predict(data)` vs `data.y`) and stores nothing.

### Rule-evaluation heuristics

`RuleHeuristic` (in `pyrulearn.heuristics`) is a pluggable rule-quality
score, evaluated against `RuleStats` -- the confusion-matrix quartet
`tp`/`fp`/`fn`/`tn` (treating "covers" as "predicts positive"), plus
optional `length` and `parent` -- rather than positional args, so adding
a new stat later doesn't force every heuristic's signature to change.
`length`/`parent` deliberately aren't part of `Rule` itself: they're
heuristic-search context (a bare rule has no inherent notion of its own
refinement history), not coverage information, which is why `RuleStats`
lives here rather than in `pyrulearn.rule`.

```python
class RuleHeuristic(ABC):
    @abstractmethod
    def score(self, stats: RuleStats) -> float: ...   # higher = more preferred
    def score_rule(self, rule, data, positive_class=None) -> float: ...  # convenience
```

Every heuristic follows the same "higher = more preferred" convention,
so any of them can be dropped in as a ranking key without the caller
caring which one is active -- `Rule.order_by_precision`'s greedy
criterion is one hardcoded instance of this general idea. `pyrulearn.
evaluation.sort_rules(rules, by=None, data=None, descending=True)` is
the general-purpose consumer: `by=None` ranks by descending
Laplace-on-measured-stats (the same default `HeuristicMaxCombiner`
uses), a `RuleHeuristic` scores fresh against `data` (unless
`needs_data` is `False`, e.g. `MinimalLength`), or pass a plain callable.
`RuleStats.from_rule(rule, data, positive_class=None, example_mask=None)`
computes `tp`/`fp`/`fn`/`tn` from a rule's actual coverage
(`positive_class` defaulting to `rule.target`) and `length` from
`rule.length()`. A gain-style heuristic needing a *parent* rule's stats
too (see `GainHeuristic`, below) just calls this a second time on the
parent rule -- there's no dedicated parameter for it here.

Built in (all in `pyrulearn.heuristics`). The four confusion-matrix quadrant
counts ("covered/uncovered" x "positives/negatives"); the sign follows whether
the quadrant is a correct outcome (`CoveredPositives`/`UncoveredNegatives`,
scored directly) or an error (`CoveredNegatives`/`UncoveredPositives`,
negated), so more of it is never accidentally "better":

- `CoveredPositives` -- tp alone; ignores fp entirely.
- `CoveredNegatives` -- -fp alone; ignores tp entirely.
- `UncoveredPositives` -- -fn.
- `UncoveredNegatives` -- tn.

Precision-like heuristics:

- `Precision` -- tp/(tp+fp); also called Confidence in association-rule-mining
  terminology.
- `Recall` -- tp/n_pos, a.k.a. sensitivity/TPR/hit rate. `CoveredPositives`'s
  own rate-normalized twin (the same relationship `Support` has to
  `Coverage`); not to be confused with raw `CoveredPositives`, a count, not a
  rate.
- `FBeta(beta=1.0)` -- the weighted harmonic mean of `Precision` and `Recall`;
  `beta=1`, the default, is the standard F1 score, weighing them equally,
  `beta<1` favors precision, `beta>1` favors recall. Rewritten as
  `(1+beta**2)*tp / (tp+fp+beta**2*n_pos)`, its isometrics turn out to be a
  pencil pivoting at `(-beta**2*n_pos, 0)`, the same fp-axis sub-family as
  `GHeuristic`, just tied to the dataset's actual class balance instead of a
  free constant.
- `Laplace` -- Laplace-smoothed precision.
- `MEstimate(m)` -- generalizes `Precision` at m=0, pulled toward the prior
  positive rate as m grows. Isometrics: a pencil pivoting at
  `(-m*(1-p0), -m*p0)`.
- `GeneralizedMEstimate(m, cost)` -- replaces `MEstimate`'s prior `p0` with a
  free `cost` parameter (`MEstimate(m)` is exactly
  `GeneralizedMEstimate(m, cost=p0)`); freeing `cost` lets the pivot land
  anywhere on the line `fp+tp=-m`, not just at the single point the dataset's
  own prior would put it.
- `GHeuristic(g)` -- Gamberger & Lavrač's expert-guided subgroup discovery "g
  heuristic", used in CN2-SD: tp/(fp+g), where `g` is meant to be a positive
  constant; larger `g` tolerates more covered negatives, favoring more
  general rules.

Accuracy-, coverage- and cost-style heuristics:

- `WRAcc` -- weighted relative accuracy.
- `YoudenJ` -- Youden's J statistic (Youden, 1950), tpr - fpr =
  tp/n_pos - fp/n_neg; an alternative to `WRAcc` that trades off true/false
  positive *rates* rather than raw coverage-weighted counts. Also known as
  informedness, the vertical distance above the random-guess diagonal in ROC
  space.
- `Accuracy`.
- `CoverageDifference` -- tp - fp; ranks identically to `Accuracy` (same
  isometrics) but skips the normalizing division, so prefer it when only
  relative order matters and `Accuracy` when the actual percentage does. Also
  exactly `LinearCost(cost_ratio=1.0)`, kept as its own name for
  discoverability.
- `Support` -- coverage rate, ignoring purity.
- `Coverage` -- tp + fp; the unnormalized version of `Support`, the same
  relationship `CoverageDifference` has to `Accuracy`.
- `LinearCost(cost_ratio=1.0)` -- tp - cost_ratio*fp.
- `LinearCostRates(cost_ratio=1.0)` -- tpr - cost_ratio*fpr; the rate-space
  counterpart to `LinearCost`, the same way `YoudenJ` is to
  `CoverageDifference` (`YoudenJ` is exactly `LinearCostRates(cost_ratio=1.0)`).

Statistical and information-theoretic heuristics:

- `Correlation` -- FOSSIL's four-field/Matthews correlation coefficient
  between "rule covers this example" and "example is positive",
  `(tp*tn - fp*fn) / sqrt((tp+fp)(tp+fn)(fp+tn)(fn+tn))`, in `[-1, 1]`.
- `ChiSquare` -- Pearson's chi-square statistic for the same 2x2 table
  (`n * phi**2`, optionally Yates-corrected); CMAR's significance test.
- `Entropy` -- CN2's original heuristic: the *negated* binary entropy of the
  covered class distribution, `p*log2(p) + (1-p)*log2(1-p)` with
  `p = tp/(tp+fp)`. Negated so a pure rule scores 0, the best, and an even
  50/50 split scores -1 bit, the worst, matching this module's convention
  where CN2 itself minimizes plain entropy directly.
- `LikelihoodRatio` -- CN2's significance-testing statistic: the G-test /
  log-likelihood-ratio comparing a rule's covered class counts against what
  the dataset's prior alone would predict, `2*(tp*ln(tp/e_tp) + fp*ln(fp/e_fp))`
  with `e_tp`/`e_fp` the prior-implied expected counts. Higher means further
  from chance, already matching the higher-is-better convention with no sign
  flip.

Heuristics that read past `tp`/`fp`/`fn`/`tn`, or compose others:

- `LengthPenalized(base, penalty)` -- wraps another heuristic and subtracts
  `penalty * stats.length` (a simple Occam's-razor-style complexity penalty).
- `MinimalLength` -- `-length`: shorter rules score higher. Meant purely as a
  lexicographic tie-break inside a `LEF`, not as a heuristic on its own.
- `LEF` -- Michalski's Lexicographic Evaluation Functional (AQ): an ordered
  list of heuristics, where the first one decides unless it ties. It is itself
  a `RuleHeuristic`, so it works anywhere one is expected.
- `FoilGain` -- Quinlan's information gain, `tp * (log2(tp/(tp+fp)) -
  log2(parent.tp/(parent.tp+parent.fp)))`, the one built-in `GainHeuristic`.
- `DeltaGain(base)` -- turns any ordinary heuristic into a `GainHeuristic`:
  scores a refinement by how much `base`'s own score improved relative to its
  parent.

A `GainHeuristic`'s `score(stats, parent_stats)` takes the parent rule's
`RuleStats` as a *mandatory second argument*, not an optional field read off
`stats`: a gain score is only meaningful relative to one specific parent, not
on a shared scale comparable across different parents/search steps, so a
caller needs to know it's holding a `GainHeuristic` (`isinstance(heuristic,
GainHeuristic)`) and supply the parent explicitly; calling
`FoilGain().score(stats)` with one argument raises `TypeError`.
`RuleStats.universal(n_pos, n_neg)`/`RuleStats.empty(n_pos, n_neg)` give the
"covers everyone"/"covers no one" degenerate stats directly from a dataset's
totals, without needing an actual `Rule` on hand -- the natural parent when
there's no specific prior refinement to compare against, e.g. for
`RuleHeuristic.plot_isometrics`'s `parent=` argument, below.

How heuristics differ geometrically -- whether their isometrics (curves of
constant score in coverage space) are a pencil of lines, parallel lines, or
curves -- is described under *Isometrics* in *Coverage space*, below.

### Coverage space

Rules and rule models can be analysed in coverage space (Fürnkranz & Flach,
2005): a rule is a point (negatives covered, positives covered), so a rule
set, a decision list or a single rule's refinement can be read off a plot.
Every `RuleModel` gives the coordinates, and `pyrulearn.evaluation` adds the
plots and the convex-hull AUC:

- `coverage_space(data, positive_class)` gives the classic (negatives
  covered, positives covered) coordinates for one-vs-rest analysis, using
  the raw, order-independent coverage, the same for every type.
- `RuleList.coverage_path(data, positive_class)` gives the *cumulative*
  (negatives, positives) coverage as rules are tried in order, starting at
  `(0, 0)`: point i is what's been decided after the first i rules, using
  each one's unique/fired coverage.

`coverage_space_plot(ruleset, data, positive_class)` (in
`pyrulearn.evaluation`) adapts to the classifier type: a `RuleSet` is drawn
as a scatter of *every* rule's own raw coverage w.r.t. `positive_class`
(regardless of the rule's own target), optionally with
`build_refinement_graph` edges connecting rules that differ by one
literal. `show_convex_hull=True` additionally draws the boundary of the
area under the ROC-convex-hull curve -- `(0, 0)` to `(N, 0)` to `(N, P)`
and back to `(0, 0)` via the hull itself, *not* a bare closing edge
straight back to `(0, 0)` (which would just sit on top of the
random-guess diagonal) -- restricted to just the rules whose own
`target == positive_class`: the hull/AUC are only meaningful for one
target class at a time, so a mixed-target `RuleSet` needs one call per
class rather than one hull mixing them (`coverage_space_auc(ruleset,
data, positive_class)` gives the number directly -- exactly the
enclosed area normalized by `N * P` -- and is also shown in the
boundary's legend label). A `RuleList` is instead drawn as
`coverage_path`'s points connected by directional **arrows**, in order,
starting at `(0, 0)` (`show_refinements`/`show_convex_hull` don't apply
in this mode -- the arrows already show the sequential structure). Pass
`aspect="square"` (old behavior), `"rectangular"` (box width:height =
negatives:positives, unclamped), or the default `"auto"` (same, but
clamped to `[1/max_aspect_ratio, max_aspect_ratio]` so a very skewed
dataset still gets a readable plot rather than a sliver).

`rule_refinement_plot(rule, data, positive_class=None)` plots a
single `Rule`'s own specialization path instead, also as directional
arrows: starting at `(N, P)` (the empty rule, covering everyone) and
adding one condition at a time in the rule's own order -- monotonically
non-increasing in both coordinates, ending at the rule's actual coverage.
`positive_class` defaults to `rule.target` -- for a single rule,
"positive" naturally means the class *it* predicts (its head), so
plotting a rule that predicts `"neg"` without overriding `positive_class`
correctly counts `"neg"` examples as the true positives, not whichever
class happens to be the caller's usual default. A rule needs a
meaningful order for this to say much; `Rule.order_by_precision(data)`
computes one greedily (FOIL/CN2-style: at each step, add whichever
remaining condition gives the highest precision *in combination with*
what's already chosen) if the rule doesn't already have one --
`Rule.reorder(...)` sets one explicitly instead.

#### Isometrics: pencil, parallel and curved

An *isometric* is a curve of constant heuristic score in coverage space
(`RuleHeuristic.plot_isometrics`, below, draws them), and its shape sorts the
heuristics into three groups (see Fürnkranz & Flach, *"ROC 'n' Rule
Learning"*, Machine Learning 2005, for the full analysis). In their
terminology the isometrics of the first two groups, pencil and parallel, are
both *linear* (straight lines), and those of the third are *non-linear*.

- **Pencil: straight lines through a common pivot point.** `Precision`,
  `Laplace`, `MEstimate`, `GeneralizedMEstimate`, `GHeuristic` and `FBeta`.
  The pivot is the origin for `Precision`, `(-1,-1)` for `Laplace`, a
  prior-dependent sliding point on the line `fp+tp=-m` for `MEstimate`
  (freely placeable on that same line via `GeneralizedMEstimate`'s `cost`
  parameter), `(-g, 0)` on the fp-axis for `GHeuristic`, and
  `(-beta**2*n_pos, 0)`, also on the fp-axis, for `FBeta`. Because the lines
  are not parallel, these heuristics can prefer a lucky low-coverage rule
  over a robust one. `Entropy` belongs here too -- same pivot as `Precision`
  (it also depends only on `p = tp/(tp+fp)`, constant along any ray from the
  origin) -- but each score value is a symmetric *pair* of rays, not one:
  `h(p) == h(1-p)`, so entropy alone can't tell a 90%-positive rule from a
  90%-*negative* one apart.
- **Parallel: parallel lines.** `CoveredPositives`, `CoveredNegatives`,
  `UncoveredPositives`, `UncoveredNegatives`, `Recall`, `WRAcc`, `Accuracy`,
  `CoverageDifference`, `Support`, `Coverage`, `LinearCost`, `YoudenJ` and
  `LinearCostRates`. `CoveredPositives`/`CoveredNegatives` are this group's
  two degenerate limits, horizontal/vertical, i.e. zero weight on the other
  variable, with `UncoveredPositives`/`UncoveredNegatives` in the same two
  groups, just affine-shifted by the constant `n_pos`/`n_neg`; `Recall` is
  `CoveredPositives`'s rate-normalized twin. A rule is optimal for *some* such
  heuristic if and only if it lies on `coverage_space_plot`'s convex hull.
- **Curved (non-linear).** `Correlation` (hyperbolic isometrics) and
  `LikelihoodRatio` (isometrics that bow away from the origin, since it
  depends on absolute covered counts, not just their ratio); and `FoilGain`,
  which, once made plottable via a fixed `parent`, isn't even a single smooth
  curve -- its `tp *` factor can make a low-tp, low-precision point score the
  same as a higher-tp, higher-precision one, producing a visibly non-convex,
  "hooked" isometric.

#### CoverageSpace: layering heuristics, rulesets, and refinement paths

`coverage_space_plot`/`rule_refinement_plot` are thin, standalone-figure
wrappers around `CoverageSpace` (also in `pyrulearn.evaluation`) -- the
reusable, composable version of the same idea. A `CoverageSpace` owns
the plot's dimensions (`n_pos`/`n_neg`) and its matplotlib `Axes`;
`plot_ruleset`/`plot_rule_refinement` draw a layer into it (the same
content as the two standalone functions, minus the one-off figure
setup), and `RuleHeuristic.plot_isometrics` (in `pyrulearn.heuristics`,
inherited by every concrete heuristic since it only needs `score`) draws
that heuristic's isometrics -- curves of constant score -- into one too,
so several layers can share one plot:

```python
from pyrulearn.evaluation import CoverageSpace
from pyrulearn.heuristics import WRAcc

space = CoverageSpace.from_data(train_rep, positive_class="pos")
WRAcc().plot_isometrics(space=space, levels=10, filled=True)
space.plot_ruleset(rules, train_rep, "pos", show_refinements=False, annotate=True)
```

`n_pos`/`n_neg` are either given explicitly, taken from real data via
`CoverageSpace.from_data(data, positive_class)`, or
-- if both are omitted -- default to `100`/`160` (the smaller value on
Pos), a fixed, dataset-agnostic space for looking at a heuristic's
isometrics in the abstract (e.g. bare `Precision().plot_isometrics()`,
no `data` needed at all). `normalized=True` switches to the
classic **ROC space** (`FPR`/`TPR` in `[0, 1] x [0, 1]`) instead of raw
counts -- the heuristic is still evaluated against the space's real
`n_pos`/`n_neg` internally (most heuristics, e.g. `WRAcc`, genuinely
depend on the class prior, not just the displayed rate), only the
*display* coordinates are normalized. A normalized space's box is always
displayed as a perfect square, regardless of `n_pos`/`n_neg` or
`aspect=` -- the conventional ROC-space shape; a non-normalized space's
box genuinely reflects the true negatives:positives ratio (`aspect=
"rectangular"` for the unclamped ratio, the default `"auto"` for the
same ratio clamped to `[1/max_aspect_ratio, max_aspect_ratio]` so a
heavily skewed dataset doesn't render as an unreadable sliver, or
`"square"` to ignore the ratio entirely).

`show_labels` controls whether *concrete numbers* -- tick labels beyond
the two endpoints, and isometric value labels (`ax.clabel`) -- are drawn
at all. `None` (the default) means "yes, if `n_pos`/`n_neg` were
actually given", "no" otherwise, since the default space's `100`/`160`
are arbitrary placeholders and printing them as if they meant something
would be misleading; pass `True`/`False` explicitly to override either
way. Axis text (`"negatives covered"`/`"positives covered"`, or
`"FPR"`/`"TPR"` when normalized) and the two endpoint ticks are always
shown regardless -- only the tick *labels* change: symbolic `"0"`/`"N"`
and `"0"`/`"P"` when `show_labels` is `False` and the space isn't
normalized (the real counts aren't meaningful), literal `"0"`/`"1"` when
it is normalized (that endpoint is always true), or the real numbers
(with matplotlib's normal, denser default ticks) when `show_labels` is
`True`. A light gray minor-tick grid is always drawn too, styled like
the random-guess diagonal, for visual reference regardless of whether
concrete numbers are shown. The plot's `title` (if given) is shown
regardless of `show_labels`.

`plot_isometrics(levels=10, filled=False, cmap=None)` draws line
contours (label text via `ax.clabel`, skipped when `show_labels` is
`False`) by default, or filled bands with `filled=True`. Its colors run
from red for low scores to green for high ones, as in a heatmap
(`isometrics_cmap()` in `pyrulearn.evaluation`, the default for
`cmap=None`), and the lines lie on lightly shaded bands of the same
colors (`shade=0.3`, their opacity; `shade=0` for bare lines); `levels` is
passed straight to matplotlib's `contour`/`contourf`, as is everything
else passed via `**kwargs` -- `colors=`, `linestyles=`, `linewidths=`,
etc. are all genuinely configurable this way, e.g. `colors="tab:blue",
linestyles="dashed", linewidths=0.8` to keep one heuristic's isometrics
visually distinct from another's (or from an overlaid rule's own
refinement-path color) when layering several on the same space --
`cmap` and the shading are dropped automatically when `colors` is given,
since matplotlib rejects both together. `parent=` fixes the parent `RuleStats` passed to
a `GainHeuristic`'s `score` across every grid point -- the same value
throughout, not one per point -- which is what makes a gain-style
heuristic like `FoilGain` plottable at all here (it otherwise raises
without one); `RuleStats.universal(space.n_pos,
space.n_neg)` is the natural choice, reading the result as "gain over
the rule that covers everyone." Plotting several heuristics' isometrics
side by side makes their families concrete -- `Precision`/`Laplace` fan
out from the origin, `WRAcc` is parallel lines, `Correlation` bows into
hyperbolic curves -- and, empirically, that `Entropy`'s isometrics are
*also* a pencil through the origin (same pivot as `Precision`), just
with each score value split into a symmetric pair of rays, and that
`FoilGain`'s (with a fixed parent) aren't even a single smooth curve
(see *Isometrics*, above). See `demos/heuristic_isometrics.py`
for all of this in practice, including layering `Precision`/`Accuracy`
isometrics under a rule's own refinement path.

`plot_isometric_through_point(neg, pos, space=None, mark_point=True,
...)` draws just the *one* isometric that passes through a specific raw
coverage point -- every other point on that curve scores exactly as well,
under this heuristic, as `(neg, pos)` does -- and scatters the point
itself so it's clear which one. `plot_isometric_through_rule(rule,
data, positive_class=None, space=None, ...)` is the common-case
convenience: the point is `rule`'s own raw coverage (`positive_class`
defaulting to `rule.target`, same resolution as `RuleStats.from_rule`),
and `space` defaults to `CoverageSpace.from_data(data,
positive_class)` -- sized to the real data the rule was scored against,
not the dataset-agnostic default. Both accept the same styling `**kwargs`
as `plot_isometrics` (they're built on it).

## Learning algorithms

`RuleLearner` (in `pyrulearn.learners`) is the shared entry point for
running a comparative experiment across several rule-learning
algorithms without the driver caring which is which:

```python
class RuleLearner(ABC):
    def fit(self, data: DataRepresentation, model: Optional[type] = None, **model_kwargs) -> RuleModel: ...
```

Learners come in two kinds, and both produce the same `RuleModel` types,
so their results can be compared directly:

- **native learners** (`NativeRuleLearner`) are implemented in pyrulearn
  itself and learn directly on the `DataRepresentation` -- see *Native
  learning algorithms*;
- **interfaced learners** (`ExternalRuleLearner`) run an external
  implementation (scikit-learn, wittgenstein, imodels, Weka, LORD, pyarc,
  mlrl-boomer, realkd) and import its model back as rules -- see
  *Interfaced learning algorithms*.

How several classes are handled is the same for both -- see *Multiclass
classification* at the end of this section.

### Native learning algorithms

**`NativeRuleLearner`** -- for a rule-induction algorithm implemented
directly in pyrulearn (e.g. `pyrulearn.learners.seco`'s from-scratch Pypper/CN2/
AQR, or `pyrulearn.learners.pylord.PyLORD`): `fit` induces straight against
`data`, with no external algorithm call and no `RuleImporter`
round-trip at all.

**Components first, specialized versions second.** pyrulearn builds its
learners from exchangeable components -- a search, a heuristic, a
covering strategy, a combiner, a multi-class decomposition -- rather
than implementing each algorithm in one piece. A variant is then one
constructor argument away, and a learner runs on every data
representation, since only the components touch the data. Where an
algorithm also has a much faster implementation in one piece, that is
kept as a separate learner under its own name, learning the same models.

CPAR is the example. `CPAR` is a configuration of the SeCo framework
(see *Separate-and-conquer*): `FoilGain` as the heuristic, a gain ascent
that branches into copies of the rule at nearly-as-good conditions
(`GainAscentHillClimbing(min_gain=0.7, branch_similarity=0.99)`, which
yields several rules per search), `WeightedCovering(MultiplicativeReweighting,
PositiveWeightBelow)` as the covering, and `TopKMeanCombiner` to predict;
the search counts every open condition at once, scores them with one
`batch_score` call and builds only the rules followed, with constraint
propagation. Nothing CPAR-specific is left in the class but that choice
of parts -- and the branching search is now a part any SeCo learner can
use. `DenseCPAR` scores
every feature of a search node in one matrix product over a dense copy
of the data, with no propagation. Both learn the same models (tested);
`DenseCPAR` is 2-4x faster, but needs a `HorizontalBitmapRepresentation`
and only does one-vs-rest. Exchanging a component works the same for
both:

```python
from pyrulearn.heuristics import DeltaGain, Laplace
from pyrulearn.learners.cpar import CPAR, DenseCPAR
from pyrulearn.learners.seco import AdditiveReweighting, PositiveWeightBelow, WeightedCovering

CPAR(heuristic=DeltaGain(Laplace()), min_gain=0.01)          # grow by Laplace improvement
DenseCPAR(covering=WeightedCovering(AdditiveReweighting(), PositiveWeightBelow(0.05),
                                    max_rounds=None))       # CN2-SD's additive reweighting
```

`ENDER`/`DenseENDER` and `Boomer`/`DenseBoomer` are paired the same way:
`ENDER`'s loss and impurity criterion are components, its search runs on
every representation, and `DenseENDER` learns the same models 1-8x
faster on a dense matrix.

#### Separate-and-conquer (SeCo)

`pyrulearn.learners.seco.SeCo` is the separate-and-conquer (covering)
framework (Fürnkranz 1999; Fürnkranz, Gamberger & Lavrač 2012) that most
native learners are configurations of. For one target class, the
**covering loop** repeats: learn one rule, remove every example it covers
(of both classes) from the scope, and continue on the rest -- until
`stop_covering` fires on the rule just learned (it is then discarded), no
rule is found, no positive example is left uncovered, or `max_rules` is
reached. An optional `optimization=` then revisits the finished rule set
as a whole (`ReplaceReviseOptimization`, RIPPER's replace/revise phase).
Instead of removing covered examples, `covering=` can also reweight them
-- see *Weighted covering* below.

Each rule comes from a **`SingleRuleLearner`**, assembled from
exchangeable building blocks:

| building block | options |
|---|---|
| search | `BranchAndBoundSearch(max_conditions, k)` (exhaustive: the optimal rule, or the `k` best, by branch and bound, after OPUS); `BeamSearch(beam_width)` (keeps the best `beam_width` candidates, returns the best rule seen); `HillClimbing` (one candidate, stops at a local optimum of the heuristic); `GainAscentHillClimbing` (the only search for a gain heuristic such as `FoilGain`, which scores a refinement relative to its parent; `min_gain=` raises the bar from any positive gain, and `branch_similarity=` makes it yield several rules, see below) |
| heuristic | any `RuleHeuristic` -- see *Rule-evaluation heuristics* -- or, more generally, any `Objective` (see below) |
| where the search starts | `EmptyRuleAllFeatures` (the empty rule, all features open -- default); `FeatureSubset`; `SeedExample` (a rule that must keep covering one chosen example, as in AQ) |
| data preparation | `NoSplit` (default); `GrowPruneSplit` (grow on one part, prune on a held-out part -- IREP, RIPPER) |
| post-processing | `NoPostProcessing` (default); `ReducedErrorPruning` (drop trailing conditions while that improves the rule on the pruning data) |
| `filtering` / `stopping` | pre-pruning criteria from `pyrulearn.pruning` (`ThresholdPrePruning(heuristic, threshold)`, `EncodingLengthRestriction`, `ProductiveRule`, combinable with `AllOf`/`AnyOf`) |

`filtering` and `stopping` take the same criteria but act differently: a
**filtering** criterion only decides which rules may be returned -- the
search still runs to its end -- while a **stopping** criterion also ends
the search as soon as its current best candidate fails it. A candidate
that covers no positive example is never chosen or returned, whatever the
heuristic says about it.

**What a search maximizes: objectives.** A search never calls a
heuristic directly; it talks to an `Objective`
(`pyrulearn.learners.seco`), which owns everything about scoring:

- the statistics of a rule, and of all open one-condition refinements of
  it at once, counted through the data representation's own primitives
  without building any refinement;
- their scores, and an optimistic bound on what any further refinement
  can score (for pruning and for branch and bound);
- which refinements change nothing (they keep exactly their parent's
  statistics, so float rounding can't make them look better) and which
  have nothing left to gain (dropped and masked out further down).

A `RuleHeuristic` is one objective, `HeuristicObjective`: its statistics
are the class counts `RuleStats` (tp, fp, fn, tn), its bound is the
score at (tp, 0), and a refinement covering no positive is dead -- every
search wraps a heuristic this way, so passing a heuristic works as
always. A `ValueSumObjective` is a function of sums of arbitrary per-row
values over the covered rows instead -- class counts are the special case
of class indicators as values -- counted with one `batch_cover_sums` per
search step on every representation. Optimal rule boosting's gain is one
(`XGBGain`, over the loss derivatives g and h; see *Boosting*):
with it, `BranchAndBoundSearch`, `BeamSearch` and `HillClimbing` find
boosting rules exactly as they find classification rules. An objective
may also give an exact bound from a rule's covered rows
(`exact_bound`), which branch and bound uses for every refinement it
builds. Filtering and stopping criteria take `RuleStats`, so they work
with heuristic objectives only.

The named learners are configurations of these blocks, and every block
can still be overridden through the constructor:

| learner | search | heuristic | criterion | also |
|---|---|---|---|---|
| `CN2` (Clark & Boswell 1991) | `BeamSearch(5)` | `Laplace` | likelihood-ratio significance test, `stopping` | |
| `AQR` (Clark & Niblett 1989) | `BeamSearch(maxstar=5)` from a `SeedExample` | a `LEF` (covered positives, then covered negatives, then length) | consistency: no covered negative, `filtering` | the AQ baseline; overfits noisy data |
| `PFoil` (Mooney 1995; Quinlan 1990) | `GainAscentHillClimbing` | `FoilGain` | Quinlan's encoding-length restriction, `stopping` | |
| `PFossil` (Fürnkranz 1994) | `HillClimbing` | `Correlation` | correlation below 0.3, `filtering` | |
| `Pypper` (Cohen 1995) | `GainAscentHillClimbing`, not stopping at a gain peak | `FoilGain` | at least two covered positives | `GrowPruneSplit` (2/3 grow, 1/3 prune) + `ReducedErrorPruning`; covering stops on the encoding-length restriction or a pruned rule below 0.5 precision; then `ReplaceReviseOptimization` (`k=2` passes) |
| `Opus` (Webb 1995) | `BranchAndBoundSearch`: the optimal rule | `Laplace(n_classes=c)` | | in `learners.opus` |
| `CPAR` (Yin & Han 2003) | `GainAscentHillClimbing(min_gain=0.7, branch_similarity=0.99)`: several rules per search | `FoilGain` | | `WeightedCovering(MultiplicativeReweighting(2/3), PositiveWeightBelow(0.05))`; predicts with `TopKMeanCombiner` -- see *Weighted covering* |

```python
from pyrulearn.learners.seco import CN2, PFossil

CN2(beam_width=1).fit(train_rep)                  # CN2 with a greedy search
PFossil(correlation_threshold=None).fit(train_rep)   # FOSSIL without its cutoff
```

**Several rules per search.** A search can also yield several rules at
once (`RuleSearch.search_all`; by default just the one `search` returns),
and the covering loop then accepts each in turn -- judged on the scope
they were all found in, each reweighting or removing the examples it
covers. `GainAscentHillClimbing(branch_similarity=s)` is such a search:
at every step it also follows each condition whose gain is at least `s`
times the best one, as a copy of the rule grown on the same way, and
yields every copy's final rule. That's CPAR's search (Yin & Han 2003),
which makes `CPAR` a configuration of `SeCo` like the learners above --
and the branching usable with any of them:

```python
from pyrulearn.learners.seco import GainAscentHillClimbing, PFoil

PFoil(search=GainAscentHillClimbing(branch_similarity=0.9)).fit(train_rep)   # FOIL with copies
```

**Optimal rules.** `BranchAndBoundSearch` finds the rule with the
highest score among *all* conjunctions of the open conditions (up to
`max_conditions`), not just the best one a beam or a hill climber
reaches -- after OPUS (Webb 1995). It walks the set-enumeration tree
(a rule is only extended by conditions after its last one, so every
condition set comes up once), best-first by an optimistic bound -- the
score if every covered positive were kept and every covered negative
dropped -- and never expands a rule whose bound can't beat the best (or
`k`-th best) found. Like OPUS, it removes a condition from a whole
subtree once that condition can't help there (bounded out, covering no
positive, or changing nothing); for the single best rule it also
applies Webb's *cannotImprove* rule (`dominance_pruning`): a condition
that drops no covered negative, or whose child a sibling dominates
(covering all its positives and none of the negatives it drops), can't
lead to a better rule. The bound needs a heuristic that rewards
covered positives and penalizes covered negatives (`Laplace`, `WRAcc`,
`Precision`, ...). With `k`, `search_all` yields the `k` best rules of
distinct coverage. Two learners use it, in `pyrulearn.learners.opus`:
`Opus`, separate-and-conquer with the optimal rule in every step (a
`SeCo` configuration; by default with the Laplace estimate over the
data's classes, `Laplace(n_classes=c)`, as in Webb's experiments), and
`OpusTopK`, k-optimal rule discovery in the manner of Webb's Magnum Opus -- the `k` best rules per class by `WRAcc`
(leverage), on all the data, no covering. Like Magnum Opus, it keeps
only *productive* rules by default (`pyrulearn.pruning.ProductiveRule`,
a `filtering=` criterion: a rule's precision must be strictly higher
than that of every generalization, the empty rule included), so the `k`
rules aren't padded with variants of a better, simpler rule such as
`sex = male, parch != 6` next to `sex = male`; `productive=False` turns
it off. Exhaustive search can be slow: with `Laplace`, whose optimum is a narrow, pure rule, the bound
prunes little and covering needs many rules -- `max_conditions` and a
coarser heuristic help.

```python
from pyrulearn.learners.opus import Opus, OpusTopK
from pyrulearn.heuristics import WRAcc

Opus(heuristic=WRAcc(), max_conditions=3).fit(train_rep)   # covering with optimal WRAcc rules
OpusTopK(k=10).fit(train_rep)                               # the 10 best rules per class
```

A new variant is just a `SeCo` with its own blocks -- here the plain
learner of Janssen & Fürnkranz's heuristic study (greedy search that
returns the best rule seen, no stopping criterion, covering stops once a
rule covers no more positives than negatives), with the m-estimate:

```python
from pyrulearn.heuristics import CoverageDifference, MEstimate
from pyrulearn.learners.seco import BeamSearch, SeCo, SingleRuleLearner
from pyrulearn.pruning import ThresholdPrePruning

learner = SeCo(
    SingleRuleLearner(heuristic=MEstimate(22.466), search=BeamSearch(beam_width=1)),
    stop_covering=ThresholdPrePruning(CoverageDifference(), 0, operator="<="),
)
model = learner.fit(train_rep)
```

The covering loop always learns rules for one class against the rest.
With several classes (or both classes of a binary problem), `fit(data)`
decomposes the problem: one-vs-rest (`ConceptSet`) for `CN2`, `PFoil` and
`PFossil`; least-frequent-first ordered peeling (`ConceptCascade`) for
`Pypper`; one decision list over all classes for `AQR`. Other
decompositions are chosen with `fit(data, model=...)` -- see
*Multiclass classification* below.

**`PyLORD`** (`pyrulearn.learners.pylord`) reuses the same building
blocks without a covering loop. After LORD (Huynh, Fürnkranz & Beck
2023), every training example seeds a rule search (`SeedExample`): a
greedy m-estimate search (`BeamSearch(beam_width=1)`), then
`ReducedErrorPruning` on the full training data. It keeps each rule that
is the best one for at least one example, and predicts with the best
covering rule. It is natively multi-class, and a simplified
reimplementation -- the reference LORD is interfaced as `JavaLORD` (see
*LORD: the Java reference implementation*).

#### Weighted covering

`SeCo`'s `covering=` sets how the covering loop's scope changes after
each accepted rule. `RemovalCovering` (the default) is classic
separate-and-conquer: everything the rule covers leaves the scope.
`WeightedCovering(reweighting, stop)` (Gamberger & Lavrač 2002;
Lavrač et al. 2004) keeps covered examples in scope with new weights
instead. Later rules may then overlap earlier ones, and are steered
towards the examples the weights emphasize. Weighted covering
algorithms differ in two pluggable components:

- the **reweighting scheme** (a `Reweighting`, like a heuristic):

  | scheme | new weights | used by |
  |---|---|---|
  | `MultiplicativeReweighting(gamma)` | a positive covered by `k` rules: `gamma ** k`; negatives 1 | CN2-SD; CPAR's decay (`gamma = 2/3`) |
  | `AdditiveReweighting()` | a positive covered by `k` rules: `1 / (k + 1)`; negatives 1 | CN2-SD |
  | `AdaBoostReweighting()` | covered examples `* exp(-y C)`, `C` the rule's confidence; rescaled | Slipper |
  | `LRIReweighting(power=3)` | `1 + e ** 3`, `e` the rules that erred on the example (either class) | Lightweight Rule Induction |

- the **stop criterion** (`CoveringStop`; several may be given, any ends
  the loop): `CoveredAtLeast(k)` (every positive covered `k` times,
  CN2-SD, the default with `k = 5`), `PositiveWeightBelow(fraction)` (the
  positives' total weight below a fraction of the start, CPAR), or
  `Rounds(n)`. `max_rounds` (default 100) is a safety net on top.

The loop keeps its state in a `CoveringState`: the current scope, how
often each example was covered, how many rules erred on it, and the
round. A rule found again counts as a round and reweights as usual, but
appears only once in the resulting rule set, since a copy adds nothing
to an unweighted set. `CN2`, `AQR`, `PFoil` and `PFossil` accept
`covering=`; the all-classes seed covering behind `model=DecisionList`
requires removal, since first-match semantics assumes it. A reweighting
that fits a weight for each rule (`AdaBoostReweighting`) belongs to a
boosting learner (`Slipper`, below), whose model can hold the weights;
`SeCo`, which builds unweighted rule sets, refuses it.

CN2-SD is CN2 with `WRAcc` and weighted covering:

```python
from pyrulearn.heuristics import WRAcc
from pyrulearn.learners.seco import CN2, CoveredAtLeast, MultiplicativeReweighting, WeightedCovering

model = CN2(heuristic=WRAcc(),
            covering=WeightedCovering(MultiplicativeReweighting(0.5), CoveredAtLeast(5))).fit(train_rep)
```

The scope the building blocks receive as `example_mask` is then a weight
vector instead of a boolean mask; it multiplies with the data's own row
weights. Seed picking and the grow/prune split use the rows with a
positive weight, and everything that counts uses the weights.

Two learners are built on this machinery, each with its own way of
growing rules and of combining them in prediction; their rules carry
no weights of their own:

**CPAR.** `pyrulearn.learners.cpar.CPAR` (Yin & Han 2003; `DenseCPAR` is the
faster implementation, see *Native learning algorithms*) grows rules for each
class FOIL-style on weighted examples: starting from the empty rule, it
adds the condition with the highest weighted FOIL gain until none gains
at least `min_gain` (0.7). Every other condition whose gain is within
`gain_similarity` (0.99) of the best starts a copy of the rule that is
grown on as well, so one search can yield several rules. After each rule
the positives it covers are multiplied by `decay` (2/3,
`MultiplicativeReweighting`), until the positives' total weight is below
5% of the start (`PositiveWeightBelow`). The model is a `ConceptSet`
predicted by `TopKMeanCombiner`: the class whose best `k` (5) covering
rules have the highest mean expected accuracy `(nc + 1) / (n + K)`
(`GeneralizedMEstimate(m=K, cost=1/K)` for `K` classes). The combiner is
usable with any rule set: `TopKMeanCombiner(heuristic, k)`. `CPAR` is a
`SeCo` configuration (see *Separate-and-conquer*: the search is
`GainAscentHillClimbing(min_gain, branch_similarity=gain_similarity)`),
and each part is a constructor argument: `heuristic=` (a
`GainHeuristic`, default `FoilGain()`; `DeltaGain(h)` turns any heuristic
into one), `covering=` (default
`WeightedCovering(MultiplicativeReweighting(decay),
PositiveWeightBelow(min_total_weight))`) and `combiner=`. Like every
SeCo learner it also decomposes the classes differently on request
(`model=ConceptCascade`, `model=PairwiseModel`).

```python
from pyrulearn.learners.cpar import CPAR

model = CPAR(k=5).fit(train_rep)
```

**Lightweight Rule Induction (LRI).** `pyrulearn.learners.lri.LRI` (Weiss & Indurkhya 2000) learns, for every
class, `n_rules` unweighted DNF rules, and predicts the class with the
most satisfied rules. A DNF rule is a `ConceptModel` -- up to `max_terms`
terms with the same head, firing where any term does -- and the model an
`EnsembleModel` voting over them all. Per class, each rule is grown on
the whole (reweighted) data. A term is what a `SingleRuleLearner` finds:
a `search` (default `HillClimbing(max_conditions=5,
stop_at_local_optimum=False)`: add the best condition each step until the
term covers no negative) over the objective `LRIError`, the weighted
`FP + k·FN` among conditions keeping a true positive (`k` doubled while
the cheapest condition would drop them all). The next term is grown on
the cases no earlier term covers, until `max_terms` or no positive is
left. After each rule, a `WeightedCovering` reweights the cases by `1 +
e**3`, `e` the number of rules so far that err on them (`LRIReweighting`).
No pruning, no default rule, and a rule found again is kept, adding
votes. After `freeze_features_after` rules only the features used so far
remain candidates (a `FeatureSubset`). The paper's Table 2 grows a term
"until FN = 0", which can't be meant (FN only grows as conditions are
added); terms stop at FP = 0 here. `LRI` runs on every data
representation; `DenseLRI` is the same learner with a hand-written term
search on a dense matrix. On features without constraints among them
(no negations, no thresholds) both learn identical models; with them,
`LRI` never adds a condition the term already implies, where `DenseLRI`
can. Unlike `CPAR` and `ENDER`, the modular learner is the faster one on
larger data -- 7s (horizontal bitmap) or 2.5s (N-lists) against 47s for 10 rules
per class on 32,000 rows of `adult` -- since it only counts the rows a
term still covers; on a few hundred rows `DenseLRI` is about twice as
fast.

```python
from pyrulearn.learners.lri import LRI
from pyrulearn.learners.seco import BeamSearch

model = LRI(n_rules=50, max_terms=4).fit(train_rep)
beam = LRI(n_rules=50, search=BeamSearch(beam_width=3, optimistic_pruning=False)).fit(train_rep)
```

**Deep rule networks.** `pyrulearn.learners.deep` learns `DeepModel`s:
fully connected networks of alternating AND and OR layers with 0/1 edge
weights over the data's features (see `DeepModel`). `layers` are the
hidden layer sizes (by default one AND layer of `ceil(sqrt(n))` nodes,
a DNF), `first` the type of the layer above the inputs (`"and"`, or
`"or"`); the output layer has one node per class, or for two classes
one for the less frequent class, the other the default -- or, with
`target_class`, one for that class (the most frequent other class the
default). Which class the output learns can matter a lot: a formula may
have a small DNF where its negation has none (see the demo). A row several
outputs cover gets the class with the most covering rules, a row none
covers the most frequent class. The start is a component, `init=`
(`DeepRuleLearner._initialize`, which a subclass may override):
`RandomInit()` (random rules of about `avg_rule_length` conditions, none
empty, none contradictory by the `DataSpec`'s constraints; several
outputs divide the rules among them), `ModelInit(deep_model)` (another network's
weights) or `RuleSetInit(rules)` (a rule set, or a learner fitted first:
the network starts out as the rules' DNF per output, passed up through
the layers). `DeepLocalSearch` is Beck & Fürnkranz's stochastic hill
climbing: on mini-batches of `batch_size` (50) rows, the best single
edge flip until none improves, `n_epochs` (5) passes, the best weights
on the whole training set kept, then a last climb on all rows. A flip
that adds a condition to a rule drops the conditions contradicting it by
the `DataSpec`'s constraints (`moves="consistent"`, through
`contradictions(spec)`: other values of a nominal attribute, a negation,
contradicting thresholds; Beck's `moves="attribute"` drops the other
values of its attribute). Beck's
own setup is `moves="attribute", init=RandomInit(min_rule_length=0,
consistent=False)`. The errors of all flips of a node are computed at
once from the node's counts of true (OR) or false (AND) inputs.
`DeepModel.pruned()` keeps the backbone: edges into nodes the outputs
don't depend on are set to 0.

`DeepGenetic` is a memetic algorithm on the same base: a population of
starts (`init`, plus `seeds=`, e.g. `RuleSetInit(Pypper())`), tournament
selection, the best kept (`elite`), and children made by crossover,
mutation (`mutations` random flips on average) and a few hill-climbing
steps (`local_flips` on a mini-batch); finally the best network is
climbed on all rows. Crossover first aligns the parents: hidden nodes
can be permuted without changing the network's function, so the second
parent's nodes are matched to the first's, layer by layer, by their
values on the training data (Hungarian method); the child then takes
each node's incoming edges from either parent (`crossover="nodes"`), or
one node of the second parent with its cone -- everything below it
connected through edges of weight 1 -- and the rest from the first
(`"subtree"`); `"mixed"` picks one per child, `"none"` copies the first
parent (the control).

Both are anytime algorithms. With `trace=True`, `model.trace` lists every
new best network on the training set during the fit (`TracePoint`: the
effort so far in candidate-rows -- one candidate network on one row, so
local search and the GA, and batches of any size, compare --, the
seconds, the training error, the weights), and `model.trace_model(i)`
turns a point into a model, e.g. to measure the test accuracy along the
run.

`DeepSAT` fits the training data exactly by SAT solving (Seip, Fürnkranz
et al.): the weights and, per training example, every node's value are
Boolean variables, clauses tie each node to its inputs and each output
to the label (the known inputs and labels as constants), solved through
PySAT (`pip install python-sat`, the extra `sat`).
`minimize="inputs"` or `"weights"` makes it MaxSAT: the fewest
conditions, or the fewest edges -- the smallest network that still fits
exactly. By default by linear search (`maxsat="linear"`): solve, demand a
smaller network (an incremental totalizer, the bound as an assumption),
repeat; an anytime algorithm whose every step is a smaller exact fit,
recorded with `trace=True` and returned when `max_seconds` or
`max_conflicts` (per solver call, the same result on every machine) runs
out; `maxsat="rc2"` uses RC2, which only has a network at the optimum.
Solvers: Glucose 4 by default (PySAT can interrupt it; CaDiCaL, the
papers' solver, it can't), CaDiCaL for RC2. `constraints=True` (default)
adds the `DataSpec`'s constraints as clauses on the first layer's weights
-- no contradictory rules, no implied conditions.
Identical rows are merged; rows with the same features and different
labels keep the majority label (`conflicts="majority"`), since no network
fits them otherwise; an architecture too small for the data raises
`ValueError`. `size_penalty` (lambda) makes the examples soft: a
misclassified example costs 1, an edge (or condition, by `minimize`)
lambda, and MaxSAT minimizes `errors + lambda * size` -- a regularized
loss (0-1 loss, L0 penalty), anytime from the first solver call, and no
exact fit needed (conflicting rows simply cost an error); `None`, the
default, is the exact fit. Then `init` (e.g. `RuleSetInit(Pypper())`)
is the first solution: the search only looks for cheaper networks, so
the result is never worse than the start.
On the truth table of `(a | b) & (c | d) & (e | (f & g))`,
the papers' architecture `[6, 6]` with MaxSAT gives exactly that formula,
16 edges.

`DeepNeuroFuzzy` learns the weights by gradient descent on a fuzzy
relaxation (Wang et al., AAAI 2020: multilayer logical perceptrons with
random binarization): during training the weights are continuous in
[0, 1], an AND node computes `prod_j (1 - w_j (1 - x_j))`, an OR node `1 -
prod_j (1 - w_j x_j)`; mini-batches, Adam, `loss="mse"` (the paper's) or
`"log"`, an `l2` penalty that pushes unneeded edges to 0, the weights
clipped to [0, 1]. Random binarization replaces a random share
(`binarization_rate`, 0.75) of the weights by their 0/1 value in every
step, so the network learns to live with the discrete weights it ends
with; their gradient still updates the continuous weight
(`straight_through`, default -- without it, from the paper's small
start, an OR layer's weights hardly ever cross 0.5 and the network ends
up empty). After every epoch the weights are thresholded at 0.5; the
best of these networks on the training set is the result, and with
`trace` each new best is a point of the anytime curve. `init` maps a 0/1
start (e.g. `RuleSetInit(Pypper())`) to high and low weights. Our version
keeps `DeepModel`'s alternating layers. Deeper networks on small data need
more steps (`epochs`, `batch_size`).

All four take `size_penalty` (lambda) with one meaning: the objective is
the training error plus lambda per edge of the backbone (dead edges are
free), so an edge has to be worth 1/lambda misclassified examples;
`None` (default) minimizes the error alone. Local search and the genetic
algorithm apply it where the whole training set is evaluated -- the best
network, the fitness, the final climb -- not on mini-batches, where many
conditions change nothing and the penalty would strip the rules;
`DeepNeuroFuzzy` sums its loss over the training set and adds lambda
times the sum of the weights (L1); `DeepSAT` minimizes it exactly with
soft examples. `max_seconds` gives local search, the genetic algorithm
and `DeepNeuroFuzzy` a time budget (DeepSAT has its own): the search ends
with its best network when the time is up -- with unbounded epochs or
generations, an anytime run judged by time.

```python
from pyrulearn.learners.deep import DeepLocalSearch, DeepNeuroFuzzy, DeepSAT, RuleSetInit
from pyrulearn.learners.seco import Pypper

model = DeepLocalSearch(layers=[32, 8, 2]).fit(train_rep)
improved = DeepLocalSearch(layers=[32, 8, 2], init=RuleSetInit(Pypper())).fit(train_rep)
print(model)  # the stacked rules, without pass-through nodes
```

#### Boosting

Boosting learners reweight the examples too, but they also fit a
real-valued weight for every rule and predict with the weighted sum of
the rules that cover an example: the model is a `LinearRuleModel`, and
a rule learned in several rounds appears once, with its weights summed.

**Slipper.** `pyrulearn.learners.boosting.Slipper` (Cohen & Singer 1999) boosts
rules for one class against the rest. Each round it grows a rule on two
thirds of the examples, greedily maximizing `SlipperZ` (`sqrt(W+) -
sqrt(W-)` over the boosting weights), prunes it on the other third to
the prefix with the lowest boosting loss, takes it or the default rule
(whichever reaches the lower loss), gives it the confidence `C = 1/2
ln((W+ + eps) / (W- + eps))`, and reweights the examples it covers
(`AdaBoostReweighting`). After `n_rounds` rounds the model predicts the
class where the covering rules' confidences sum to more than zero. It is
a `LinearRuleModel`: a rule taken in several rounds appears once with its
confidences summed, and the default rules sum into the intercept.

```python
from pyrulearn.learners.boosting import Slipper

model = Slipper(n_rounds=20, random_state=0).fit(train_rep)
print(model.to_string(show_stats=False, weight_format="6.2f"))
```

```prolog
% conflict resolution: sum of rule weights per class, highest wins

% class: pos
 -1.09::pos(X) :- true.
  1.23::pos(X) :- f2(X), f3(X).
  1.25::pos(X) :- f0(X), f1(X).
  0.63::pos(X) :- \+f1(X), f2(X), f3(X).
...
```

With two classes it boosts the less frequent one (or `target_class`);
with more, one run per class, all in one model, the highest sum winning.
The data's row weights are the initial boosting weights.

**ENDER.** `pyrulearn.learners.boosting.ENDER` (Dembczyński, Kotłowski & Słowiński,
DMKD 2010; and its MLRules instance, ICML 2008) keeps a score per class
and adds rules that each vote for one class with a positive weight; the
class with the highest total wins. It starts from a default rule for one
class, with the weight minimizing the loss. Each round draws a subsample
of the rows (without replacement) and grows a rule on it greedily,
choosing conditions and class to minimize an impurity derived from the
loss, until no condition lowers it; the rule is kept only if the impurity
is negative. The rule's weight is computed on *all* rows and shrunk by
`shrinkage` -- together with the subsampling the paper's alternative to
pruning.

- **Losses** (a pluggable `BoostingLoss`): `LogisticLoss` (the
  multinomial log-likelihood; any number of classes; weight: Newton
  step), `ExponentialLoss` (AdaBoost's, two classes; weight: the exact
  minimizer, smoothed as in Slipper), `SigmoidLoss` (a bounded, non-convex
  approximation of the 0-1 loss, two classes; weight: its constant step
  `SigmoidLoss(beta=0.2)`).
- **Impurities** (`method`: a pluggable `ImpurityCriterion`, ENDER's
  counterpart of a rule-evaluation heuristic), with `g`/`h` the loss's
  first/second derivatives for a vote for the class, summed over the
  covered rows: `ConstantStep(beta=0.2)` (the default; the change of the
  loss for a step of `beta`; any loss; `beta` controls coverage --
  larger, smaller and purer rules), `Gradient()` (`g`; the most general
  rules), `GradientBoosting()` (`g / sqrt(covered weight)`),
  `Simultaneous()` (exponential loss: `-sqrt(W+) + sqrt(W-)`, the loss
  with the rule's exact weight), `Newton()` (`g / sqrt(h)`, MLRules'). A
  criterion implements `terms` (per-row quantities) and `impurity` (the
  criterion from their sums over the covered rows). Losses and criteria
  are passed as objects, like a SeCo learner's components.
- **Defaults** are the paper's constant-step logit setting (`beta = 0.2`,
  `shrinkage = 0.1`, `subsample = 0.25`, 500 rules), among its best and
  usable with any number of classes. Its best-ranked setting, CS-Exp, is
  `ENDER(loss=ExponentialLoss())`; SM-Exp adds `method=Simultaneous()`;
  CS-Sigm is `ENDER(loss=SigmoidLoss(), shrinkage=0.2, subsample=0.5)`;
  MLRules is `ENDER(method=Newton(), subsample=0.5)`.
- `early_stopping=True` (MLRules') uses the rows left out of each
  subsample as a holdout: a rule is acceptable if its error on the
  holdout rows it covers is below guessing (`1 - 1/K`), and growth stops
  once 8 of the last 10 rules weren't.

The result is a `LinearRuleModel`; repeated rules are merged by summing
their weights. The paper also covers regression (squared-error loss),
which pyrulearn doesn't.

`ENDER` grows its rules through the data representation's own primitives
(summing every open condition's impurity terms with one
`batch_cover_sums` call per step, with constraint propagation), so it
runs on every representation. `DenseENDER` is the same algorithm on a
dense float matrix: same parameters, same models, 1-8x faster on a
`HorizontalBitmapRepresentation` -- most with Newton steps on all rows (see
*Native learning algorithms* for this pairing).

```python
from pyrulearn.learners.boosting import ENDER, ConstantStep, ExponentialLoss

model = ENDER().fit(train_rep)                                   # CS-Log
few = ENDER(n_rules=3, shrinkage=1.0, subsample=1.0, loss=ExponentialLoss(),
            method=ConstantStep(beta=0.6)).fit(train_rep)
```

**Boomer.** `pyrulearn.learners.boosting.Boomer` (Rapp, Loza Mencía,
Fürnkranz, Nguyen & Hüllermeier 2020) is BOOMER's single-label case as an
`ENDER` configuration: the logistic loss with L2-regularized Newton steps
(each rule maximizes `(Σg)² / (Σh + λ)` and gets the weight
`-Σg / (Σh + λ)`, shrunk), with BOOMER's defaults -- up to 1,000 rules,
shrinkage 0.3, L2 weight 1.0, no subsampling. BOOMER itself is a
multi-label learner, which pyrulearn doesn't support yet; the original
is interfaced as `MLRLBoomer` (see *BOOMER and realkd: boosted rule
ensembles*), and the two differ in details such as feature sampling.
`DenseBoomer` is the same on `DenseENDER`: same models, faster.

**Optimal rule boosting.** `pyrulearn.learners.boosting.ORB` (Boley,
Teshuva, Le Bodic & Webb 2021; the `realkd` package) adds, in each round,
the rule that maximizes the XGBoost-style gain (Σg)² / (λ + Σh) of the
loss derivatives g and h -- found with the library's rule searches on
the `XGBGain` objective (see *What a search maximizes* under SeCo). The
search is passed as a `RuleSearch`, as in every SeCo learner: by default
`BeamSearch(beam_width=10)`, which was the most accurate on average in a
first comparison with greedy search, narrower beams and capped branch
and bound (see `ROADMAP.md`), at a fraction of a second per fit;
`BranchAndBoundSearch` finds the optimal rule (the paper's and realkd's
exhaustive search, with the paper's prefix/suffix bound), and
`HillClimbing` grows it greedily. A search's `max_conditions` caps the
rule length. Binary classification, with `realkd`'s losses as
`MarginLoss` objects -- `LogisticMarginLoss()` (default) or
`SquaredMarginLoss()` (`realkd` itself is interfaced as
`RKDRuleBoosting`); a rule's class is the sign of its weight. `DenseORB` is the same learner with two hand-written
searches (`search="exhaustive"`/`"greedy"`) on a dense matrix; `ORB` with
`BranchAndBoundSearch`/`HillClimbing` and `DenseORB` both reproduce
`realkd` exactly (`tests/test_realkd_import.py`).

```python
from pyrulearn.learners.boosting import ORB, Boomer
from pyrulearn.learners.seco import BranchAndBoundSearch

model = Boomer(n_rules=50, random_state=0).fit(train_rep)
short = ORB(n_rules=5).fit(train_rep)                     # beam search, width 10
optimal = ORB(n_rules=5, search=BranchAndBoundSearch(max_conditions=3)).fit(train_rep)
```

### Interfaced learning algorithms

An external learner is a two-way bridge. `fit(data)` first hands the
`DataRepresentation` to the tool in the format it expects
(`ExternalRuleLearner.prepare`, then `fit_external`), and then an importer
reads the fitted model, or the tool's printed output, back as rules bound to
the same `DataSpec`. This part covers both directions: first what each tool
receives, then the importers.

**Naming.** A learner that runs an external tool carries a short prefix for
the tool, so results tables show which implementation produced them: `SKL`
(scikit-learn: `SKLDecisionTree`, `SKLRandomForest`), `Witt` (wittgenstein:
`WittIREP`, `WittRIPPER`), `IMod` (imodels: `IModBayesianRuleList`,
`IModBayesianRuleSet`, `IModRuleFit`, `IModSlipper`), `Weka` (`WekaJRip`,
`WekaPART`, `WekaJ48`), `PArc` (pyarc: `PArcCBA`), `MLRL` (mlrl-boomer:
`MLRLBoomer`), `RKD` (realkd: `RKDRuleBoosting`); the reference LORD
implementation is `JavaLORD`. Native learners have plain names (`CN2`,
`Slipper`, `Boomer`). Every learner's `display_name` gives the name for a
table -- `"IMod:Slipper"`, `"CN2"`. The importers keep their names
(`JRipImporter`, ...). The names before 0.2.0 (`JRip`, `DecisionTree`,
`RIPPERk`, `LordJar`, ...) were removed in 0.9.0.

**`ExternalRuleLearner`** -- runs an external algorithm (e.g. a
scikit-learn estimator) on `data`'s data, then hands the
fitted model to an existing `ObjectRuleImporter` (declared via the
`IMPORTER` class attribute) to convert it into rules. This is
deliberately *not* a second hierarchy parallel to `RuleImporter`: a
learner reuses its importer's rule-extraction logic via composition
rather than duplicating it, and lives in the *same file* as that
importer -- `pyrulearn.interfaces.sklearn.SKLDecisionTree`/
`SKLRandomForest` sit right next to `SklearnTreeImporter`/
`RandomForestImporter`. So adding support for one more external
algorithm means adding one file (or, if its importer already exists,
one small class in it) -- never two separately registered/maintained
class hierarchies to keep in sync.

Breaks down into three named steps: `prepare` (converts a
`DataRepresentation` into the algorithm's own native input format; default
is `data.X` as-is, right for most estimators) -> `fit_external` (calls the
algorithm) -> `import_model`, bound to the *existing* `DataSpec` (no new
features, just lookups). `fit(data)` composes them automatically and is the
only method most callers need. Concrete subclasses implement `fit_external`
(required) and `prepare` (only if the native format genuinely differs from
`data.X`).

```python
from pyrulearn.interfaces.sklearn import SKLDecisionTree, SKLRandomForest

learners = [
    SKLDecisionTree(max_depth=4, random_state=0),
    SKLRandomForest(n_estimators=10, max_depth=4, random_state=0),
]
for learner in learners:
    rules = learner.fit(train_rep)   # DisjointRuleSet or RuleSet, depending on the algorithm
    print(type(rules).__name__, len(rules), "rules")
```

An algorithm can also be fit directly on its own raw, un-pre-binarized
input (numeric, already-encoded categorical, or a mix -- whatever sklearn
itself was trained on; no threshold cap chosen ahead of time), with no
`DataSpec` yet: call `fit_external` yourself, discover a `DataSpec` from the
fitted model (`ObjectRuleImporter.infer_dataspec`, see below), then
`import_model` against it, this time actually adding features as they're
found:

```python
from pyrulearn.interfaces.sklearn import SKLDecisionTree, SklearnTreeImporter

learner = SKLDecisionTree(max_depth=4, random_state=0)
model = learner.fit_external(X_raw, y, feature_names=["age", "income", "score"])

importer = SklearnTreeImporter()
ds = importer.infer_dataspec(model, feature_names=["age", "income", "score"])
rules = importer.import_model(model, ds, feature_names=["age", "income", "score"])
```

`pyrulearn.interfaces.wittgenstein.WittIREP`/`WittRIPPER` are the same
idea, wrapping `wittgenstein.IREP`/`wittgenstein.RIPPER`. `pos_class` is
required (pyrulearn never guesses which class is "positive"); `neg_class`
is optional and, if omitted, auto-resolved from `data.y` when
there's exactly one other label besides `pos_class` -- with more than
one other label present (real multi-class, which wittgenstein itself
doesn't support natively), `default_prediction` is left `None` unless
`neg_class` is given explicitly:

```python
from pyrulearn.interfaces.wittgenstein import WittIREP, WittRIPPER

learners = [
    WittIREP(pos_class="pos"),
    WittRIPPER(pos_class="pos", k=2, random_state=0),
]
for learner in learners:
    rules = learner.fit(train_rep)  # RuleSet, default_prediction wired up automatically
```

#### What each tool receives

| Package | Learner classes | Rules imported | What is handed over | Feature names |
|---|---|---|---|---|
| scikit-learn | `SKLDecisionTree`, `SKLRandomForest` | directly | `data.X` as-is: the Boolean feature matrix, in memory | not needed to fit; the importer binds the thresholds back to the `DataSpec` |
| wittgenstein | `WittIREP`, `WittRIPPER` | directly | `data.X` and the labels, in memory; binary only, so `pos_class` is required | the `DataSpec`'s own feature names |
| imodels | `IModBayesianRuleList`, `IModBayesianRuleSet`, `IModRuleFit` | directly | `data.X` as a 0/1 integer matrix, in memory (a Boolean dtype breaks BRL; RuleFit gets floats); BRS and RuleFit refuse anything that isn't 0/1; binary only | the `DataSpec`'s own feature names |
| Weka | `WekaJRip`, `WekaPART`, `WekaJ48` | by parsing text | an ARFF file in a temporary directory: every feature a `{False,True}` nominal attribute, plus a `class` column; run as `java -cp weka.jar <classifier> -t train.arff -no-cv` | placeholders `f0..fN` |
| LORD | `JavaLORD` | by parsing text | a CSV in a temporary directory (`data_train_01.csv`, plus the same rows as the test file LORD insists on): 0/1 columns with the class as the last column; run as `java -cp <classpath> run.LordRun` | placeholders `f0..fN` |
| pyarc | `PArcCBA` | directly | transactions (via `TransactionDB.from_DataFrame`) holding only each row's True features plus the class; False cells are dropped, which matches this library's own item semantics | placeholders `f0..fN` |

Rules are imported either directly from the fitted model object (an
`ObjectRuleImporter`) or by parsing text the tool printed or wrote (a
`StringRuleImporter`); see *Importer base classes* and *Text formats*, below.

Tools that can't cope with a `DataSpec`'s feature names (`age>=30`,
`color=red`, ...) get the placeholders instead, and the matching importer is
then created with `placeholder_features=True` so that the rules bind back to
the real `DataSpec` by column position. Files are written to temporary
directories that are removed after the run. The Weka and LORD runners can also
be used on their own (`run_weka`, `run_lord`).

#### Importer base classes

`RuleImporter` (in `pyrulearn.interfaces`) is the shared base for
everything that brings rules into pyrulearn from an external source --
not instantiated directly. Two intermediate layers derive from it,
matching the two shapes a source actually comes in, and every concrete
importer (of either shape) registers itself in one shared registry:

```python
class RuleImporter(ABC):
    SOURCE: str  # e.g. "sklearn.tree.DecisionTreeClassifier"
    def _stamp_rule_provenance(self, rules, **params): ...   # wraps as SingleRule, sets .provenance
    def _stamp_provenance(self, model, **params): ...         # sets the returned model's own Provenance

class ObjectRuleImporter(RuleImporter):   # an already-fitted live model object
    @abstractmethod
    def import_model(self, model, dataspec: DataSpec) -> RuleModel | Sequence[RuleModel]: ...

class StringRuleImporter(RuleImporter):   # a string/serialized rule format
    @abstractmethod
    def parse(self, source) -> RuleModel: ...

register_importer(name: str, cls)         # e.g. register_importer("sklearn_tree", SklearnTreeImporter)
get_importer(name: str) -> Type[RuleImporter]
```

**`ObjectRuleImporter`** -- `pyrulearn.interfaces.sklearn` has two
concrete ones so far, both grouped in the same module since they share
the same per-tree leaf-extraction logic (`_rules_from_tree`) -- concrete
importers are grouped by shared implementation, not rigidly one file
per library or one per algorithm. `dataspec` is the caller's *own*
`DataSpec` -- rules bind to it directly rather than a throwaway shadow
copy built from just its feature names, so imported rules share typed
display/constraints and object identity with the rest of the pipeline
(no separate object to `remap` later, if you're already using one
shared `DataSpec`).

#### Text formats: `StringRuleImporter`

**`StringRuleImporter`** -- for a string/serialized rule format (RIPPER/JRip,
CN2, FOIL, CBA/CMAR-style association rules, decision-list text dumps,
...); named after the Python-level shape of the input, same as
`ObjectRuleImporter` (`object` vs. `str`), so a hypothetical future
importer of some other shape (e.g. an image) wouldn't be ambiguously
grouped in with either. `pyrulearn.interfaces.base.PatternStringImporter`
is the reference implementation -- one rule per line, a whitespace-/
comma-separated pattern of 0/1/`-` tokens (`Rule.from_pattern_string`),
where `1` at position i means feature i is a condition of the rule and
`0`/`-` mean it is not (there is no "required-False" token any more --
negation is its own feature), optionally followed by `=> target`:

```python
from pyrulearn import PatternStringImporter

importer = PatternStringImporter(feature_names=["f0", "f1", "f2"])
rules = importer.parse("1 - - => pos\n- 1 1 => neg\n")
```

#### scikit-learn: decision trees and random forests

`SklearnTreeImporter` extracts one `DecisionTreeClassifier`'s leaves as
a `DisjointRuleSet` (a tree's leaves are pairwise disjoint by
construction):

```python
from pyrulearn.interfaces.sklearn import SklearnTreeImporter, from_sklearn_tree

rules = SklearnTreeImporter().import_model(fitted_tree, my_dataspec)
# or the convenience wrapper (also accepts feature_names= if you don't
# have a DataSpec yet):
rules = from_sklearn_tree(fitted_tree, dataspec=my_dataspec)
```

Both `SklearnTreeImporter` and `RandomForestImporter` also implement
`infer_dataspec(model, feature_names)` (the generic `ObjectRuleImporter`
hook -- see `pyrulearn.interfaces.base`): fit directly on *raw* numeric
data and this discovers a `DataSpec` from the thresholds actually used,
rather than requiring them fixed in advance by a separate discretizer.
Pass `feature_names=` to `import_model` too when using a discovered
`DataSpec` this way -- see the raw-input example under *Learning
algorithms*, above.

`RandomForestImporter` extracts every tree's leaves from a
`RandomForestClassifier` and combines them all into one flat `RuleSet`
-- not a sequence of one `DisjointRuleSet` per tree. Each tree is
individually exhaustive+disjoint, so a k-tree forest always has exactly
k covering rules per example; `pyrulearn.combiners` is what turns that
into forest-style prediction (see *Conflict resolution*, above) -- either
`CountVoteCombiner` for hard voting (or `HeuristicVoteCombiner` for a
heuristic-weighted one), or `MacroVoteCombiner` for soft voting straight
from each leaf's own measured per-class distribution (requires
`import_model(..., data=...)` so every leaf has stats --
`DistributionCombiner` raises otherwise), which is the one that
actually matches sklearn's own averaging mechanism. No `default_rule`
is needed: since
every tree always fires, the combined set is always exhaustive too.
(Note: sklearn
encodes labels once at the *forest* level and fits each tree on those
encoded values, so `rf.estimators_[i].classes_` is just `[0., 1., ...]`,
not the original labels -- `RandomForestImporter` passes the forest's
own `classes_` through explicitly to get real targets back.)

```python
from pyrulearn.interfaces.sklearn import RandomForestImporter, from_random_forest
from pyrulearn import HorizontalBitmapRepresentation, MacroVoteCombiner

rules = RandomForestImporter().import_model(fitted_forest, my_dataspec)
my_rep = HorizontalBitmapRepresentation(my_dataspec, X, y)
preds = rules.predict(my_rep, combiner=MacroVoteCombiner())  # soft voting, ~matches sklearn
```

This module also has the reverse direction: `RuleSetClassifier` wraps any
`pyrulearn.models` rule model as an `sklearn.base.BaseEstimator`/
`ClassifierMixin`, so it isn't an importer at all and doesn't fit the table
above. Two modes -- a fixed `rules=` (annotate given data, e.g. to evaluate
externally-mined rules via `cross_val_score`), or a `learner=` callback
mining rules fresh on every `fit` call, which plugs any rule-learning
algorithm into `cross_val_score`/`GridSearchCV`/sklearn pipelines.

#### wittgenstein: IREP and RIPPER

`IREPImporter`/`RIPPERImporter` (`pyrulearn.interfaces.wittgenstein`,
an optional dependency -- only importing that module pulls in
`wittgenstein`) extract a `RuleSet` from a fitted `wittgenstein.IREP`/
`wittgenstein.RIPPER` model. Both algorithms share the same `Ruleset`/
`Rule`/`Cond` structure, hence one shared conversion helper and one
file for both, same grouping-by-shared-implementation rule as
`pyrulearn.interfaces.sklearn`. Every extracted rule predicts the model's single
`pos_class` -- IREP/RIPPER are fundamentally binary, confirmed directly
against the library rather than assumed: no internal one-vs-rest for
more classes; `fit` raises without an explicit `pos_class` for anything
beyond two/boolean-like labels, but given an explicit one it silently
proceeds on *any* number of classes instead, collapsing everything that
isn't `pos_class` into one undifferentiated negative with no warning --
`WittIREP`/`WittRIPPER` (below) refuse this outright rather than let it happen
unnoticed. So rules never actually
conflict when several cover the same example (any `RuleCombiner`
agrees on the answer). What *does* need resolving is only the
*fallback* for what's uncovered, and the fitted model alone doesn't
retain what label that should be (its own `.predict()` only returns
booleans, "matched `pos_class`" or not) -- used directly,
`default_prediction` is therefore left `None`:

```python
from pyrulearn.interfaces.wittgenstein import RIPPERImporter
import wittgenstein as lw

model = lw.RIPPER(k=2, random_state=0)
model.fit(my_rep.X, y=my_rep.y, pos_class="pos", feature_names=my_rep.spec.feature_names)
rules = RIPPERImporter().import_model(model, my_dataspec)  # rules.default_rule is None
```

Use `WittIREP`/`WittRIPPER` (below) instead of the bare importer to get that
fallback wired up automatically from training labels.

Fit on **raw** numeric/categorical columns and there are two ways to
still get a correct conversion, the same two as for
`pyrulearn.interfaces.sklearn`:

- **Binarize first**, via `pyrulearn.data.io.
  build_dataspec`/`binarize` (the same pipeline the
  `RandomForestImporter` demo above uses), so wittgenstein only ever
  sees already-Boolean columns named after real `DataSpec` features
  (`age>=20`, `color=red`, ...); every `Cond` it then produces is
  already a plain Boolean literal on a feature already in the target
  `DataSpec`, complete with `ThresholdChain`/`ExactlyOne` provenance:

  ```python
  from pyrulearn.data.io import build_dataspec, binarize
  from pyrulearn.data import HorizontalBitmapRepresentation

  ds = build_dataspec(df, target="label", arff_types={...}).build()
  X = binarize(ds, df)
  rep = HorizontalBitmapRepresentation(ds, X, df["label"].to_numpy())
  rules = WittRIPPER(pos_class="pos").fit(rep)
  ```

- **Fit on raw columns** directly and let
  `IREPImporter`/`RIPPERImporter.infer_dataspec` parse wittgenstein's
  *own* discretization/nominal output back into typed attributes: a
  bin-range `Cond` like `num_feat=0.53 - 0.84` becomes two `<=`-family
  threshold literals (`num_feat<=0.53` false, `num_feat<=0.84` true),
  an unbounded one like `num_feat=>1.32` becomes one, and a bare
  category like `color=red` becomes a nominal-equality literal --
  decided per attribute from the `Cond.val`s actually seen for it, not
  globally. Both bounds go through the `<=` family, not `>=`:
  wittgenstein's own bins are right-closed/left-open, `(lo, hi]` --
  confirmed directly against `wittgenstein.discretize.BinTransformer`
  (its own docstring, its `closed='right'` interval construction, and
  empirically, a value exactly at a bin's lower edge lands in the
  *previous* bin) rather than assumed:

  ```python
  from pyrulearn.interfaces.wittgenstein import WittRIPPER, RIPPERImporter

  learner = WittRIPPER(pos_class="pos", k=2, random_state=0)
  model = learner.fit_external(df.to_numpy(), y, feature_names=["age", "color"])

  importer = RIPPERImporter()
  ds = importer.infer_dataspec(model, feature_names=["age", "color"])
  rules = importer.import_model(model, ds, feature_names=["age", "color"])
  ```

#### imodels: Bayesian rule lists, rule sets, RuleFit and Slipper

`BayesianRuleListImporter` (`pyrulearn.interfaces.imodels`, an
optional dependency on `imodels`) extracts Letham et al.'s Bayesian
Rule Lists from a fitted `imodels.BayesianRuleListClassifier` — and is
the one importer so far producing a **`RuleList`**, not a `RuleSet`. A
Bayesian Rule List is genuinely ordered (its own `__str__` prints
`IF ... ELSE IF ... ELSE ...`), each rule implicitly conditioned on
every earlier one *not* firing, so first-match-wins is the real
semantics and the trailing `ELSE` becomes the list's `default_prediction`.

No discretization wrinkle here, unlike wittgenstein: `imodels`' own
`fit` *requires* strictly 0/1 input (it raises `"All numeric features
must be discretized prior to fitting!"` otherwise) — exactly what a
`HorizontalBitmapRepresentation` already holds — so `infer_dataspec` is
inherited unchanged and both approaches collapse to the same thing.
Confirmed directly (not just from the error message): a categorical
column has to go through `build_dataspec`/`binarize`'s **one-hot**
nominal encoding first, same as any numeric attribute — a raw string
column fails even earlier (`could not convert string to float`), and
an *ordinal*-encoded one (`0`/`1`/`2`, no one-hot) hits the exact same
"must be discretized" error, since a 3-valued int column still isn't
0/1. Its `fit` also refuses >2 classes properly (`"Only binary
classification is supported at this time!"`, with no built-in
one-vs-rest fallback), so no extra guard is needed on pyrulearn's side
— genuine multi-class support would mean pyrulearn adding its own
cascade on top, same open question wittgenstein raises.

One thing to know when reading imported rules: each rule carries a
posterior *probability* rather than a label, so a rule with
`theta=0.11` is a confident **negative** rule — `target` is read off as
whichever class `theta` favors (`> 0.5` -> positive class, else
negative). The raw `theta`/confidence-interval numbers aren't kept on
the rule; `pyrulearn.combiners`' `HeuristicCombiner`s score each rule
from its own measured `stats()` instead (pass `data=` to
`import_model`/`fit` to get them). Antecedents come from FP-growth
itemsets, so rules only ever contain **positive** literals.

```python
from pyrulearn.interfaces.imodels import IModBayesianRuleList

rules = IModBayesianRuleList(random_state=0).fit(train_rep)   # a RuleList
print(rules.to_string("logic"))
# if   f0 ∧ f1 → neg
# elif f0 ∧ f2 → pos
# else → neg
```

`BayesianRuleSetImporter` (same module) extracts the "BOA"/Bayesian
Or-of-And algorithm from a fitted `imodels.BayesianRuleSetClassifier`
into a plain **`RuleSet`**, not a `RuleList`: it fits a genuinely
*unordered* pattern set via simulated annealing and predicts OR-of-AND
(any one rule firing predicts `classes_[1]`, regardless of which), so
rules never actually disagree in target when several cover the same
example. Same Boolean-input requirement as BRL, but `imodels` doesn't
raise a clear error for either that or a non-binary target here (an
opaque downstream `AssertionError`/`ValueError` instead), so
`IModBayesianRuleSet.fit_external` checks both itself first. Rule items are
real feature names, optionally `_neg`-suffixed for a negative literal
(`model.rules_`, unlike BRL, can and does contain both), looked up via
`dataspec.feature_index` directly (no `X_0`-placeholder indirection).
There's no per-rule posterior the way BRL's `theta` is (the annealing
search optimizes the whole rule set jointly), so imported rules carry
no per-rule confidence score — moot anyway here, since firing rules
never disagree. **Two genuine bugs found in `imodels` itself** while testing
this importer (both documented in the module's docstring, not fixable
from this side): its own `fit` mixes seeded `np.random` with
*unseeded* stdlib `random.sample` (worked around in
`IModBayesianRuleSet.fit_external` by seeding that too), and its
simulated-annealing "clean" move has a genuine off-by-value-vs-index
bug that can raise a bare `list.remove(x): x not in list`.

```python
from pyrulearn.interfaces.imodels import IModBayesianRuleSet

rules = IModBayesianRuleSet(random_state=0, maxlen=3).fit(train_rep)   # a RuleSet
```

`RuleFitImporter` reads an `imodels.RuleFitClassifier` as a
`LinearRuleModel`, the model the native `learners.rulefit.RuleFit`
returns, and `IModRuleFit` is the learner (named apart from the native
one). The conversion is exact on 0/1 input: tree-path tests become
features or their negation features, imodels' scaled linear terms become
length-1 rules (with any constant part folded into the intercept), and
the imported model's scores equal imodels' own decision function.

**imodels predicts with the wrong threshold.** The regression is logistic,
so the positive class should win where its output `f` is above 0; but
`predict_proba` takes the softmax of `[1 - f, f]`, so `predict` needs `f >
0.5`. By default the import reproduces imodels' `predict`, and shows the
offset as a rule of its own; `imodels_threshold=False` imports the
logistic model as fitted:

```prolog
% conflict resolution: sum of rule weights per class, highest wins

% class: bad
0.50::bad(X) :- true.

% class: good
0.00::good(X) :- true.
6.53::good(X) :- f0(X), f1(X).
-2.19::good(X) :- \+f0(X), \+f2(X).
...
```

`rulefit_candidates(data)` runs only RuleFit's candidate generation
(imodels' boosted trees, same parameters and defaults) and returns the
rules as a pool, so `RuleFit(rules=rulefit_candidates(data))` is RuleFit's
two steps with pyrulearn's own fit. `examples/demo_rulefit_comparison.py`
compares the two implementations step by step.

```python
from pyrulearn.interfaces.imodels import IModRuleFit, rulefit_candidates
from pyrulearn.learners.rulefit import RuleFit

imported = IModRuleFit(random_state=0).fit(train_rep)         # a LinearRuleModel
native = RuleFit(rules=rulefit_candidates(train_rep, random_state=0), cv=5).fit(train_rep)
```

`SlipperImporter` reads an `imodels.SlipperClassifier` as a
`LinearRuleModel`, reproducing its predictions exactly, and
`IModSlipper` is the learner. Despite its name, imodels' Slipper is
scikit-learn's AdaBoost (SAMME) over imodels' own rule learner, not Cohen
& Singer's confidence-rated boosting (that is the native
`learners.boosting.Slipper`): each rule votes with its AdaBoost weight α
for the positive class where it fires and for the negative class where it
doesn't, which is a linear model with the rules' α as weights and the
intercept −Σα/2. Reading imodels' source turned up several problems:
its rules are never pruned (the pruning step compares a rule's objective
with itself), its "default rule" is a random conjunction of conditions
like `x > min(x)` (on 0/1 features an arbitrary ordinary rule), the
boosting weights are indexed with row positions of the growing subset,
and AdaBoost typically stops after one to three rules. **With labels
other than 0/1, its `predict` always returns the first class**:
`BoostedRulesClassifier.fit` restores the original labels in `classes_`,
which scikit-learn's AdaBoost compares its rules' 0/1 predictions with.
`IModSlipper` fits on 0/1 labels to avoid that, and the import
reproduces the intended model.

#### weka: JRip, PART and J48

`pyrulearn.interfaces.weka.JRipImporter` is a real, non-reference
example of the shape: it parses `weka.classifiers.rules.JRip`'s printed
rule-set text (Weka is not a Python library, so there's no live model
object to hand an `ObjectRuleImporter` -- this is genuinely the
`StringRuleImporter` case, not just a stand-in for one). Verified
against a real `weka.jar` run, not assumed:

```python
from pyrulearn.interfaces.weka import JRipImporter

text = """
(color = red) and (age >= 41) => label=pos (81.0/2.0)
(income <= 19992) => label=pos (32.0/1.0)
 => label=neg (287.0/15.0)
"""
rules = JRipImporter().parse(text)   # a RuleList, not a RuleSet
```

Produces a **`RuleList`**, unlike wittgenstein's IREP/RIPPER importers
(`RuleSet`): JRip is genuinely multi-class (classic RIPPER sequential
covering -- one class's rules at a time, in printed order, falling back
to a default/majority class), so two rules with different targets can
both cover the same new example, and only trying them in the printed
order reproduces JRip's own predictions correctly -- confirmed by
`.predict()` on the underlying data matching JRip's rule logic exactly
*and* Weka's own reported training accuracy. No `dataspec=` needed up
front: `parse` discovers and caches a typed `DataSpec` from the rule
text itself, with numeric attributes getting a proper `NumericGroup`
on *both* sides: `>=`/`<` conditions go through `add_numeric`'s usual
`ge_thresholds=` family, `<=`/`>` conditions through its `le_thresholds=`
family (see `DataSpecBuilder.add_numeric`'s own docstring for why
that's a second, genuinely independent family rather than a
reinterpretation of the first) -- so both binarize straight from a raw
numeric column with no manual precomputation needed either way. Pass
`dataspec=` at construction to reuse an already-built one instead, e.g.
across a train/test pair of Weka runs.

`PARTImporter` (same module) parses `weka.classifiers.rules.PART`'s
decision-list text -- Frank & Witten's "partial decision trees" learner,
separate-and-conquer over a pruned partial C4.5 tree each round. Same
`RuleList` result and `dataspec=`/auto-discovery convention as
`JRipImporter`, but a genuinely different text format underneath (also
verified against a real `weka.jar` run):

```python
from pyrulearn.interfaces.weka import PARTImporter

text = """
income > 19992 AND
color = green: neg (125.0/8.0)

age > 39: pos (100.0/3.0)

: pos (13.0)
"""
rules = PARTImporter().parse(text)   # a RuleList
```

No `=>`, no parentheses, one condition per line ending in a trailing
`` AND`` (except a rule's last line, which ends in `` : target
(stats)`` instead), and rules are blank-line-delimited blocks rather
than one line each -- so `PARTImporter` parses at the block level
(`_collect_part_conditions`), unrelated to JRip's line regex, but hands
off to the *same* shared condition-parsing/`DataSpec`-building code once
it has `(conditions, target, covered, errors)` tuples, since that part
only cares about the six recognized operator strings, not which
classifier produced them. Two format quirks worth knowing: PART's
numeric splits are classic C4.5-style `<=`/`>` (`>=`/`<` never appear),
and its trailing stats can omit the error count entirely when it's zero
(`(13.0)`, not JRip's always-both-numbers `(13.0/0.0)`) -- both handled
already.

`J48Importer` (same module) parses `weka.classifiers.trees.J48` --
Weka's C4.5 reimplementation -- and produces a **`DisjointRuleSet`**,
not a `RuleList`: a decision tree's leaves already partition the
feature space disjointly and exhaustively by construction, the same
reasoning `pyrulearn.interfaces.sklearn.SklearnTreeImporter` uses
for its own trees, so there's no rule-order/default-rule question here
at all -- confirmed, not just assumed (`rules.is_disjoint(rep)` and
`rules.is_exhaustive(rep)` both hold on real data):

```python
from pyrulearn.interfaces.weka import J48Importer

text = """J48 pruned tree
------------------

income <= 19992: pos (40.0/1.0)
income > 19992
|   color = red
|   |   age <= 40: neg (45.0/3.0)
|   |   age > 40: pos (73.0/2.0)
|   color = green: neg (125.0/8.0)
|   color = blue: neg (117.0/4.0)
"""
rules = J48Importer().parse(text)   # a DisjointRuleSet
```

The text is an indented tree, not a flat list: each `` |   `` group is
one nesting level, a line ending in `` : target (stats)`` is a leaf
(its rule is the conjunction of every still-open ancestor condition
plus its own), and a line without that suffix is an internal split
whose condition applies to everything nested under it -- walked with a
depth-indexed stack (`_collect_j48_conditions`), pushing on a deeper
split and truncating back to the current depth before every line (the
backtrack, whenever a sibling appears at the same or a shallower
depth). Because C4.5 (unlike CART) allows a nominal split to fan out
into more than two branches, `color` above genuinely has three
siblings under one parent -- handled without any special-casing, since
each sibling is just another line at that depth; verified against a
deliberately deep, 33-leaf/depth-5 tree too, not only the shallow
example above. One thing this importer needs that JRip/PART's don't:
scanning a *whole* raw console dump line-by-line for tree-shaped
content isn't safe here, since the confusion matrix's own column
layout can produce a line reading as both a `` |   ``-indented node
*and* a valid `` attr = value`` condition (e.g. `` 110  15 |   a =
pos``) -- `parse` instead anchors explicitly on the `` J48 ... tree``
header and its `` ---- `` divider before reading the tree body.

#### LORD: the Java reference implementation

`pyrulearn.interfaces.lord` runs the Java implementation of LORD (Huynh,
Fürnkranz & Beck, 2023; [vqphuynh/LORD](https://github.com/vqphuynh/LORD)) as
a subprocess. `JavaLORD` is the `fit()` learner (`variant=` selects
`run.LordRun`, `LordStarRun` or `LordLoopRun`; `metric=`/`metric_arg=` are
LORD's `-mt`/`-ma`), `LORDImporter` parses the `eg_output.txt` it writes, and
`run_lord` is the function that drives the jar. `classpath=` (else
`$LORD_CLASSPATH`, or `$LORD_JAR` for a fat jar) and `java=` (else
`$LORD_JAVA`) locate the build.

The result is a `FlatRuleSet`: LORD predicts with the single covering rule
with the highest metric value, so pass the same metric to
`HeuristicMaxCombiner` if you want an identical ranking (its default is
`Laplace`). LORD does not print its default class, so `fit` sets the
training-majority class. A condition `(f3=0)` imports as a positive literal on
the paired negation feature of `f3`, so the bound `DataSpec` must have
negation features (`DataSpecBuilder`'s default). LORD is natively multi-class.

#### pyarc: CBA

`pyrulearn.interfaces.pyarc.PArcCBA` fits the external `pyarc` package's CBA
and returns a `DecisionList`; `PyarcCBAImporter` reads an already-fitted
`pyarc.CBA`. It is the reference that the native
`pyrulearn.learners.associative.CBA` is cross-checked against, and a faster
drop-in (the C library `fim` that `pyarc` mines with is roughly 10x faster than
this package's pure-Python miner). Its defaults (`min_support=0.01`,
`min_confidence=0.5`, `max_len=4`) mirror the native `CBA`'s, and `max_len`
means the same in both (`pyarc`'s own `maxlen` also counts the class item,
which `PArcCBA` adjusts for); `algorithm=` picks `pyarc`'s `"m1"` (default)
or `"m2"` classifier builder. `pyarc` needs Borgelt's `pyfim` C extension,
which has no Windows wheels and must be built separately; importing the module
needs neither.

#### BOOMER and realkd: boosted rule ensembles

`pyrulearn.interfaces.boomer.MLRLBoomer` fits BOOMER (Rapp et al. 2020;
the `mlrl-boomer` package), gradient-boosted multi-output rules, for binary
classification, and `BoomerImporter` reads a fitted
`mlrl.boosting.BoomerClassifier` with its own rule visitor into a
`LinearRuleModel` whose scores equal BOOMER's `decision_function`. On a 0/1
feature BOOMER's conditions `x <= t` / `x > t` are the feature's negation
feature / the feature. A rule with a positive score votes for the positive
class, one with a negative score for the other class with the absolute
score. BOOMER's single-output case -- logistic loss with L2-regularized
Newton steps -- is natively `learners.boosting.Boomer`, `ENDER` with
BOOMER's defaults; on binary data it predicts like `MLRLBoomer`, and its
dense version `DenseBoomer` runs about ten times faster, though BOOMER's
feature sampling gives it somewhat different rules. BOOMER's multi-label learning isn't supported
yet (see *Not yet implemented*).

`pyrulearn.interfaces.realkd.RKDRuleBoosting` fits Mario Boley's
`realkd` rule boosting, the reference implementation of optimal rule
boosting (Boley et al. 2021), and `RealkdImporter` reads its rule ensemble.
`realkd` is given the data's positive features as columns `c0, c1, ...`;
its propositions `c<=0` / `c>=1` become the negation feature / the
feature. The native `learners.boosting.ORB` and `DenseORB` reproduce it
exactly -- the same rules and weights with greedy or exhaustive search,
with or without an intercept (see `tests/test_realkd_import.py`) -- and
`DenseORB` runs several times faster.

```python
from pyrulearn.interfaces.boomer import MLRLBoomer
from pyrulearn.interfaces.realkd import RKDRuleBoosting
from pyrulearn.learners.boosting import ORB
from pyrulearn.learners.seco import BranchAndBoundSearch

boomer = MLRLBoomer(max_rules=50, random_state=0).fit(train_rep)
native_boomer = Boomer(n_rules=50).fit(train_rep)      # from pyrulearn.learners.boosting
reference = RKDRuleBoosting(n_rules=10, search="exhaustive").fit(train_rep)
native = ORB(n_rules=10, search=BranchAndBoundSearch()).fit(train_rep)   # the same model
```

#### Provenance of imported rules

Either shape's `import_model`/`parse` should end by calling
`self._stamp_rule_provenance(rules, **params)` and
`self._stamp_provenance(model, **params)`, which give every produced
rule (and the model itself) their own `Provenance(source=..., learner=...,
params={...})` -- how you tell which base learner produced which rule
once you're comparing many imported models from many sources at once.

### Multiclass classification

Multiclass support is one `fit(data, model=...)` switcher
(`pyrulearn.learners.DecomposingLearner`) shared by every learner family
-- native (the SeCo family, `PyLORD`) and external alike
(`RelabelingExternalLearner` -- sklearn's `SKLDecisionTree`/`SKLRandomForest`;
`_WittgensteinLearner` -- wittgenstein's `WittIREP`/`WittRIPPER`). `fit(data,
model=ConceptSet | ConceptCascade | PairwiseModel)` decomposes into
per-class binary sub-fits, each through the learner's own `_fit_binary`.
`pyrulearn.learners.multiclass` wraps these three as thin sugar over the same
calls, for the older call style or to bundle a `random_state`/non-default
`combiner`:

- `model=ConceptSet` (`OneVsRest(base_learner)`) — one `ConceptModel`
  per class, pooled into a `ConceptSet`; `default_prediction =
  MajorityClass(data)` (fires only for a row no class's rules cover),
  `HeuristicMaxCombiner()`. Reassign either on the result.
- `model=ConceptCascade, order="least_frequent"`
  (`OrderedOneVsRest(base_learner, order=...)`) — peel the classes off
  one at a time (`least_frequent` / `most_frequent` / `"random"` / an
  explicit sequence): model c1 vs. the rest, then c2 vs. `{c3..cn}` on
  the rows not yet peeled (`DataRepresentation.select_rows`), … the
  last class becomes the cascade's catch-all `default_prediction`.
  `least_frequent` (default) leaves the majority class as the catch-all
  and tends to give the most compact list.
- `model=PairwiseModel, positive="smaller"`
  (`Pairwise(base_learner, positive=...)`) — one binary model per
  unordered class pair ("round robin"; Fürnkranz, JMLR 2002), each
  trained on just that pair's rows, `positive` picking the pair's
  positive label: `"smaller"` (default) / `"larger"` / `"random"` /a
  callable `f(a, b) -> pos`, or `"both"` (double round robin — one
  model per *ordered* pair, `2·C(n,2)` models, so an asymmetric learner
  gets a member for every class). At predict time each member votes and
  a `PairwiseCombiner` (`pyrulearn.models`) turns the votes into a
  label:
  - `combiner=MajorityVote()` (default) — one hard vote per member;
    `tie_break` `"direct"` (default: the tied labels' own duel, then
    training frequency, then label order -- never member order),
    `"prior"`, `"first"` or a callable. Training frequencies are the
    model's `label_priors`, else read from its rules' stats.
  - `combiner=WeightedVote(heuristic=Laplace())` —
    the deciding rule's weight `p_ij` (the heuristic on its frozen
    training stats, 0.5 if it has none) goes to the predicted label,
    `1 - p_ij` to the other (clamped to `[0, 1]`). The deciding rule is
    the best-scoring covering rule of the predicted label, or the
    sub-model's default rule when no rule covers the row (pulled per row
    via `covered_by`). Swap it onto an already-fitted model
    (`pm.combiner = WeightedVote()`) --
    no retraining, just re-predict. `WeightedVote` is a real gain when
    per-rule reliability varies (`PyLORD`'s m-estimate scoring;
    imprecise base learners on many-class sets) but roughly neutral
    otherwise — `MajorityVote` stays the default.
  - `combiner=AccuracyWeightedVote()` — same
    `p_ij` / `1 - p_ij` split, but `p_ij` is the *member's* accuracy on
    its own two-class sub-problem (one scalar per member, constant over
    rows, recorded once at fit time as `PairwiseModel.member_weights`
    by `DecomposingLearner._fit_pairwise`), so a cleaner pair pulls
    harder -- also a no-retrain combiner swap.

  `MajorityVote.scores()` / `WeightedVote.scores()` return a per-label
  score vector, so the same aggregation feeds a future label-ranking
  layer (argsort instead of argmax).

The `SeCo` family reaches for this itself: `fit(data)` with no
`target_class` and no `model=` builds each learner's own multi-class
default (`_MULTICLASS_DEFAULT`) — `ConceptSet` (one-vs-rest) for
`CN2`/`PFoil`/`PFossil`/`Pypper`, so `CN2().fit(iris_rep)` just works;
`DecisionList` for `AQR` (one seed-covering loop over all classes at
once, each rule seeded on a random uncovered example and headed with
its own label — AQ's multi-class covering, the same per-example seeding
`PyLORD` does; rules stay in learn order and the first match wins). Pass `model=ConceptCascade`/`PairwiseModel` explicitly
for the other decompositions on any of them.

## Experiments

`pyrulearn.experiments` is the infrastructure behind the comparisons in
`demos/`: a catalog of benchmark datasets, a cross-validation runner,
ranking statistics and Markdown report helpers. They are separate building
blocks -- a comparison picks its learners and datasets, calls `run_cv`, and
decides which statistics and tables to report. Needs the `experiments`
extra.

```python
from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.report import render_results_table
from pyrulearn.experiments.runner import run_cv
from pyrulearn.experiments.stats import critical_difference_diagram, mean_rank
from pyrulearn.learners.seco import CN2, Pypper

datasets = Catalog.default().parse("binary,small,5", random_state=0)
results = run_cv([CN2(), Pypper()], datasets, n_folds=10, fit_timeout=300, cache_dir="_cache")
print(mean_rank(results, "accuracy"))
print(render_results_table(results, ["accuracy", "n_conditions"],
                           group_by=["dataset", "learner"], include_overall=False))
critical_difference_diagram(results, "accuracy")
```

### Benchmark datasets

`pyrulearn.experiments.catalog` is a catalog of 100 classification datasets on
OpenML (plus a few pointer-only entries): metadata and download pointers
only, no data is bundled. Each entry names
its OpenML id (plus UCI, Kaggle or original-source links where known), and
`entry.load()` downloads it once through scikit-learn's OpenML cache
(`~/scikit_learn_data`, or `$PYRULEARN_DATA_HOME`) and returns the raw data
as a pandas DataFrame, with real missing values and no imputation:

```python
from pyrulearn.experiments.catalog import Catalog

cat = Catalog.default()
for entry in cat.select(task="binary", size=["small", "medium"], attributes="categorical"):
    df, target = entry.load()
cat.select(task="binary", size="medium", n=3, random_state=0)   # three of them, at random
cat.parse("binary,medium,3")         # the same, from a compact string (e.g. a CLI option)
print(cat.summary(task="binary"))    # browse: a table of all binary datasets
cat.select(names=["vote", "mushroom", "adult"])   # then pick some by name (kept in this order)
```

Three categories are computed from each entry's statistics:

| category | values |
|---|---|
| `task` | `binary` (2 classes), `multiclass` (3 or more) |
| `size` | `small` (< 1,000 rows), `medium` (1,000-10,000), `large` (> 10,000) |
| `attributes` | `categorical` (all attributes nominal), `numeric` (all numeric), `mixed` |

`task` x `size` for the 122 loadable entries, for gauging how big a
`select(...)`/`.parse(...)` run is before starting it (a demo's default
should stay small/medium -- see *Try it* below; `size="large"` climbs
past 10,000 rows up to 581,012, so an `all` run is an overnight-on-a-
strong-machine job, not a quick one):

| | binary | multiclass | total |
|---|--:|--:|--:|
| `small` (< 1,000 rows) | 25 | 17 | 42 |
| `medium` (1,000-10,000) | 40 | 22 | 62 |
| `large` (> 10,000) | 11 | 7 | 18 |
| **total** | **76** | **46** | **122** |

`tags` record where an entry comes from: `cc18` (the OpenML-CC18 benchmark
suite, minus its 13 image/signal/text datasets with more than 200
attributes), `lord` (the datasets of the LORD evaluation), `xai` (classic
explainable-AI datasets: adult, COMPAS, German credit, HELOC, ...) and
`classic` (traditional rule-learning benchmarks: vote, mushroom, soybean,
the MONK's problems, promoters, kropt, ...), `deep` (the datasets of the
deep rule learning evaluations, Beck & Fürnkranz 2021 and Seip et al.: 20
artificial truth tables of ten Boolean features labeled by a random deep
NNF, the MONK's problems, tic-tac-toe, vote, kr-vs-kp, mushroom, and
binary versions of car and connect-4) and `artificial` (generated data).
Entries not on OpenML either read a CSV file (`csv_url`, downloaded once
into the same data home; the `deep` datasets from Beck's repository,
pinned to one commit) or are pointers only (e.g. SUSY, PAMAP2;
`entry.loadable` is False).

Entries also carry curated loading fixes: attribute names restored where
the OpenML upload anonymized them (`V1`, `x1`, ...), identifier and
leakage columns dropped (e.g. Titanic's `boat`), category codes stored as
numbers declared nominal, and values that encode "not measured" (Pima
diabetes's zeros) turned into missing values. `tools/build_catalog.py`
maintains the file: it fills in every statistic from OpenML and the data
itself, and reports columns that look like unrecognized category codes.

### Cross-validation runner

`run_cv(learners, datasets, n_folds=10, fit_timeout=60, discretizer=None,
random_state=0, cache_dir=None, measure_fn=None, verbose=True)` runs every
learner on every dataset, with stratified cross-validation:

- In each fold a `DataSpec` is built on the training part only
  (`build_dataspec`, numeric attributes discretized by `discretizer`,
  default `TreeDiscretizer()`), and the test part is binarized against it,
  so every learner sees the same Boolean data.
- Each fit runs in a persistent worker process (`TimeoutRunner`) and is
  stopped after `fit_timeout` seconds. A time-out or an exception is
  recorded in the row's `error` column instead of ending the run.
- Every model is measured the same way (`accuracy`, `n_rules`,
  `n_conditions`, `fit_time`), plus whatever `measure_fn(model, test_data)`
  returns.
- With `cache_dir`, each finished (dataset, fold, learner) result is saved,
  keyed by the dataset, the split and the learner's name and parameters,
  so a restarted run -- or one with an added learner -- only fits what is
  new.
- `verbose` prints a line per dataset and per fit, marking cached ones.

The result is one long-format pandas `DataFrame`, one row per (dataset,
fold, learner).

### Statistics and reports

`pyrulearn.experiments.stats` works on that table: `mean_rank` (a
learner's average rank across datasets, failures ranked last),
`win_counts` (datasets where a learner was best, ties split),
`friedman_test`, `critical_difference` (Nemenyi) and
`critical_difference_diagram` (Demšar's diagram). `pyrulearn.experiments.
report` renders Markdown: `render_setup_section` (a report's Setup
section, from a description string) and `render_results_table` (means per
`group_by`, optionally with an overall row).

### Comparisons built on it

Each comparison in `demos/` has a quick default run (a few small
datasets, fewer folds, output not checked in) and a `--full` run whose
report and plots are checked in next to the script:

- `ripper_comparison` -- the RIPPER family: Weka's JRip, wittgenstein's
  RIPPER, Pypper and Slipper.
- `numeric_discretization` -- the discretizers (decision tree, FUSINTER,
  equal width, equal frequency, with and without boundary correction) at
  3 to 12 intervals with Pypper, PFossil and PFoil; per class vs. all
  classes on many-class data; and JRip and J48 on raw numeric data vs.
  discretized.
- `heuristic_comparison` -- rule learning heuristics inside one plain
  separate-and-conquer learner, after Janssen & Fürnkranz (2010).
- `seco_learners_comparison` -- CN2, PFoil, PFossil, Pypper and PyLORD
  against JRip and the Java LORD.
- `representations` -- fit time of the four data representations over
  training-set size and density (its own loop, not `run_cv`).
- `multiclass_decomposition` -- one-vs-rest, ordered one-vs-rest and
  pairwise decomposition with Pypper and PFossil, after Fürnkranz's round
  robin study: accuracy, and training and prediction time over the
  number of classes.
- `native_vs_interfaced` -- four native rule learners (RuleFit, Boomer,
  CBA, Pypper) against the external reference implementations they're
  modeled on (IMod:RuleFit, MLRL:Boomer, PArc:CBA, Weka:JRip); a fifth
  pair, ORB vs. RKD:RuleBoosting, is checked separately
  in its own preliminary section, since neither's exhaustive search
  scales to this demo's wider data.
- `dense_vs_modular` -- every `Dense*` learner against its modular twin
  (CPAR, CPAR PRM, ENDER, ENDER CS-Exp, ENDER MLRules, Boomer, ORB, LRI):
  accuracy, fit time and whether the models are identical; the modular
  ones again on the horizontal bitmap, and ORB and LRI with beam searches of width 5
  and 10. Exhaustive ORB is checked separately on a few tiny datasets,
  since it doesn't scale to the rest.

`heuristic_isometrics` and `covering` are illustrations, without
cross-validation.

## Try it

```
python examples/demo.py
python -m pytest                       # the whole test suite
python -m pytest tests/test_models.py  # or a single file
```

`examples/demo.py` exercises the whole surface: manual rules, dataset
coverage annotation, decision-tree rule extraction, `cross_val_score`,
a pattern-string import round-trip, and two
coverage-space plots -- a rule set as a `DecisionList`'s arrow-connected
cumulative path (`coverage_space_rulelist.png`), and one rule's own
precision-ordered refinement path (`rule_refinement.png`).

Other demos: `demos/heuristic_isometrics.py` (heuristic isometrics),
`examples/demo_workflow_comparison.py` and
`demos/seco_learners_comparison.py` (cross-benchmark model
comparisons, each writing a `*_report.md`), and
`demos/representations.py` (fit time of the four data representations
over growing training sets and falling density, checking at every point
that all of them give the same rules as `HorizontalBitmapRepresentation`).

## Not yet implemented (left as clear extension points)

Development plans, with the reasons behind them, are in
[`ROADMAP.md`](ROADMAP.md).

- **More rule learners and importers.** `pyrulearn` supports a wide suite of classic and modern rule learning algorithm. Beyond what's already interfaced, the framework supports to be continuously expanded via 
 `StringRuleImporter` for importing rule in text formats or
  `ObjectRuleImporter`s for direct import of Python objects from other learners. We intend to add new learners on a per-need basis. 

- **Closing gaps between native algorithms and their reference papers.**
  `AQR`'s literal *star* search and `PyLORD`'s exhaustive branch-and-bound
  are both currently approximated by a seed-restricted `BeamSearch` (see
  `pyrulearn.learners.seco`/`pyrulearn.learners.pylord`); `Pypper`'s
  residual IREP\* re-growth pass is the one piece still missing from its
  optimization phase (`ReplaceReviseOptimization` itself is already done).

- **Inverted heuristics** (Stecher, Janssen & Fürnkranz): evaluating a
  refinement in the coverage space of the rule it refines -- the origin
  moved to the parent rule's own coverage point -- rather than from the
  empty rule, which favors longer, more specific rules in top-down search.
  They would fit in as a heuristic wrapper next to `DeltaGain`, plotted
  by `heuristic_isometrics` and compared in `heuristic_comparison`.

- **Representation and analysis extensions.** Rule bodies beyond pure
  conjunctions (e.g. general CNF/DNF rules); native readers for sparse
  ARFF (`{index value, ...}`) and libsvm files (`SparseDataRepresentation`
  and `from_scipy` exist, but `pyrulearn.data.io` and the demos' OpenML
  loader still go through a dense DataFrame, so a genuinely sparse dataset
  like OpenML's `connect-4` densifies on the way in); and a logic/SAT
  interface (export to `sympy`/DIMACS, SAT-based analysis) -- an earlier
  version was removed from this release and will return in a reworked
  form.

- **Error-correcting output codes** (Dietterich & Bakiri 1995) for
  multi-class classification, next to the existing one-vs-rest, ordered
  and pairwise decompositions (see *Multiclass classification*): one
  binary model per bit of a code word assigned to each class, and the
  class whose code word is nearest to the predictions wins.

- **Multi-label classification, preference learning and label ranking.**
  Learning tasks beyond a single class per example: several labels per
  example (multi-label; e.g. BOOMER's full multi-output rules, of which
  `Boomer` and `MLRLBoomer` cover only the single-label case), and
  predicting preferences or rankings over labels (preference learning,
  label ranking; the rules' per-class score vectors could feed a
  label-ranking combiner). Each needs its own data representation (label
  sets or rankings instead of `y`), models and evaluation measures.

- **Project housekeeping.** Publishing to PyPI (installation is currently
  from GitHub); revising the demos in `examples/`, some of which were
  written against earlier versions of the API.

## Authors and contributions

**Johannes Fürnkranz** ([Johannes Kepler University Linz](https://www.jku.at),
`juffi@faw.jku.at`) is the author of pyrulearn and responsible for the project.

The programming was done by **Claude** (Anthropic's AI model, used through
[Claude Code](https://claude.com/claude-code)) in continuous dialogue with the
author, who directed the design and reviewed the results. Contributions, following
the [CRediT](https://credit.niso.org) contributor roles:

| Role | Contributor |
|---|---|
| Conceptualization | Johannes Fürnkranz |
| Software design (architecture, APIs, choice and adaptation of algorithms) | Johannes Fürnkranz, Claude |
| Software (implementation, tests, documentation) | Claude |
| Validation (review; cross-checks against reference implementations) | Johannes Fürnkranz, Claude |
| Supervision and project administration | Johannes Fürnkranz |

Since an AI system cannot take responsibility for a work, Claude is not
listed as an author in the citation metadata, in line with the guidance of
publishers and bodies such as COPE and ICMJE on AI tools; its contribution is
documented here and through `Co-Authored-By: Claude` trailers on commits.

## How to cite

If you use pyrulearn, please cite it via [`CITATION.cff`](CITATION.cff)
(GitHub's "Cite this repository" button), or as

```bibtex
@software{furnkranz2026pyrulearn,
  author  = {F{\"u}rnkranz, Johannes},
  title   = {pyrulearn},
  year    = {2026},
  version = {0.6.0},
  url     = {https://github.com/juffif/pyrulearn}
}
```

Please also cite the original publications of the algorithms you use
(see the tables above and [`references.bib`](references.bib)).

## License

pyrulearn is free software, released under the
[GNU General Public License, version 3 or (at your option) any later version](LICENSE).
Copyright © 2026 Johannes Fürnkranz.

## Acknowledgements

pyrulearn re-implements algorithms published by many others (see
[`references.bib`](references.bib)) and interfaces to scikit-learn,
`wittgenstein`, `imodels`, `pyarc` and `pyfim`, Weka and
LORD. These are used through their public interfaces under their own
licenses; none of them is bundled with pyrulearn. Datasets used in the demos
come from public repositories such as OpenML and the UCI repository and are not
redistributed here.
