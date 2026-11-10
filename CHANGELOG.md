# Changelog

All notable changes to pyrulearn are listed here, newest first. The project
is in early development (alpha): until 1.0, minor versions may change the API.

## Unreleased

### Fixed

- `TimeoutRunner` (and so `run_cv`) no longer counts a new worker
  process's start-up (a fresh interpreter importing numpy, scikit-learn
  and pyrulearn, a second or more) as part of a call: `run` starts the
  timeout only once the worker is up (`wait_until_ready`, new
  `startup_timeout=`). Before, the first fit of a run and the first fit
  after every timeout could time out early, and their `fit_time` (with
  `n_jobs=1`) included the start-up.
- A timed-out fit's child processes (a Weka or LORD JVM, a joblib pool)
  are stopped again: `_kill_tree` ran only after the worker had been
  killed, when its children could no longer be found from it, so they
  kept running. It now runs first. `psutil`, which it needs on Linux and
  macOS, is in the `experiments` and `all` extras.

## 0.6.0 (2026-11-08)

Deep Models: `DeepModel`, a rule network in negation normal form, and four
learners for it -- `DeepLocalSearch`, `DeepGenetic`, `DeepSAT` and
`DeepNeuroFuzzy` --, generated datasets (`pyrulearn.datasets`: KRK
illegality, parity) and `demos/deep_rule_networks.py`.

### Added

- `DeepModel` is no longer a stub: a formula in negation normal form, a
  fully connected network of alternating AND and OR layers with 0/1 edge
  weights, `DeepModel(input_spec, labels, layers)` with the hidden layer
  sizes like a neural network's (by default one AND layer of
  `ceil(sqrt(n))` nodes for `n` features, a DNF) or `weights=`, instead of members and
  `dependencies`. The weights are the model; `to_concepts()` reads them
  as stacked rules -- every AND node a rule body, every OR node a concept
  with one rule per incoming edge, a concept a feature for the layers
  above -- by default without the pass-through nodes, as `to_string`
  prints it. `node_values`/`concept_values(data)` give every node's
  value, hidden ones included.
- `pyrulearn.learners.deep`: `DeepLocalSearch` learns a `DeepModel` by
  Beck & Fürnkranz's stochastic hill climbing (best single flip on
  mini-batches, best weights kept), with flips that keep rules
  consistent with the `DataSpec`'s constraints (`moves=`,
  `contradictions`); on the shared base `DeepRuleLearner`
  (architecture, targets, the start as a component: `RandomInit`,
  `ModelInit`, `RuleSetInit` -- a rule set or a rule learner's rules as
  the start network). `nnf_forward` evaluates weight matrices on a
  Boolean matrix. `DeepModel.pruned()`, `used_nodes()`, `n_edges`.
  `DeepGenetic`: a memetic algorithm -- crossover of networks aligned by
  their nodes' values, mutation, a few hill-climbing steps per child; the
  population can be seeded with rule learners' rules; `crossover=`
  `"nodes"` (default), `"subtree"` (a node with its cone), `"mixed"`,
  `"none"`.
  `trace=True` on both: the model's `trace` lists every new best network
  during the fit with its effort (candidate-rows), seconds and training
  error (`TracePoint`), `DeepModel.trace_model(i)` the network of a point.
- `DeepSAT`: the weights by SAT solving, an exact fit of the training
  data (Seip, Fürnkranz et al.), with the known inputs and labels as
  constants; `minimize="inputs"`/`"weights"` for MaxSAT, the fewest
  conditions or edges -- by an anytime linear search with a trace
  (`maxsat="linear"`, default; `max_seconds`, `max_conflicts`) or RC2.
  `constraints=True`: the `DataSpec`'s constraints as clauses on the
  weights. PySAT as the optional extra `sat`. Checked against the
  authors' code (same satisfiability, same minimal size).
  `TracePoint.size`: the size a learner minimizes.
  `size_penalty` (lambda): soft examples -- MaxSAT minimizes `errors +
  lambda * size` (a regularized loss; anytime from the first solver call,
  for the linear search by a second totalizer over the errors; integer
  costs for the solvers); `None` (default) is the exact fit. With soft
  examples a given `init` is the linear search's first solution. The
  linear search prunes every network to its backbone (`pruned_weights`)
  before its cost bounds the next step, the start too: dead edges don't
  count.
- `pyrulearn.datasets`: generated datasets. `krk_illegal(n, random_state)`
  -- the KRK illegality task (Muggleton et al. 1989): positions of white
  king, white rook and black king, white to move, labeled illegal or
  legal by the rules, `n` distinct ones drawn from all 8^6; the six
  coordinates nominal. `KRKIllegal`: the same as a dataset for `run_cv`.
  `relations=True`: the ILP background knowledge as 24 relational
  features (`eq`, `adj`, `lt` of two pieces' files or ranks), with which
  the task has an exact theory; `coordinates=False` keeps only those.
- Relational attributes from data: a frame's `attrs["relational"]`
  (`{column: {"sources", "expression"}}`) makes those precomputed Boolean
  columns RELATIONAL attributes in `build_dataspec`; `evaluate_feature`
  reads a relational feature's value from its column (it raised
  `NotImplementedError` before); `DataSpecBuilder.add_relational(...,
  negation=True)` adds ``not {name}`` (off by default).
- `DeepNeuroFuzzy`: the weights by gradient descent on a fuzzy
  relaxation (Wang et al., AAAI 2020): continuous weights in [0, 1],
  product AND/OR nodes, Adam on mini-batches, MSE or log loss, an L2
  penalty, random binarization (by default with the straight-through
  estimator), the weights thresholded at 0.5 after every epoch and the
  best network kept; anytime with a trace, `init` as a start. numpy only.
- `size_penalty` (lambda) on all four deep learners, one objective:
  training errors + lambda per edge of the backbone (dead edges free).
  Local search and the GA apply it on the whole training set (best
  network, fitness, final climb), not on mini-batches; `DeepNeuroFuzzy`
  as an L1 term on a summed loss; `DeepSAT` as before (soft examples).
  `None` (default) leaves the learners as they were.
- `max_seconds` on `DeepLocalSearch`, `DeepGenetic`, `DeepNeuroFuzzy`: a
  time budget for the fit, the best network so far when it is up.
- `pyrulearn.datasets.parity(n_relevant, n_irrelevant, n, noise)` and
  `Parity`: the parity of a few relevant Boolean features among
  irrelevant ones (Pypper, 5-fold: 0.71 for 3 of 10, 0.62 for 4 of 10,
  0.52 for 5 of 10).
- `pyrulearn.datasets.parity_groups(n_groups, group_size, threshold,
  n_irrelevant, n, noise)` and `ParityGroups`: at least `threshold` of
  `n_groups` groups of Boolean features have odd parity -- hidden
  concepts that pay off one by one, and a flat DNF much larger than the
  deep network (2 of 4 groups of three: 96 rules of six conditions
  against a network of 82 edges).
- `target_class` on the deep learners: the class the output node learns
  (the most frequent other class the default); without it, as before,
  the less frequent of two classes, or one output per class. On Beck's
  artificial datasets it decides everything: plain SAT on NNF(1) found an
  exact fit for the formula's class in 0.4 s, and none in an hour for
  its negation, the less frequent class there.
- `DeepModel(first="and" | "or")` and any number of hidden layers: the
  first layer's type, the others alternating, so the output can be an
  AND (e.g. the papers' `[6, 6]`); an OR first layer gives CNFs. All the
  deep learners take `first`.
- Catalog: `CatalogEntry.csv_url`, a CSV file as a dataset's source,
  downloaded once into the data home (`fetch_csv_frame`; filled in by
  `tools/build_catalog.py` too). The tags `deep` and `artificial`: the
  datasets of Beck & Fürnkranz (2021) and Seip et al. -- their 20
  artificial datasets (`artificial-nnf-5`, ...), `car-binary` and
  `connect-4-binary` from Beck's repository, and the MONK's problems,
  tic-tac-toe, vote, kr-vs-kp and mushroom already in the catalog.
- `build_dataspec`: a column of dtype `bool` is a Boolean attribute
  (`x`, and `not x` with negations) instead of a binary nominal one
  (`x=False`/`x=True`).

## 0.5.2 (2026-11-08)

Isometrics in heatmap colors, and a leaner PrePost representation.

### Changed

- Isometrics (`RuleHeuristic.plot_isometrics`, `CoverageSpace.contour`)
  are drawn from red for low scores to green for high ones, as in a
  heatmap (`isometrics_cmap()`), instead of in viridis, and lie on lightly
  shaded bands of the same colors (`shade=`, 0 for bare lines).
- `PrePostNListRepresentation` no longer keeps the path bitmask of every
  tree node, only its pre-order number, subtree end and parent, as the
  N-lists of PrePost (Deng et al. 2012): a refinement and `coverage` test
  ancestors by binary search, and `features_of` walks up to the root. On
  `adult` (8,000 examples) it holds 5.2 instead of 12.0 MiB with negation
  and 1.9 instead of 2.9 MiB without. Fits learn the same rules; there,
  CN2 and ENDER take about as long as on `NListRepresentation`, PFossil,
  Pypper and PyLORD up to 30% longer with negation and up to 15% without.
  `demos/representations.py` measures it again, as "PrePost".

## 0.5.1 (2026-11-02)

Fixes to CN2 and Pypper, and the demos re-run with them: CN2's
significance test filters rules instead of stopping the search, and its
search ends once nothing significant is left to find; Pypper prunes with
the Laplace estimate, as JRip does. The separate-and-conquer comparison is
rebuilt on the 69 datasets of the other comparisons.

### Added

- `demos/noise_robustness.py`: rule learners under class noise in the
  training data, by learner family.
- `RuleHeuristic.bound`/`batch_bound`: an upper bound on the score of every
  refinement of a rule, by default the score at the ``(tp, 0)`` projection;
  `LikelihoodRatio`, which isn't monotone in `tp` and `fp`, bounds by the
  maximum over the corners of the refinements' coverage. The searches'
  optimistic pruning uses it.
- `CN2(stop_covering=...)`, passed through to `SeCo`.
- `pyrulearn.pruning.LikelihoodRatioTest`: CN2's significance test over
  all classes, as Clark & Niblett (1989) state it -- the covered class
  distribution against the scope's, at significance level `alpha` with
  one degree of freedom less than the classes in scope; on two classes the
  two-class test. With `CN2(heuristic=Entropy())` and `model=DecisionList`,
  the original ordered CN2. CN2's default stays the two-class test of the
  rule's head. `chi2_critical_value(df, alpha)` without scipy.
- `run_cv(cached_only=True)`: the table of what `cache_dir` already holds,
  with nothing fitted and no data loaded.

### Changed

- `Pypper` prunes to the prefix with the best `Laplace` estimate instead of
  the best precision (`prune_heuristic=`, `Precision()` for Cohen's
  ranking), as Weka's JRip does: plain precision kept long prefixes that
  cover a few pruning examples without error. On 68 datasets (5-fold)
  82.72% instead of 82.45%, with a few more rules.
- `demos/seco_learners_comparison.py` rebuilt: the 69 datasets of the other
  comparisons (`letter` with a 3600s cap); CN2 with beam 1 next to beam 5,
  and Clark & Niblett's ordered CN2 (entropy, the significance test over all
  classes) at both beams; AQR and Opus(Laplace) in the main comparison
  instead of a preliminary check. Fit times are the geometric mean of their
  ratios to Weka:JRip's, a time-out counted at the cap; the plots show one
  point per learner (mean accuracy rank against time and against size)
  next to the accuracy-complexity scatter, one point per dataset. AQR and
  both Opus variants are optional, 92% of the fit time: `--full` reports them
  from the cache, `--slow` runs them.
- `CN2`'s significance test is a `filtering` criterion by default, as in
  Clark & Niblett's and Clark & Boswell's search, no longer a `stopping`
  criterion, which halted a search the moment the round's best candidate
  was insignificant and made CN2 underfit, most of all under class noise.
  The old behavior is
  ``CN2(stopping=ThresholdPrePruning(LikelihoodRatio(), 3.841, "<"))``.

### Fixed

- `BeamSearch` with a `filtering` criterion ends once no rule in its beam
  has a refinement that could pass it
  (`PrePruningCriterion.may_accept_refinements`, given the rule, its
  statistics, the data and a bound from the objective's
  `criterion_bound`). Without it, a search with nothing significant left
  to find ran every refinement out, so the default CN2 timed out on small
  datasets (sonar, ozone-level-8hr in `demos/rule_models.py`), most of all
  in a covering loop that tries every remaining seed. The rules learned
  are the same.

## 0.5.0 (2026-10-31)

### Added

- `Bagging` (`pyrulearn.learners.bagging`): each member learned from a
  bootstrap sample of the training data, represented by the drawn rows
  weighted by their numbers of draws, so learners whose counts are sums of
  example weights learn the same rules as from duplicated rows.
- `AdaBoost` (`pyrulearn.learners.adaboost`): SAMME over any learner, the
  members voting with weights ``log((1 - err) / err) + log(K - 1)``, a
  member's error at least half an example. `early_stopping=False`: where
  SAMME would stop (a member no better than random guessing, or one without
  training errors), the example weights are reset to those of a bootstrap
  sample and boosting goes on, for up to twice `n_estimators` rounds.
- `EnsembleModel(combiner=...)`: the members' deciding rules
  (`RuleModel.deciding_rules`, the strongest covering rule of the predicted
  class, or a decision list's first covering rule) resolved by any rule
  combiner, as a rule set resolves its covering rules -- every conflict
  resolution at the level of the members, too. `ensemblemodel_to_flatruleset`
  is the same at the level of the rules.
- `staged(n)`: a model after its first `n` rounds, identical to a refit
  with `n` rounds -- `EnsembleModel.staged` keeps the first `n` members
  (bagging, AdaBoost); Slipper, ENDER and Boomer record their rounds in
  `LinearRuleModel.stages`, from which `LinearRuleModel.staged` rebuilds the
  model. All sizes of a series from one fit.
- `abstain=` for `EnsembleModel`, `Bagging` and `AdaBoost`, on by default: a
  member votes only where its rules cover the example, its default ignored,
  with a combiner as well as for the plain and the `member_weights` vote.
  Suits one-vs-rest concept sets: with their defaults
  voting, boosted Pypper loses accuracy on data with many classes.
- Rules of any class: a search with no target class looks for the best rule of
  any class, its heuristic choosing the rule's head. Every heuristic gets this
  from its `batch_score` alone (`RuleHeuristic.best_head`: each class scored
  against the rest, the best one wins); the `MajorityHead` mixin is a speed-up
  for heuristics that always prefer the covered majority (`Precision`,
  `Laplace`, `GHeuristic`, `CoveredPositives`), and the `ClassNeutral` mixin
  is for heuristics defined on the whole class distribution (`Entropy`, now
  CN2's multi-class entropy there). In the search, `HeadObjective`; the head
  may change from a rule to its refinement, except with a gain heuristic
  (`FoilGain`), which compares heads only at the first refinement, where all
  of them start from the scope, and then keeps the head.
- `fit(data, model=DecisionList)` for CN2, PFoil, PFossil, SeCo and Pypper:
  one covering loop over all classes, each rule the best of any class, the
  default the majority of what no rule covers -- CN2's ordered mode.
- Seeds and heads as three independent choices of a covering loop: `seed=`
  (`FirstSeed()`, `RandomSeed()`) picks the example a search explains,
  `space_init=` where the search starts, and `target_class=` the head --
  a class (`FixedClass`) for one concept, or `HeadPerRefinement()` (the heuristic
  chooses), `HeadPerSeedExample()` (the seed's class) or `HeadPerRule()` (the best of
  one search per class) for a decision list. A seed implies its
  generalizations and its class unless those are given, so
  `CN2(seed=RandomSeed())` is AQ's seed covering with CN2's search. All
  in `pyrulearn.learners.seco`, for every SeCo learner (CN2, AQR, PFoil,
  PFossil, Opus, CPAR) and Pypper, which also take `space_init=` now, and as
  `fit(data, model=DecisionList, target_class=..., seed=...)`. An unset
  `seed` or `space_init` leaves a learner's parameters (and so cached results)
  unchanged. With weighted covering (CPAR), a seed only seeds the class loops:
  a decision list needs covered examples removed.
- `demos/rule_models.py`: rule sets (four combiners: best rule, voting, micro
  and macro distribution voting), cascades and decision
  lists of CN2, PFossil and Pypper, on multi-class and two-class data, with a
  ranking of the kinds of model over all three base learners per group, its
  Friedman test and critical-difference diagram, and decision lists against
  rule sets over the number of classes. The large `kropt` and `letter` (18 and
  26 classes) run 5-fold, with 1800 s per fit.
- `demos/base_vs_composite.py`: bagged concept sets of Pypper, PFossil and
  PFoil (10, 25 and 50 members from one fit), combined at the level of the
  members or of the union of their rules, with each of the four conflict
  resolutions. It replaces the forest demo of the same name (the trees' vote
  against the rule set of their leaves).
- `demos/model_vs_rule_weighting.py`: AdaBoost over concept sets (model
  weighting; members abstaining or voting with their defaults) against
  Slipper, ENDER and Boomer (rule weighting), by accuracy against the number
  of rules.
- `demos/multiclass_decomposition.py`: the decompositions against
  one-vs-rest over the number of classes, on the 26 multi-class datasets of
  the rule-models demo.
- `experiments.stats.critical_difference_groups`: the groups a
  critical-difference diagram joins by a bar.

### Changed

- The combiners describe themselves with the paper's names, so a printed rule
  set's `% conflict resolution:` line reads `voting (one vote per covering
  rule)`, `micro distribution voting (sum of covered class counts)` (CN2's) or
  `macro distribution voting (sum of covered class proportions)`; the class
  names (`CountVoteCombiner`, `MicroVoteCombiner`, `MacroVoteCombiner`) stay.
- `critical_difference_diagram` draws every maximal group of learners within
  one critical difference of each other, overlapping ones included, in rows
  below the points. It used to skip a group that overlapped the previous one,
  so two learners within one CD could look significantly different.

- `model=DecisionList` (and `model=FlatRuleSet`) for a learner that doesn't
  seed its search (CN2, PFoil, PFossil) now searches rules of any class instead
  of AQ's seed covering, which AQR keeps -- and which every learner gets with
  `seed=RandomSeed()`. Pypper's `model=DecisionList` is the new covering loop
  instead of its converted class cascade.

### Fixed

- `Correlation` and `ChiSquare` take a product of counts that a rounding
  error puts below 0 (fractional example weights, e.g. from AdaBoost) as 0:
  the batch score warned of an invalid square root, the scalar one raised a
  math domain error.
- `run_cv`: a cached timeout is reused only if the current `fit_timeout` is
  no longer than the time that fit was given; a re-run with a longer timeout
  used to return the old timeouts.

## 0.4.2 (2026-10-25)

The horizontal bitmap searches on its packed bits: fits on it need about as little
memory as on the vertical bitmap and run faster, with the same rules. The
representations demo is re-run on it, and the tutorial follows 0.4.1. No API change.

### Changed

- `HorizontalBitmapRepresentation` no longer unpacks the whole feature matrix
  during a search. `refine_cover`, `features_of`, `batch_cover_counts` and
  `batch_cover_sums` read the packed bits directly, and the batch counts unpack
  only the open features' bytes, a bounded block of rows at a time
  (`BATCH_BLOCK_CELLS`). Fits keep their peak memory far lower (on adult,
  8,000 rows: CN2 from 21.5 to 1.4 MiB, Pypper from 15.1 to 1.2 MiB, ENDER
  from 7.9 to 3.5 MiB) and run faster, and they learn the same rules.
- `demos/representations.py`: a cap per learner (`FIT_TIMEOUTS`; PyLORD
  600 s, the others 300 s), and a cached time-out is fitted again when the
  current cap is larger. With 600 s, PyLORD finishes 16,000 rows of `adult`
  (with negation features) on every representation and 32,000 rows on the
  vertical bitmap (505 s).
- The tutorial, re-executed for 0.4.1: accuracies as percentages, the inferred
  attribute types (ordinal, numeric, binary, nominal), learning without negated
  features, and the lower bound for ordinal columns.

## 0.4.1 (2026-10-21)

Discretization becomes an object passed as `discretizer=` -- the decision tree so far,
still the default at 8 intervals, and new ones: FUSINTER, equal width, equal frequency,
and the `BoundaryCorrection` and `PerClass` wrappers. `demos/numeric_discretization.py`
compares them. A small break: `max_intervals=` becomes
`discretizer=TreeDiscretizer(max_intervals=...)`. No change to what the default learns.

### Added

- Discretizer objects (`pyrulearn.data.discretization`, also exported from
  `pyrulearn` and `pyrulearn.data`): `TreeDiscretizer` (the decision-tree
  discretization so far, still the default), `FusinterDiscretizer`
  (FUSINTER, Zighed, Rabaseda & Rakotomalala 1998: supervised bottom-up
  merging of class-homogeneous intervals, native),
  `EqualFrequencyDiscretizer` and `EqualWidthDiscretizer` (unsupervised
  equal-frequency and equal-width binning, native), and `BoundaryCorrection`,
  which wraps another discretizer and moves each of its thresholds to the
  nearest boundary point -- a gap between adjacent values where the class
  changes (Fayyad & Irani, 1993), and `PerClass`, which runs a
  discretizer once per class (that class against all others) and keeps
  the union of the thresholds.
- `TreeDiscretizer(max_intervals=None)` grows the full tree, which cuts
  at exactly the boundary points; `FusinterDiscretizer(max_intervals=...)`
  caps FUSINTER by going on merging the least harmful pair.
- `demos/numeric_discretization.py`, rewritten: the discretizers at 3 to
  12 intervals with Pypper, PFossil and PFoil; per class vs. all classes
  on many-class data; JRip and J48 on raw numeric data vs. discretized.
  Its results are saved, so `--replot` redraws report and plots without
  running.

### Changed

- `run_cv` looks up the cache before loading and encoding: a fold whose
  cells are all cached is not discretized and encoded, a dataset whose
  cells are all cached not even loaded. A fully cached rerun takes
  seconds.

### Changed (breaking)

- `build_dataspec`, `read_csv`, `read_arff` and `run_cv` take
  `discretizer=` (a `Discretizer`, default `TreeDiscretizer()`) instead of
  `max_intervals=`: `max_intervals=4` becomes
  `discretizer=TreeDiscretizer(max_intervals=4)`. `run_cv`'s cache keys of
  the default tree are unchanged, so existing caches stay valid.
- `pyrulearn.interfaces.sklearn._prettify_threshold` is now
  `pyrulearn.data.discretization.prettify_threshold`; the constant
  `DEFAULT_MAX_INTERVALS` lives there too (`pyrulearn.data.io` still has
  it; `pyrulearn.experiments.runner` no longer does).

## 0.4.0 (2026-10-20)

A change to the feature space: numeric attributes always test both directions, and a
new attribute type, ORDINAL, for few ordered values and small sets of numbers, with
equality and inequality tests. Few-valued numeric columns, which `read_csv` and `run_cv`
so far made nominal, are now ordinal, so experiments on data with such columns learn
from a different encoding than in 0.3.x. Together with the beta clean-up: the names
deprecated in 0.2.0 and 0.3.4 are removed. The demos are not re-run in this release.

### Added

- `AttributeType.ORDINAL` and `DataSpecBuilder.add_ordinal(attribute, values)`: for
  values `v1 < ... < vk`, the features `x=v` for every value, `x<=v` and its
  complement `x>v` for every value but the largest, and with negation `x!=v`. One
  `OrdinalGroup` constraint propagates all of them through the value positions
  (`size=medium` implies `size<=medium` and `size>small`; `size<=medium` with
  `size>small` implies `size=medium`; exact for the known values, like a nominal
  attribute's group). Values compare by their position in the given
  order, or numerically if they are all numbers, so an unseen number still satisfies
  the right `<=`/`>` tests; a value outside a non-numeric order counts as missing.
  Missing values: every feature false by default; `MAJORITY` imputes the lower median
  value; `SEPARATE` adds a missing feature as one more position, outside the order.
  `merge_dataspecs` unions numeric orders (sorted) and requires non-numeric ones to
  agree; pretty rule display merges `<=`/`>` into an interval and `!=` into a value
  set, as for numeric and nominal attributes.

### Changed (breaking)

- Numeric attributes always get both `x>=t` and `x<t` (and both `x<=t` and `x>t` for
  `le_thresholds`), whatever the negation setting; until 0.3.4, `x<t` existed only
  with negation on, so the encoding without negation could bound a numeric attribute
  from below only. `DataSpecBuilder.add_numeric` loses its `negation=` parameter.
  Negation features are now `not f` and `x!=v` only: `x<t`/`x>t` no longer count
  (`DataSpec.has_negation_features`, `without_negations()`), though `negation_of`
  still pairs them with their complements.
- `build_dataspec` (and so `read_csv` and `run_cv`) infers a numeric column with few
  distinct values -- what the cardinality heuristic so far made NOMINAL -- as
  ORDINAL, from three values on; with two it stays BINARY. A column declared numeric
  in an ARFF header stays NUMERIC. The bound is the new `max_ordinal_values=10`
  (`build_dataspec`, `read_csv`, `read_arff`, `run_cv`; the heuristic's former
  fixed 10), independent of `max_intervals`.
- `run_cv`'s fold cache key includes the encoding: a version
  (`experiments.runner.ENCODING_VERSION`), so fold results cached by 0.3.x are not
  reused for the new encoding, and `max_intervals` and `max_ordinal_values`, which it
  didn't detect before.
- `JavaLord` is renamed to `JavaLORD`, to spell LORD as everywhere else
  (`PyLORD`, `LORDImporter`, `LORD_JAR`); no alias is kept.

### Removed (breaking)

- The pre-0.2.0 learner names (`JRip`, `PART`, `J48`, `IREP`, `RIPPERk`,
  `DecisionTree`, `RandomForest`, `BayesianRuleList`, `BayesianRuleSet`,
  `ImodelsRuleFit`, `ImodelsSlipper`, `RealkdRuleBoosting`, `PyarcCBA`,
  `LordJar`) and the 0.2.0 name `MlrlBoomer`, which were served with a
  `DeprecationWarning`, together with `interfaces.base.deprecated_aliases`.
- `BooleanDataRepresentation`, the name `HorizontalBitmapRepresentation`
  had until 0.3.4, and with it the deprecation hooks (module-level
  `__getattr__`) of `pyrulearn`, `pyrulearn.data` and
  `pyrulearn.data.representation`.

## 0.3.4 (2026-10-19)

A new default representation, the vertical bitmap: the bit matrix packed by feature
instead of by example. The separate-and-conquer learners and PyLORD fit fastest on it,
ENDER close to the fastest, at the memory of a bit matrix. The row-packed matrix is
renamed `HorizontalBitmapRepresentation` to match. Weka's learners lose most of the
time our interface spent writing their training file. All timed demos re-run. No
change to what any learner learns.

### Added

- `VerticalBitmapRepresentation`: the bit matrix packed by feature -- one
  bit-vector over the examples per feature, in 64-bit words. Covering a
  rule ANDs its features' bit-vectors, counting is a popcount against a
  class's bit-vector, and a search's batch counting ANDs the cover into
  the open features' words, over just the words it still touches. Same
  memory as the row-packed matrix (one bit per cell); identical rules.
  In `demos/representations.py` as `VBitmap`.
- `demos/ripper_comparison.py`: a critical-difference diagram of the
  learners' fit times, next to the one for accuracy.

### Changed

- `VerticalBitmapRepresentation` is the default representation (first in
  `REPRESENTATION_PREFERENCE_ORDER`): what `read_arff`/`read_csv`/`encode`
  and `run_cv` build when no `representation=` is given, instead of
  `NListRepresentation`. On `adult` (32,000 rows, with negation features)
  CN2, PFossil and Pypper fit 1.5-2.7x faster on it than on N-lists,
  PyLORD finishes 16,000 rows where every other representation times out,
  ENDER is about 10% slower; a CN2 fit needs 4.5 MiB in total instead of
  72 MiB. Building one from a `scipy.sparse` matrix sets the bits
  directly, without a dense intermediate. The timed demos run on it, and
  `dense_vs_modular` compares the modular learners on it with the
  horizontal bitmap.
- `BooleanDataRepresentation` is renamed to
  `HorizontalBitmapRepresentation`, so that the two bitmaps' names say
  what distinguishes them: the matrix packed by example (horizontal) or by
  feature (vertical). The old name still works, with a
  `DeprecationWarning`, and will be removed in 0.9.0. In
  `demos/representations.py` it is `HBitmap` (was `Boolean`).
- The Weka learners (`WekaJRip`, `WekaPART`, `WekaJ48`) write their
  training file as byte arrays instead of through the general
  `write_arff`, which converted and quoted every cell on its own: 0.3 s
  instead of 17.8 s for `adult` (32,000 x 278), time that counted in every
  Weka fit. The file is byte for byte the same -- JRip seeds its random
  grow/prune splits from instance text, so even ``0``/``1`` for
  ``False``/``True`` would change its rules -- and so are the models.
- `demos/representations.py` no longer measures `PrePostNListRepresentation`
  (within a few percent of `NListRepresentation` everywhere in 0.3.3), and
  records in its report that the numerical libraries ran on one thread.
- All timed demos re-run on the new default (`--full --n-jobs 10`, numerical
  libraries on one thread).

## 0.3.3 (2026-10-15)

Faster N-lists for the cases the demos showed to be slow, and a `run_cv` that gives up
on a timed-out cross-validation: row subsets of an N-list are derived from its tree
(5-8x faster, which pairwise and ordered decompositions need), batch counting uses a
dense product where the patterns are dense (the boosting learners are now as fast on
N-lists as on Boolean data or faster), and a timed-out fold fails the whole
cross-validation of its learner and dataset (`stop_on_timeout`, with `timed_out` for the
caller). All demos re-run. No change to what any learner learns, apart from rare
tie-breaking.

### Added

- `demos/representations.py`: ENDER as a fifth learner; `--parallel`, which prepares every
  point first (the representations, their memory, the `coverage()` timings) and then
  fits the learners side by side, one worker process each; a table of the growth
  exponent of each learner's fit time over the training-set size (time ~ n^b, from
  2,000 examples on). `demos/multiclass_decomposition.py`: PFoil as a third base
  learner, and a section that sets each base learner's measured growth against the
  pairwise speed-up it predicts.

### Changed

- `run_cv(..., stop_on_timeout=True)` (the new default): once a fold of a
  (dataset, learner) times out, the whole cross-validation of that pair counts
  as timed out -- its remaining folds are not run, and all its folds, the
  finished ones included, come back as `error="timeout"` with NaN measures, so
  no mean rests on just the folds that happened to finish. Only timeouts. The
  cache keeps each fold's real outcome. `stop_on_timeout=False` restores the
  old behaviour.
- `experiments.runner.timed_out(results)`: the (dataset, learner) pairs of a
  `run_cv` table whose cross-validation timed out -- e.g. to drop a learner
  from the larger sizes of a growing-size run. A verbose `run_cv` ends with a
  line naming them. The multi-class demo's growth phase goes through each
  learner's subsamples smallest first and skips the larger ones after a
  timeout.

### Fixed

- A process using `TimeoutRunner` (so `run_cv`) could hang at exit after a
  worker was killed before it had read its task: the queue's feeder thread
  stayed blocked on the full pipe, and Python joins it at exit. The old
  queues are now closed when a worker is killed.
- `NListRepresentation.select_rows` (and `PrePostNListRepresentation`'s) derives the
  subset's tree from the existing one instead of building it anew: the nodes no
  selected row passes through are dropped and the rest renumbered, keeping the
  full data's feature order. Coverage, rows and counts are those of a rebuilt
  N-list (weighted sums up to floating-point rounding). 5-8x faster on the
  multi-class datasets measured (the pairs of `letter`: 21.3 s -> 3.1 s), which
  is what pairwise and ordered decompositions spend most of their time on.
  `relabel` shares the tree and only rebuilds the per-class sums. An integer
  index that reorders or repeats rows still builds anew.
- N-lists count candidate features in one batch (`batch_cover_sums`,
  `batch_cover_counts` -- ENDER, Boomer, ORB) with a dense product over the
  distinct row patterns when at least 2% of the pattern matrix is true (and it
  has at most 2^26 cells), the sparse product otherwise. On real data with
  negation features the boosting learners now run about as fast on N-lists as
  on Boolean data (credit-g: ENDER 1.68 s -> 1.25 s, Boolean 1.18 s; Boomer
  3.08 s -> 2.23 s, faster than Boolean), while wide sparse data keeps the
  sparse product (20,000 features: unchanged). The dense patterns cost one byte
  per cell, built on first use, and are converted to floats a block of at most
  2^18 cells at a time, so the product's temporary copy stays small (peak memory
  of a CN2 fit on 8,000 rows of adult: 12.1 MiB before, 3.1 MiB now).

## 0.3.2 (2026-10-15)

Parallel experiments and consistent numbers: `run_cv` takes `n_jobs`, every demo that fits through it takes
`--n-jobs`, and all demos were re-run on 0.3.1's N-lists. Accuracies are shown as percentages and counts with
one decimal throughout. No change to what any learner learns.

### Added

- `run_cv(..., n_jobs=N)` fits `N` cells (dataset, fold, learner) at a time, each in its own
  `TimeoutRunner` worker process. The table has the same rows, in the same order, with the
  same accuracies and rule counts as with `n_jobs=1`; the default is 1, so existing
  behaviour is unchanged. The `fit_time` of a parallel run is measured inside the worker and
  is skewed by the concurrent load, so it should not be compared across runs -- use it for
  accuracy-, size- and count-based comparisons, not for timing reports.
  A request for more workers than the cores available to the process is capped at that
  number, with a `RuntimeWarning`; `n_jobs="10!"` (a string ending in `!`) starts that many
  workers regardless.
- `pyrulearn.evaluation.format_accuracy` and `format_mean_count`: the one way
  pyrulearn shows numbers -- accuracies as percentages with two decimals
  (`86.31%`), means of counts (rules, conditions, conditions per rule) with
  one decimal (`12.3`). `ModelStats`' repr, `run_cv`'s progress lines and
  `experiments.report.render_results_table` use them (accuracy columns
  print as percentages, `n_rules`, `n_conditions`, `conds_per_rule`,
  `conditions/rule`, `n_features` and `overlap` with one decimal).
- `JavaLord(thread_count=...)` and `run_lord(thread_count=...)`: limit the
  threads of the reference LORD, which uses half the machine's cores by
  default. `None` (the default) keeps that.
- Demos: `--n-jobs N` for every demo that fits through `run_cv`;
  `covering_boosting` and `pool_distillers` now fit through `run_cv` too (the
  latter in two passes: first the pools, then the distillers on them).

### Changed

- Every demo and its report shows accuracies as percentages and counts with
  one decimal, and (except `covering`, `heuristic_isometrics` and
  `numeric_discretization`, which use their own data setup) runs on the
  library's default representation, N-lists. `dense_vs_modular` compares
  the dense learners on the Boolean matrix with the modular ones on N-lists,
  and has a separate section for the modular learners on Boolean data.
- `native_vs_interfaced` runs `MLRL:Boomer` and `seco_learners_comparison`
  the reference LORD on one thread, as every native learner does (in these
  demos only; the learners themselves keep their defaults).
- `pool_distillers` seeds both IDS variants, so a run is reproducible.
- `covering_boosting`: the legends of the accuracy-vs-fit-time plot sit outside
  the axes; the report no longer fails when no fit timed out.

## 0.3.1 (2026-10-14)

A memory-efficiency release: the N-list and sparse representations hold
much less, and the representations demo measures memory -- held, peak
while building, and what a fit costs in total. No API changes; every
learner learns the same rules as before.

### Changed

- `NListRepresentation` stores the tree *instead of* the table: per node
  the path bitmask and its row range (the rows are kept in tree order, so
  the rows through a node are one contiguous range), per row its place in
  that order and its end node. `features_of` and `X` are decoded from the
  tree; class counts are differences of cumulative sums. Built in numpy,
  without Python node objects. `adult`, 32,000 rows: 27.9 -> 4.2 MiB held
  without negation features, 136.6 -> 37.7 MiB with; built 6x faster,
  with a lower peak. Fits are as fast or faster for most learners;
  coverage-heavy ones (Pypper) up to ~15% slower.
- `PrePostNListRepresentation` shares that storage and only keeps the
  nodes' pre/post codes for its search.
- `SparseDataRepresentation` and the N-lists are built from the packed
  bits (`HorizontalBitmapRepresentation.to_csr()`, new), no longer leaving the
  unpacked matrix cached on the Boolean source: `Sparse` 10.2 -> 5.9 MiB
  held, peak while building no higher than what it holds.
- `select_rows`, `relabel` and `without_negations` on N-lists decode to a
  sparse matrix instead of caching a dense `X`.

### Added

- `HorizontalBitmapRepresentation.to_csr()`: the matrix as an `int32` CSR,
  unpacked a block of rows at a time.
- `demos/representations.py`: memory held by each representation and its
  peak while building, the memory a fit costs in total, and a width sweep
  (a fixed number of true features per example, up to 20,000 features)
  showing where sparse storage pays off. README: which representation to
  use -- N-lists for speed (the default), Boolean for dense and normal
  data when memory matters, Sparse for sparse data.
- Demos read the Weka and LORD locations from `$WEKA_JAR`, `$WEKA_JAVA`
  and `$LORD_DIR`; `tools/run_all_demos.sh` runs every full demo in turn.

## 0.3.0 (2026-10-13)

A modular release: the rule learners are built from exchangeable
components (searches, objectives, covering and boosting loops) that run on
every data representation. The previous hand-written versions stay,
under `Dense*` names, for speed. API changes are listed under *Changed*.

### Added

- `Objective`: what a rule search maximizes -- statistics, batch scoring,
  bounds and the "same", "dead" and "consistent" predicates. Searches
  (`BeamSearch`, `HillClimbing`, `GainAscentHillClimbing`,
  `BranchAndBoundSearch`) talk only to an objective; a rule heuristic is
  one (`as_objective`). Further objectives: `XGBGain` (ORB) and `LRIError`
  (LRI).
- `ORB` (optimal rule boosting) and `LRI` (lightweight rule induction) on
  the rule searches, taking any `RuleSearch`; `ORB` defaults to
  `BeamSearch(beam_width=10)` instead of exhaustive search. `MarginLoss`
  objects for ORB's loss.
- `CPAR` and `ENDER` (with `Boomer`) built from the SeCo components;
  `CPAR` is a configuration of the SeCo learner, and a search can yield
  several rules.
- OPUS: `BranchAndBoundSearch` with dominance pruning, `Opus` and
  `OpusTopK`, the productive-rule filter, and Laplace and sibling pruning
  (Webb 1995; the OPUS references are in `references.bib`).
- `Dense*` learners: `DenseCPAR`, `DenseENDER`, `DenseBoomer`,
  `DenseORB`, `DenseLRI` -- the earlier dense-matrix implementations,
  kept for speed (2-4x on small Boolean data) and as references.
- `encode()`: like `binarize`, but sparse throughout, never densifies; the
  loaders and `run_cv` use the preferred representation
  (`default_representation()`, `REPRESENTATION_PREFERENCE_ORDER`, N-list
  first). External learners convert to Boolean themselves.
- Faster counting: numeric thresholds in one pass on
  `HorizontalBitmapRepresentation`, batched counting on N-lists and sparse
  data, one `batch_score` call per node, constraint closures passed down.
- Demos: `dense_vs_modular` (every `Dense*` learner against its modular
  twin, ORB and LRI with beam widths 5 and 10, exhaustive ORB on tiny
  datasets); Opus in the SeCo comparison.

### Changed

- Learners are seeded by default (`random_state=0`).
- Components are passed as objects, not names: `search=`,
  `heuristic=` and the like take a `RuleSearch` or `Objective` (`CN2` and
  `PFossil` lose `mode=`, AQR loses `seed_strategy=`; `ORB(search=...)`
  takes a search object instead of `"exhaustive"`/`"greedy"`).
- `CPAR`, `ENDER` and `ORB` are the component-built versions; the old
  ones are `DenseCPAR`, `DenseENDER` and `DenseORB`. `LRI` likewise, with
  `DenseLRI` the old one; `LRI`'s `max_length`, `power` and `max_errors`
  moved onto the search and `LRIReweighting`.
- Pretty printing shows one condition per attribute (a value set as
  `in`/`not in`, numeric ranges as intervals).
- A search never takes a condition that changes nothing as an improvement.

## 0.2.2 (2026-10-07)

A feature release: shared cross-validation infrastructure for demos and
other experiments (`pyrulearn.experiments`), plus the fixes below.

### Added

- `pyrulearn.learners.pylord.pylord_candidates(data)`: LORD's raw
  per-example local-search candidate pool, exposed directly -- e.g. for
  use as a rule pool with `RuleFit` or another distiller -- instead of
  only LORD's own coverage-filtered result.
- `experiments.runner.TimeoutRunner.run(fn, *args, timeout=None,
  **kwargs)`: a per-call override of the runner's timeout, so one
  persistent worker process can serve calls that need different budgets
  (e.g. a longer one for building a rule pool than for fitting a
  distiller) without running two `TimeoutRunner`s at once -- two
  concurrent `spawn`-context worker processes were found to corrupt
  each other's pipe handles on Windows.
- `pyrulearn.experiments`: shared infrastructure for demos and other
  cross-validation-style experiments, replacing the loading/splitting/
  timeout/measurement/report code each `examples/demo_*.py` script
  otherwise reimplements.
  - `experiments.catalog`: a catalog of 100 benchmark datasets on OpenML
    (OpenML-CC18, the LORD evaluation's datasets, classic XAI and
    rule-learning datasets) plus a few pointer-only entries, metadata and
    download pointers only. Select by
    task (binary/multiclass), size (small/medium/large), attribute types
    (categorical/numeric/mixed), tags or name -- `Catalog.default().select(...)`
    or `.parse("binary,small,medium")`, optionally picking `n` of the matches
    at random -- and `entry.load()` downloads and
    caches the raw data with curated fixes applied (restored attribute
    names, dropped leakage columns, category codes, missing-value markers).
    `tools/build_catalog.py` maintains the catalog file.
  - `experiments.runner.run_cv(learners, datasets, n_folds=10, ...)`: the
    cross-validation loop -- per-fold binarization (`build_dataspec`/
    `binarize` on the training split only, test split binarized against
    that fold's `DataSpec`), a per-fit timeout (`TimeoutRunner`, a
    persistent worker subprocess so one hung fit can't block the run),
    uniform measurement via `RuleModel.evaluate` plus whatever a
    `measure_fn` adds, and optional fold caching (`cache_dir`) so
    resuming after a crash, or adding one learner, doesn't refit
    everything. Returns one long-format `pandas.DataFrame`. `verbose`
    (default on) prints live per-dataset and per-(fold, learner)
    progress, flushed immediately, so a long run's state is visible even
    piped to a file.
  - `experiments.stats`: optional, separately-callable statistics over a
    `run_cv` table -- `mean_rank`, `win_counts` (per-learner count of
    datasets it was (tied-for-)best on), `friedman_test`
    (`scipy.stats.friedmanchisquare`), `critical_difference_diagram`
    (hand-rolled Nemenyi/Demsar (2006) formula, `matplotlib`). Never run
    automatically by `run_cv`.
  - `experiments.report`: small Markdown-writing building blocks
    (`render_setup_section`, `render_results_table`). `render_results_table`'s
    `group_by` also takes a list of columns (e.g. `["dataset", "learner"]`,
    for a per-dataset table that still breaks out each algorithm -- a
    single `group_by` averages every learner together within each row,
    silently erasing the comparison); `include_overall=False` drops its
    trailing "overall" row for a `group_by` that mixes different
    algorithms together, where averaging across them isn't meaningful.
  - New `experiments` extras group (`pandas`, `scipy`, `matplotlib`).
- `pyrulearn.data.io.build_dataspec`: a new `skip_unusable` flag leaves
  out a column with no useful split (rather than raising) -- for
  automated runs like `run_cv` where a column can turn constant or
  entirely missing within one training fold.

### Changed

- `Rule.specialize` computes the parent rule's constraint closure once and
  extends it per child (new `DataSpec.extend_closure`), re-checking only
  the constraints that involve the added feature instead of propagating
  every child's conditions from scratch. Same result, 3-4x faster rule
  search on data with many numeric thresholds.

### Fixed

- `J48Importer` raised "no leaves found in J48 tree text" on a degenerate
  single-leaf tree (no attribute predictive enough to split on, so J48
  predicts the majority class unconditionally) -- Weka omits the blank
  line it otherwise prints between the tree header's divider and the
  tree body for exactly this one case, which a hardcoded line offset
  didn't account for.
- `J48Importer` and `PARTImporter` silently dropped every leaf/rule whose
  class label contains a space (e.g. glass's `build wind float`): the
  target was matched as a single whitespace-free token.
- `BeamSearch` and `HillClimbing` could pick a candidate rule covering no
  positive examples when the heuristic scored it well (Laplace gives an
  empty-coverage rule 0.5, Accuracy N/(P+N)). Such a rule can't be refined,
  so the search ended early with an overly general rule, and covering
  stopped long before the positives were covered. Candidates without
  positive coverage are now never chosen or returned, and their condition
  is masked out of the search below the same parent. This changes the
  rules learned with Laplace, Accuracy or cost heuristics -- including
  `CN2`, which now learns larger theories. Learners that did not hit this
  (e.g. `Pypper`, `PFossil`, `PFoil`, `AQR`) learn the same rules as before.

## 0.2.1 (2026-09-27)

A bug-fix release: `RuleFit` fitted dense models with scikit-learn before
1.8 (so on Python 3.10), and the abstain pseudo-label printed as an object
address. Also renames `MlrlBoomer` to `MLRLBoomer` (the old name still
works, with a warning).

### Changed (may break imports)

- `MlrlBoomer` is renamed `MLRLBoomer` (tool prefix `MLRL`); the old name
  still works, with a `DeprecationWarning`.

### Fixed

- `RuleFit` fitted dense (L2) models with scikit-learn before 1.8 (e.g. on
  Python 3.10, where 1.8 isn't available): those versions use `l1_ratio`
  only with an explicit `penalty`, which `RuleFit` now passes there.
- The `ABSTAIN` pseudo-label prints as `ABSTAIN` instead of `<object object
  at 0x...>`, which also changed with every run.

## 0.2.0 (2026-09-27)

A feature release, with breaking changes. Rule statistics are redesigned: a
rule's training statistics are frozen when set, weights are reserved for
explicitly fitted or user-set values, and printed models state how they
resolve conflicts. External learners are renamed with a short prefix for
their tool (the old names still work, with a warning). New: row weights on
all data representations and a pluggable weighted covering framework;
the rule ensembles SLIPPER, LRI, CPAR, ENDER (with MLRules), BOOMER and
optimal rule boosting; a RuleFit-style distiller with `LinearRuleModel`; and
interfaces to imodels' RuleFit and Slipper, mlrl-boomer and realkd.

### Changed (may break imports: external learners renamed)

- Learners that run an external tool now carry a short prefix for the
  tool: `SKLDecisionTree`, `SKLRandomForest`, `WittIREP`, `WittRIPPER`
  (was `RIPPERk`), `IModBayesianRuleList`, `IModBayesianRuleSet`,
  `IModRuleFit`, `IModSlipper`, `WekaJRip`, `WekaPART`, `WekaJ48`,
  `PArcCBA`, `RKDRuleBoosting`, and `JavaLord` for the reference LORD
  implementation (was `LordJar`); native learners keep plain names. The old
  names still work, with a `DeprecationWarning`. Importers keep their
  names. `RuleLearner.display_name` gives the name for results tables
  (`"IMod:Slipper"`, `"CN2"`); a learner's `TOOL` holds its prefix.

### Changed (may break code: statistics redesigned)

- Only rules store measurements: a rule's training statistics,
  `rule.stats()`, set once where it is produced (`fit`, an importer's
  `data=`, `annotate_rules`, or the new `SingleRule.set_stats(data)`).
  They are frozen: they are part of the model (combiners score from them,
  printing shows them), so setting them again raises instead of silently
  changing what the model predicts. `SingleRule.reset_stats(data)`,
  `annotate_rules(..., reset=True)` and `set_stats_from_counts(..., reset=True)`
  replace them deliberately. `annotate_rules(..., copy=True)` annotates
  fresh copies instead; the distillers (`CBA`, `IDS`) now use it, so they
  no longer re-annotate the shared rule pool's rules in place.
- New `RuleModel.evaluate(data)` measures any model (or rule) on any data
  and returns a `ModelStats` without storing it. It replaces
  `RuleModel.annotate`, `stats(data)` and the `split=` names, which are
  gone: containers no longer store snapshots of their own performance
  (nothing read them), and `RuleSetClassifier.fit` no longer computes an
  unused one.
- `to_string` no longer takes `data=`: it prints each rule's own stored
  statistics (`(tp/fp)`, or the class distribution for distribution
  combiners), which are exactly the numbers the model uses, and does so by
  default whenever rules carry statistics. `show_stats=False` prints the
  bare rules.
- `remap` (and `filter`, and model conversions) keep the rules' statistics,
  the default rule's included; `remap` used to drop them.

### Changed (may change results or printouts)

- Printed models whose rules predict more than one class now start with a
  `% conflict resolution: ...` line naming how conflicts are decided (e.g.
  `max Laplace`, `sum of covered class counts`, `first matching rule`),
  from the new `RuleCombiner.describe()`; `to_string(show_resolution=False)`
  omits it. Rule heuristics got a readable `repr` (`Laplace`,
  `MEstimate(m=5)`) for it.
- `print(model)` now shows the full rendering (`to_string()`) for every
  model, rules included; the short `repr` identifies it --
  `FlatRuleSet(3 rules)`, and for a rule `SingleRule(2 conditions)`
  instead of `SingleRule(1 rules)`.
- The default rule of a decomposed model (`ConceptSet`, `ConceptCascade`,
  `PairwiseModel`, as built by one-vs-rest, ordered or pairwise fitting)
  now carries training statistics like every other default rule: its
  coverage of the entire training data, i.e. the class distribution.
- Pairwise voting: `WeightedVote(heuristic=...)` makes the deciding rule's
  weight configurable (default `Laplace`, as before), and a printed
  `PairwiseModel` names its combination (`% conflict resolution: pairwise
  vote weighted by Laplace of each pair's deciding rule`). Vote ties go to
  the tied labels' duel, then training frequency (the recorded priors,
  else read from the rules' stats), then label order.
- Ties no longer depend on rule position. `max` first lets the tied
  top-scoring rules vote; any remaining tie, for every combiner, goes to
  the class more frequent in the training data, then to the one that
  sorts first. (Before, `max`, `vote` and the distribution combiners fell
  back on rule or class insertion order.) Only `"list"` is order-based.
  `EnsembleModel` vote ties follow the same convention instead of member
  order, and an ensemble prints `% conflict resolution: (weighted) vote of
  members`.

### Added

- Row weights on every data representation: `weights=` in the
  constructors and `with_weights(w)` (a copy sharing the storage). Every
  count becomes a sum of weights -- the searches, `RuleStats`, Pypper's
  description length, `evaluate` and the rules' stored and printed
  training stats. Integer weights give exactly the results of duplicated
  rows; without weights nothing changes. `RuleStats` counts may now be
  floats. Mined rule pools (`CARMiner`) don't use weights yet.
- Weighted covering: `SeCo(covering=...)` with `RemovalCovering` (the
  default, as before) or `WeightedCovering(reweighting, stop)`, which
  reweights covered examples instead of removing them. Reweighting
  schemes: `MultiplicativeReweighting` (CN2-SD, CPAR's decay),
  `AdditiveReweighting` (CN2-SD), `AdaBoostReweighting` (Slipper) and
  `LRIReweighting` (Lightweight Rule Induction). Stop criteria:
  `CoveredAtLeast`, `PositiveWeightBelow`, `Rounds`. The loop's state is
  a `CoveringState`. Also accepted by `CN2`, `AQR`, `PFoil` and `PFossil`.
  A search scope (`example_mask`) may now be a weight vector.
- `Slipper` (`pyrulearn.learners.boosting`): SLIPPER (Cohen & Singer
  1999), confidence-rated boosting of rules, as a `LinearRuleModel`; and
  its rule-growing heuristic `SlipperZ` (`sqrt(tp) - sqrt(fp)`).
- `ENDER` (`pyrulearn.learners.boosting`): boosting of rules by forward
  stagewise loss minimization (Dembczyński, Kotłowski & Słowiński; ENDER
  2010, MLRules 2008) -- rules voting for one class, grown on subsamples
  to minimize an impurity derived from a pluggable `BoostingLoss`
  (`LogisticLoss`, the default, multiclass; `ExponentialLoss`;
  `SigmoidLoss`) by one of the paper's techniques (constant-step, the
  default, with `beta`; gradient descent; gradient boosting; simultaneous
  minimization) or MLRules' Newton criterion, weighted by the loss's
  response on all rows, shrunk; optional holdout early stopping; a
  `LinearRuleModel`. Defaults are the paper's CS-Log setting.
- `references.bib`: entries for RuleFit, SLIPPER, confidence-rated
  boosting, LRI, CPAR, MLRules, ENDER, BOOMER and optimal rule boosting.
- `ENDER(l2_regularization=...)`: an L2 penalty on the Newton steps, as in
  BOOMER; and `Boomer` (`pyrulearn.learners.boosting`), ENDER with BOOMER's
  defaults -- single-label BOOMER, predicting like `mlrl-boomer` on binary
  data. Multi-label learning, with preference learning and label ranking,
  is now listed under *Not yet implemented*.
- `OptimalRuleBoosting` (`pyrulearn.learners.boosting`): rule boosting with
  rules that maximize the XGBoost-style gain, found by branch-and-bound
  (Boley et al. 2021) or greedily; reproduces `realkd` exactly.
- `RealkdImporter` / `RealkdRuleBoosting` (`pyrulearn.interfaces.realkd`):
  `realkd`'s rule boosting, the reference implementation of optimal rule
  boosting, as a `LinearRuleModel`.
- `BoomerImporter` / `MlrlBoomer` (`pyrulearn.interfaces.boomer`): BOOMER
  (`mlrl-boomer`) for binary classification, as a `LinearRuleModel`
  reproducing its decision function. New extras `boomer` and `realkd`.
- `CPAR` (`pyrulearn.learners.cpar`): Classification based on Predictive
  Association Rules (Yin & Han 2003) -- FOIL-gain rule growing on
  weighted examples with rule copying at nearly-as-good conditions,
  weight decay of covered positives, prediction by the best `k` rules
  per class.
- `TopKMeanCombiner` (`pyrulearn.combiners`): the class whose best `k`
  covering rules have the highest mean heuristic score wins (CPAR's
  prediction).
- `LRI` (`pyrulearn.learners.lri`): Lightweight Rule Induction (Weiss &
  Indurkhya 2000) -- per class the same number of unweighted DNF rules
  (`ConceptModel`s), grown without pruning on cases reweighted by the
  rules' cumulative errors, voted in an `EnsembleModel`.
  `LRIReweighting` halves the error counts once one exceeds 32, as in
  the paper.
- `SlipperImporter` and `ImodelsSlipper` (`pyrulearn.interfaces.imodels`):
  `imodels.SlipperClassifier` (in fact scikit-learn's AdaBoost over
  imodels' rule learner) as an exact `LinearRuleModel`. With labels other
  than 0/1, imodels' own `predict` always returns the first class (a bug);
  `ImodelsSlipper` fits on 0/1 labels to avoid it.
- `RuleFit` (`pyrulearn.learners.rulefit`), a RuleFit-style rule
  distiller: a sparse (L1, or elastic-net with `l1_ratio`) logistic
  regression over the coverage of a rule pool (`rules=`, or mined like
  CBA's), multinomial for more than two classes. `C` sets the sparsity,
  `cv=` chooses it by cross-validation, and `include_features=True` adds
  the single features as candidates. Binary pure-L1 fits use the fast
  `liblinear` solver, everything else `saga` (`solver=` overrides).
  Needs scikit-learn.
- `LinearRuleModel`, the model `RuleFit` returns: rules with signed
  weights, an empty-body rule per class as its intercept, and the class
  with the highest summed weight wins (`WeightedSum` resolution, printed
  as `% conflict resolution: sum of rule weights per class, highest
  wins`). `scores(data)` gives the per-class sums.
- `RuleFitImporter` and `ImodelsRuleFit` (`pyrulearn.interfaces.imodels`):
  `imodels.RuleFitClassifier` imported exactly as a `LinearRuleModel`.
  imodels predicts the positive class only where the logistic output is
  above 0.5 instead of 0 (a bug in its `predict_proba`); the import
  reproduces that by default, as a visible `0.5::<negative class>(X) :-
  true.` rule, and `imodels_threshold=False` gives the logistic decision.
- `rulefit_candidates(data)` (`pyrulearn.interfaces.imodels`): RuleFit's
  tree-based candidate rules as a pool, e.g. for
  `RuleFit(rules=rulefit_candidates(data))`.
- `examples/demo_rulefit_comparison.py`: imodels' RuleFit vs. the native
  one, swapping the candidate step and the fitting step separately.
- `to_string(weight_format=...)` (rules and every model): a Python format
  spec for rule weights, e.g. `"6.2f"` to line them up.
- `to_string(pretty=True)` (rules and every model): each condition on its
  own line, indented under the head, and the coverage comment on its own
  line above the head. Prolog format only so far.
### Fixed

- In the `"conditions"` and `"pattern"` formats, models now show the class
  a rule predicts where it was missing: decision lists prefix each rule
  with it (`z: ¬f0, ¬f1`), and every model's default section names it
  (`% default: x`). Before, a decision list printed in these formats
  didn't say which class any rule predicted.

## 0.1.3 (2026-09-25)

A bug-fix release: binary attributes and missing values handled
correctly, AQR's covering fixed, and several printing fixes.

### Changed (may change results or printouts)

- `Rule.to_string("prolog")` quotes a head that isn't a valid bare Prolog
  atom (`'1'(X) :- ...`, `'Yes'(X) :- ...`, `'no-recurrence-events'(X) :- ...`),
  and gives each condition that introduces a value its own variable
  (`age(X, V1), V1 < 30, income(X, V2), V2 >= 50000`) instead of reusing
  one `V`, which forced unrelated attributes to unify.
- `tree_thresholds` (decision-tree discretization) rounds each split
  point to the coarsest value that still lies strictly between the two
  neighbouring observed values -- `15.17` instead of `15.172899999999998`.
  Display only: every row stays on the same side of the split.

- `AQR().fit(data)` now returns a `DecisionList` instead of a
  `FlatRuleSet` resolved by `combiner="list"`. Predictions are identical
  (the first covering rule in learn order wins either way), but the
  model now prints in that deciding order. `fit(data, model=FlatRuleSet)`
  still gives the list-resolved rule set, e.g. to compare combiners.
- A rule set whose own combiner is `"list"` and whose rules have more
  than one head now prints in list order, like a `DecisionList`, instead
  of grouped by label, which hid the order that decides.

### Changed (may break imports)

- `pyrulearn.attributes` moved to `pyrulearn.data.attributes`, next to
  `data.spec`, which builds a `DataSpec` from it: attribute types,
  `FeatureSpec`, the feature constraints, `evaluate_feature` and
  `MissingStrategy` describe the feature space, not rules. The old module
  path no longer exists.

### Added

- Binary attributes: `AttributeType.BINARY` and
  `DataSpecBuilder.add_binary("sex", ["male", "female"])`, a closed set of
  exactly two values. They always get both features (`sex=male`,
  `sex=female`), each the other's negation, and never `!=` columns; a
  value outside the two counts as missing. This sits between `BOOLEAN`
  (one tested value, `smoker` / `not smoker`, unchanged) and `NOMINAL`
  (a possibly incomplete value list, which keeps its `!=` columns even
  for two values, since an unknown value must satisfy every `x!=v`).
- `build_dataspec(domains=...)`: declared value lists per column.
- `merge_dataspecs` merges two binary attributes over the same values as
  binary, and a binary one with a nominal one (or with a binary one over
  different values) as the nominal union. `Rule.remap` turns `x!=a` into
  `x=b` when the target attribute is binary over `{a, b}`.
- `demos/covering.py`: a step-by-step separate-and-conquer teaching
  demo (PFossil's covering and hill climbing replayed on Titanic, one set
  of plots per search step).

### Changed (may change feature counts)

- `build_dataspec`, `read_csv` and `read_arff` infer a column with
  exactly two values -- declared in an ARFF header, else observed in the
  data -- as binary instead of nominal: with negation on, it gets 2
  columns instead of 4 (e.g. kr-vs-kp's 144 columns become 76). Importers
  that infer a `DataSpec` from a rule set (Weka, LORD, wittgenstein) keep
  nominal attributes, since a rule set need not mention every value.

### Fixed

- `AQR` with a target class (and in its one-vs-rest, cascade and pairwise
  decompositions) stopped learning for a class as soon as the example it
  seeded on admitted no acceptable rule -- typically a noisy example --
  leaving every other positive uncovered: with 3% label noise it often
  learned no rule at all. A failed seed is now dropped and the next
  uncovered positive tried; learning stops only once every positive is
  covered or dropped. Learners that search all rules at once (CN2, PFoil,
  PFossil) are unchanged.
- `read_arff` turned ARFF's missing marker `?` into an ordinary category
  (`sex=?`); it is now a missing value. Nominal attributes also take their
  values from the header's declared list rather than only the values
  that occur in the data.
- `examples/demo_seco_learners_comparison.py`: `lord` failed on every fold.
  Its rules were parsed against a dataspec inferred from LORD's own output,
  whose feature numbering didn't match the training data; they now bind to
  the real dataspec by column position, as `LordJar` already did.
- `examples/demo_workflow_comparison.py` and
  `demo_seco_learners_comparison.py`: pyrulearn's own binarization now gets
  the raw data with real missing values (handled by
  `MissingStrategy.NEVER_COVERS`) instead of median/"?"-imputed data, and
  Pima diabetes's zero-encoded measurements are treated as missing. Only
  the external learners still get imputed data.

## 0.1.2 (2026-09-22)

First step toward the weight-transparency framework: none of this touches
`WeightedRule` or adds any new declarative-weight storage yet, but a printed
model's coverage decoration is now meant to let a reader manually derive
`predict()`'s decision from the page, for whatever mechanism a model's own
combiner actually uses.

### Changed (may break `to_string(data=...)` output)

- Coverage decoration: `(n_covered/n_errors)` is now `(tp/fp)`, and the
  `[n_unique_covered/n_unique_errors]` bracket is gone entirely -- "unique"
  meant something different per resolution strategy (no other same-target
  rule / no earlier rule / ...), so it wasn't a stable concept to keep
  printing uniformly.
- A model whose own `combiner` is genuinely a `DistributionCombiner`
  (`micro`/`macro` × `vote`/`max`) with more than two classes now prints the
  full per-class coverage breakdown instead of `(tp/fp)` -- everything that
  combiner's `resolve()` reads, nothing it doesn't -- with a one-line
  `% classes: [...]` legend printed once above the output. Two classes make
  the vector exactly `(tp/fp)` reordered, so it stays suppressed there.
- `CN2`'s default `ConceptSet` now resolves a clash between covering rules
  by summing their covered-class distributions (`MicroVoteCombiner`), not
  the `SeCo` family's generic `combiner="max"` -- matching Clark & Boswell's
  own unordered-CN2 algorithm. `ConceptSet`'s own default, and every other
  `SeCo` member, are unchanged.

### Added

- `to_string`'s new `show_distribution`/`show_classes` (both `None` by
  default) let a caller force either the vector or the legend on or off
  independently, for any model, any class count, any combiner.
- `EnsembleModel`/`PairwiseModel` gained `to_string` -- calling it used to
  raise `AttributeError`. `PairwiseModel` forces each pair's own two-class
  legend on by default and narrows the printed data to that pair's own rows;
  `DeepModel` still has none, being a structure-only stub.
- `ClassAssociationRuleMiner` renamed to `CARMiner` (shorter; it was the
  widest cell in several README tables).

### Documentation

- README table columns: the common `pyrulearn.*`/`pyrulearn.learners`/
  `pyrulearn.interfaces` prefix repeated in every cell is now given once in
  the header instead.
- The "Not yet implemented" list consolidated from about nine granular
  bullets into five themed ones, folding in a planned `pyIDS` cross-check
  and a planned `RuleFit`-style distiller.

## 0.1.1 (2026-09-21)

### Changed (may break imports)

- The native RIPPER re-implementation is now called `Pypper`
  (`pyrulearn.learners.seco.Pypper`); the `RIPPER` class and the old `Pypper`
  alias are gone. The external wittgenstein `RIPPERk` and Weka `JRip` wrappers
  are unchanged.
- `CBA` and `CMAR` moved into `pyrulearn.learners.associative` (next to
  `ClassAssociationRuleMiner` and `RuleDistiller`); the modules
  `pyrulearn.learners.cba` and `pyrulearn.learners.cmar` no longer exist.
  `IDS` stays in `pyrulearn.learners.ids`.
- LORD is now credited to Huynh, Fürnkranz & Beck (2023), including the
  `LORDImporter` provenance source string, which changed to
  `"LORD (Huynh, Fürnkranz & Beck, 2023)"`.

### Fixed

- `rule_refinement_plot` axis labels showed `np.str_('pos')` instead of `'pos'`;
  the start label no longer collides with the plot title.
- `examples/demo_workflow_comparison.py` ran into three problems after the model
  hierarchy refactor: the forest passed `combiner=` to `EnsembleModel.predict`,
  the gitignored ARFF scratch directory was never created, and the Original
  forest lost its leaf statistics through `Rule.remap`.

### Documentation

- README restructured: separate sections for data representation and rule
  models, conflict resolution merged with the combiner descriptions, heuristics
  next to statistics, a coverage-space section with pencil / parallel / curved
  isometrics, learners before external interfaces, and a table of what each
  external tool receives. The module map now has one row per package.
- `references.bib` gains *Foundations of Rule Learning* (Fürnkranz, Gamberger &
  Lavrač, 2012) and the corrected LORD authors.
- `examples/demo.py` no longer writes the mixed-target `coverage_space.png`; all
  demo reports and plots were regenerated.

## 0.1.0

- Initial public release.
