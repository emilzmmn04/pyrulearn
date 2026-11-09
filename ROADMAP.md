# Development roadmap

Plans for pyrulearn's development: what is intended for the next
releases, and why. The list of missing features for users is in the
README (*Not yet implemented*); the overhaul of the demos has its own
plan in [`examples/REVISION_PLAN.md`](examples/REVISION_PLAN.md).

## 0.4.0: numeric features in both directions; ordinal attributes

**Done in 0.4.0 (2026-10-20).** Two changes to the feature space,
released together with the beta clean-up (removed deprecated names).

*Numeric attributes always test both directions.* A threshold `t` gives
`x>=t` and `x<t`, whatever the negation setting: one direction alone
makes no sense, since a rule could then only bound an attribute from one
side. Until 0.3.4, `x<t` existed only with negation on, so the "without
negation" encoding (`include_negations=False`, `without_negations()`) had
`>=` only. The same for the `le_thresholds` family: `x<=t` and `x>t`.
Negation now means `x!=v` (nominal, ordinal) and `not f` (Boolean) only;
`<`/`>` are complements of their partners but no longer negation
features. `add_numeric` loses its `negation=` parameter.

*A new attribute type, ORDINAL*, for ordered values (`low < medium <
high`) and small sets of numbers (counts, ratings, integer codes). For
values `v1 < ... < vk`: `x=vi` for every value, `x<=vi` and `x>vi` for
every value but the last, and with negation `x!=vi`. Equality and
inequality on such values are not expressible with thresholds alone
(`x!=v` would be a disjunction). One `OrdinalGroup` constraint over all
of them, on value positions: every feature is true on a set of positions,
and propagation intersects them. Values compare by their position in the
declared order, so non-numeric values work; an all-numeric domain
compares numerically, so an unseen value still satisfies the right
`<=`/`>` tests. The positions are the known values only, like a nominal
attribute's (decided 2026-10-20): on the training data, `x>1` and
`x<=3` of 0, 1, 3 are `x=3`, and a search need not try `x=3` after them.
An unseen number in between (2, in a test set) is covered correctly,
but what the group derives (`x=3`) does not hold for it; that only
affects implied conditions and their display. Likewise for nominal
attributes, where `x!=red` and `x!=green` show as `x=blue`.

- Inference: a numeric column with few distinct values (the CSV
  heuristic that so far made it NOMINAL) becomes ORDINAL, from three
  values on; two values stay BINARY. Declared ARFF numeric columns stay
  NUMERIC. The bound is `max_ordinal_values=10`, configurable and
  deliberately independent of `max_intervals`: tying the two would make
  the attribute type change with the discretization, e.g. in the
  discretization demo, which varies `max_intervals`. This changes the
  encoding of such columns in `run_cv` and so in every demo; 24 of the
  small and medium catalog datasets have some (all 4 attributes of
  `balance-scale`, all 9 of `breast-w`, 9 of 41 in `qsar-biodeg`, ...).
- Missing values: every feature false by default, as for every type.
  `MAJORITY` imputes the median value (the lower of the two middle
  values, so an actual value); `SEPARATE` adds a missing feature as one
  more position, outside the order.
- Merging: numeric domains are unioned and sorted; non-numeric orders
  must agree.
- Display: `<=`/`>` merge into an interval, `!=` into a value set, as for
  numeric and nominal attributes.
- `run_cv`'s fold cache key gets the encoding (a version,
  `max_intervals`, `max_ordinal_values`), so caches from 0.3.x are not
  reused for the new encoding.

## Rules of any class: decision lists; comparing the rule models

**Done (2026-10-24), on branch `multiclass-heads`.** A search with no
target class looks for the best rule of any class
(`HeadObjective`): each rule is counted per class (one indicator column
per class through `batch_cover_sums`), and the heuristic chooses its head
(`RuleHeuristic.best_head`). The design keeps the effort for a new
heuristic at zero:

- **Default:** the heuristic scores the rule once per class, that class
  against the rest of the scope (`head_stats`), through its own
  `batch_score`; the best score wins (ties: more covered examples, then
  the first class). The covered majority is not always the best head:
  heuristics comparing with the prior (WRAcc, correlation) prefer the
  class whose share rises most, the m-estimate leans towards frequent
  classes.
- **`MajorityHead`** (a speed-up, same result): for heuristics that always
  prefer the covered majority -- `Precision`, `Laplace`, `GHeuristic`,
  `CoveredPositives`.
- **`ClassNeutral`**: for heuristics defined on the whole class
  distribution -- `Entropy`, which is now CN2's multi-class entropy there.
- **Gain heuristics** gain over the parent for the same head, so heads are
  compared only at the first refinement and the head then stays (without
  this, Pypper's decision list on `segment` stopped after 2 rules, a rare
  class showing large gains from 3 examples).

`fit(data, model=DecisionList)` is now CN2's ordered mode for CN2, PFoil,
PFossil, SeCo and Pypper; AQR keeps its seed covering. With a fixed target
class nothing changes: the search runs on `HeuristicObjective` as before
(all tests unchanged).

**Seeds and heads (2026-10-25).** A covering loop's rule comes from three
independent choices: the seed (`seed=`: `FirstSeed`, `RandomSeed`, none),
the search space (`space_init=`) and the head (`target_class=`: a class /
`FixedClass`, `HeadPerRefinement`, `HeadPerSeedExample`, `HeadPerRule`). A seed only sets the
defaults of the other two (its generalizations, its class), so AQ is
`seed=RandomSeed()` alone, and the unusual combinations need one override
(a seed with `HeadPerRefinement()`: the rule covers the seed but may predict
another class; a seed with `EmptyRuleAllFeatures()`: the seed only picks
the class). One decision-list loop (`SeCo._decision_list_fit`) replaces
the seed covering and the head covering; AQR's rules are identical to
before (`SeedExample`'s own strategies still choose the seeds when no
`seed=` is given). The defaults (which head `model=DecisionList` uses
without a `target_class`), decided 2026-10-26: `HeadPerRefinement()` -- the fastest,
and CN2's own ordered mode -- for every learner without a seed, CN2
included; `HeadPerSeedExample()` for a seeded learner (AQR). `HeadPerRule()` was the
best decision list on average in the rule-models demo, clearly so for CN2
(about three times the training time there), and CN2's docstring points
to it.

Still to do:

- **PyLORD on the same parts:** it seeds a search on every example
  (`EachExample`, not yet a `SeedSelection`) and takes its class; with
  `target_class=HeadPerRefinement()` its rules would get a free head too.

- **A demo comparing the rule models** (`demos/rule_models.py`, built
  2026-10-25; the full run is still to do), for the paper's section on rule
  models: rule sets (one-vs-rest; best rule -- an ordered rule set --,
  votes, class distributions), class cascades (rarest and most frequent
  first), decision lists (any class, each class, seed covering, the seed's
  class only) and pairwise, for CN2, PFossil and Pypper on 24 multi-class
  datasets, 8 two-class ones as a control. The quick run (5 datasets,
  3-fold) shows no model type ahead throughout.
- **Pairwise with a head per pair** (`PairwiseModel` with `positive=None`):
  each pair model a decision list over its two classes. Deferred; the
  pairwise vote schemes need a test with decision-list members.
- **CN2's significance test over all classes.** The ordered mode's test is
  the two-class one for the rule's head against the rest; the original
  uses the statistic over the whole class distribution.
- **Lexicographic scores (`LEF`)** have no single maximum over the heads
  and are refused for now.

## Count a numeric attribute's thresholds together

**Done for `HorizontalBitmapRepresentation` (2026-10-07); still open for the
other representations.** The rule searches (`BeamSearch`, `HillClimbing`,
and `GainAscentHillClimbing`, which shares `HillClimbing`'s loop) used to
score a candidate condition by counting the covered positives and
negatives separately for every candidate -- one `refine_cover` +
`cover_counts` pass per threshold of a numeric attribute.

A numeric attribute's thresholds form a chain: `x >= t_k` covers exactly
the examples in intervals `k` and above, `x < t_k` those below (reversed,
the same shape). `HorizontalBitmapRepresentation.chain_cover_counts` (approach
2 of the two once sketched here) counts a whole open run of one
attribute's chain in a single pass -- per-row count of satisfied
thresholds, histogrammed by class and summed from the top down -- instead
of one pass per threshold; `pyrulearn.learners.seco.score_children`
(shared by `BeamSearch`/`HillClimbing` until the next section) groups `rule.specialize`'s
children by chain and calls it, falling back to the original per-child
path for anything not groupable (a lone open threshold, a non-numeric
feature, or a representation without `chain_cover_counts` -- an opt-in
primitive, not a new required one, so N-list/PrePostNList/Sparse keep
working unchanged, just not yet sped up). A dead child (`tp == 0`) is now
caught from the count alone, before `refine_cover` ever builds its
handle -- *not done*: deferring handle construction further, for live
children that don't survive to the next round/beam slot, as the original
sketch's "build the covered set only for the candidates it keeps" also
proposed; nominal-attribute batching (the same idea for a multi-valued
attribute's value tests) is also still open.

**Correctness**: unweighted (plain `RemovalCovering` -- CN2/AQR/PFoil/
PFossil/Pypper/PyLORD's default) is bit-identical to the old path, every
count an exact integer; checked against the old path directly (CN2 on 5
numeric-heavy datasets: `segment`, `diabetes`, `sonar`, `ionosphere`,
`banknote-authentication`; PFoil/PFossil/Pypper on `sonar`), and the full
test suite (777 passed). Weighted (`WeightedCovering`, boosting's
per-round reweighting) can differ from the old path by float-reordering
noise (~1e-14 relative -- histogram+cumsum sums the same weights in a
different order than a direct masked sum, same issue any vectorized
reduction has) -- harmless to the actual numbers, but can rarely flip an
exact score/stopping-criterion tie to a different, equally valid rule.
Accepted as expected floating-point behavior, not a bug, after checking
it against a battery of real fits rather than assuming.

**Measured speedup** (one fit each, old vs. new, same machine): CN2 --
`segment` 1.49x, `diabetes` 2.24x, `sonar` 1.34x, `ionosphere` 1.17x,
`banknote-authentication` 1.54x; on `sonar`: PFoil 1.07x, PFossil 1.19x,
Pypper 1.12x, `WeightedCovering`+`AdditiveReweighting`+Laplace 1.10x,
Slipper 1.03x. Consistently positive, well short of the sketch's
upper-bound "~15x less counting" estimate -- that bound was for the
counting step alone; `refine_cover`'s handle construction for every live
child (unchanged here) is the rest of a round's cost, and these numbers
are the net of both.

Still open: nominal-attribute batching. (N-list, PrePostNList and
Sparse got their own batched counting instead -- see "Batched counting
on every representation" below.)

**No longer on the search path (2026-10-08).** The next section's lazy
searches count every open child with one `batch_cover_counts` matmul
instead, so `chain_cover_counts` and `score_children`'s chain grouping
currently have no callers; kept, not deleted,
until it's decided whether they come back (e.g. for representations
without a matmul).

## Build only the children a search follows

**Done (2026-10-08).** `HillClimbing` (and `GainAscentHillClimbing`)
and `BeamSearch` used to call `rule.specialize`, which
builds *every* child of a node -- a `Rule`, its constraint closure
(`extend_closure`) and its mask -- and then move to one of them (or
`beam_width`). Profiled on `sonar` (1080 features), that was ~95% of a
search's time; the counting itself ~2%. Hence also why the earlier
matmul experiment (`_score_children_matmul`, below) barely moved the
needle: it sped up the 2%.

Now `count_open_children` gives `(tp, fp, fn, tn)` for every open
feature without building anything (one `batch_cover_counts` matmul on
`HorizontalBitmapRepresentation`, one `refine_cover` + `cover_counts` per
feature on any other representation), the search scores and sorts from
those counts, and `materialize_child` builds -- closure, mask, handle --
only the child(ren) it actually follows, skipping one that turns out
contradictory exactly as `specialize` would. `BeamSearch` deduplicates
children reached from different parents by a bitmask of their condition
features (Rule equality is its condition set) before building any.
Tie-breaking is unchanged (stable sorts over `specialize`'s feature
order).

**Measured** (old eager vs. new lazy, one fit each, same machine; every
model identical except as noted):

| dataset (rows x features) | PFoil | PFossil | Pypper | Slipper | CN2 | AQR | PyLORD |
|---|--:|--:|--:|--:|--:|--:|--:|
| diabetes (768 x 112) | 5.3x | 4.1x | 2.8x | 4.0x | 7.4x | 4.5x | 3.1x |
| sonar (208 x 1080) | 19x | 17x | 15x | 22x | 42x | 4.1x | 10x |
| kr-vs-kp (3196 x 76) | 3.0x | 2.2x | 1.5x | 2.7x | 2.7x | 3.0x | 3.2x |
| spambase (4601 x 998) | 1.6x | 1.7x | 1.4x | 1.7x | -- | -- | -- |

The gain is largest with many features and few rows: building a child
cost the same per feature at every node, while the remaining counting
scales with the rows -- on `spambase` counting is now most of the time.
Full `demos/ripper_comparison.py --full` re-run (44 datasets, 10-fold):
Pypper's 440 fold results identical to the stored run; Slipper's differ
in 13 of 452, consistent with the weighted float-reordering noise
described in the previous section (Slipper's counts are weighted sums; a
matmul sums them in a different order) -- not yet confirmed, since the
stored run also predates `chain_cover_counts`.

**`CPAR` on this machinery** (`pyrulearn.learners.cpar`). `CPAR` used to
score ``(wp * cov) @ Xf`` over every feature not yet in the rule, on a
dense matrix; that version is now `DenseCPAR`. The modular `CPAR` grows
through this machinery instead -- first built (as `PropagatingCPAR`) to
test whether constraint propagation (fewer open features) pays for
itself there, and made the main `CPAR` on 2026-10-09, following the
package's components-first approach (README, *Native learning
algorithms*), with the growing heuristic, the covering strategy and the
combiner as parameters and `DecomposingLearner` for multi-class. Same
models as `DenseCPAR` (an implied condition has FOIL gain 0, below
`min_gain`, so `DenseCPAR` never picks one; checked on `diabetes`,
`sonar`, `kr-vs-kp` and in `tests/test_cpar.py`, also with exchanged
components). Built eagerly through `specialize` first, it was
3.6x (`diabetes`) to 350x (`sonar`) slower than `DenseCPAR`; lazy, 6-34x
faster than that, but still 4x (`diabetes`: 1.54s vs. 0.40s), 14x
(`sonar`: 0.82s vs. 0.06s) and 8x (`kr-vs-kp`: 1.07s vs. 0.14s) slower
than `DenseCPAR`. Profiled on `sonar`, the gap was three overheads, none
intrinsic to propagation: per-candidate Python scoring ~55%,
`batch_cover_counts`'s bool-to-float slice ~22%, recomputing the parent
closure ~21% (cProfile's estimates; see below for what fixing each one
actually measured). After all three: 2.4x (`diabetes`: 0.92s vs.
0.38s), 2.8x (`sonar`: 0.16s vs. 0.06s), 4.4x (`kr-vs-kp`: 0.56s vs.
0.13s).

Still open: AQR gains far less than CN2 on `sonar` (4x vs. 42x), so
something else dominates it -- not yet profiled.

**Closures passed down (done, 2026-10-08).** Every expanded node used to
recompute its rule's constraint closure from scratch (`parent_closure`),
although `materialize_child` had just computed it to build the child's
mask. It now returns it, and the searches carry it alongside mask and
cover handle; only a seed's is still computed. Same models, but no
measurable speedup (within +-10% everywhere) -- the profile's ~21% for
it was cProfile's per-call overhead on `propagate`'s many small Python
calls, not real time.

**Covered rows only (done, 2026-10-08).** `batch_cover_counts`
multiplied the weights of *all* rows against the open columns, copying
(and converting to float) an `n_rows x n_open` block per call, though
uncovered rows only contribute zeros. It now takes only the covered rows
(`X[covered][:, open]`, one copy), a block that shrinks with every
condition. Rejected instead: a cached float copy of `X` (8x the boolean
matrix's memory). Same models; best of 3 vs. the previous version, same
machine: `spambase` (4601 rows) 1.9-6x (CN2 23.9s -> 4.0s, PyLORD 277s
-> 145s), `kr-vs-kp` 1.2-2.3x, `sonar` 1.2-2.3x (modular `CPAR`
2.3x, now 2.8x behind `DenseCPAR`), `diabetes` ~1.1x.

**Batch scoring (done, 2026-10-08).** Each candidate used to be scored by
its own Python-level `heuristic.score` call. Every heuristic now also has
`batch_score`, which scores arrays of stats at once: the base class's
default loops over `score` (so a heuristic only *needs* `score`), every
built-in overrides it with the same formula in numpy. Two versions rather
than one array-safe `score` because numpy is slow for a single value,
and `score` stays the fast path for that; `tests/test_heuristics.py`
checks they agree (bit-identical for the arithmetic heuristics, to
rounding for the log-based ones and `Correlation`/`ChiSquare`, whose
scalar versions multiply exact Python ints). The searches score a node's
children with one `batch_score` call and rank them with one `lexsort`,
and compute everything they compare against those scores (a parent's
own score, the running best, the optimistic bound) through
`batch_score` too -- mixing the two could let a child with its parent's
exact stats win by a rounding difference. Same models in every case
measured; best of 3 fits, old vs. new, same machine: `sonar` (1080
features) 1.5-2.3x (modular `CPAR` 1.84x, now 7x behind `DenseCPAR`
instead of 14x), `diabetes` (112) 1.2-1.6x, `kr-vs-kp` (76) ~1.0x --
the gain follows the number of open features per node. AQR unchanged
(~1.0x everywhere), so its time is elsewhere.

## ENDER on the representation's primitives

**Done (2026-10-09).** `ENDER` (and `Boomer`) grow a rule the way the
lazy searches do: walk a cover handle, sum every open condition's
impurity terms with one `DataRepresentation.batch_cover_sums` call per
step (generic: one refinement per condition; `HorizontalBitmapRepresentation`:
one matrix product over the covered rows with a nonzero term), and build
only the condition added, with constraint propagation. The impurity
criterion is a component (`ImpurityCriterion`: `ConstantStep`,
`Gradient`, `GradientBoosting`, `Simultaneous`, `Newton`), like the loss.
The dense implementation stays as `DenseENDER`/`DenseBoomer`. Same
models (tested on 24 configurations, and on every representation).

**Measured** (dense vs. modular on Boolean, 100 rules, best of 3, same
machine):

| dataset (rows x features) | constant step (default, 25% subsample) | Newton, all rows |
|---|--:|--:|
| diabetes (768 x 112) | 1.1x | 1.1x |
| sonar (208 x 1080) | 1.5x | 2.9x |
| kr-vs-kp (3196 x 76) | 0.9x | 4.4x |
| spambase (4601 x 998) | 3.3x (15.1s vs. 4.6s) | 7.7x (45.2s vs. 5.9s) |

With a subsample, the rows outside it carry zero terms and are skipped
(spambase's default went from 27.6x to 3.3x with that). Newton steps on
all rows are the expensive case: every step copies the covered rows of
the open conditions, where `DenseENDER` multiplies its precomputed float
matrix.

**Rounding could add conditions that change nothing -- fixed, in both
versions and in the searches.** A condition that drops no row carrying
weight has exactly the rule's own impurity (or tp/fp), but summing a
different set of rows -- the covered rows only, as `batch_cover_counts`
and `batch_cover_sums` do -- can round it a few ulp lower, and a search
then takes it for a strict improvement. Seen in `ENDER` on `kr-vs-kp`
and, more surprisingly, in `DenseENDER` on `spambase`, whose sums always
run over all rows: a matrix product can round different output columns
differently, and `capital_run_length_total < 693.5` was added (by 4e-15)
to a rule already holding a stricter threshold on the same attribute.
Now every search counts the contributing rows exactly (integers): a
condition keeping all of them gets exactly the rule's own score
(`count_open_children`'s `same`, `batch_cover_counts`' `keeps_all`, an
indicator column in `ENDER`'s sums). Unweighted counts were always exact;
the SeCo learners and `CPAR` learned the same models as before the fix,
a few percent slower.

Built (2026-10-12, `demos/dense_vs_modular.py`, report checked in for all 44 small and medium binary datasets, 2026-10-13): a dense vs. modular demo -- every learner with both
versions side by side: `CPAR`/`DenseCPAR`, `ENDER`/`DenseENDER`,
`Boomer`/`DenseBoomer`, `ORB`/`DenseORB` (accuracy, model size, fit
time, on every representation for the modular ones). Plan (2026-10-10):
start with a quick comparison that rules out ORB's exhaustive search
(`BranchAndBoundSearch` without a length cap times out on most datasets
beyond vote/tic-tac-toe -- over 300s per fit on breast-cancer); in the
long run, compare `BeamSearch` widths for ORB. A first look (scratch
run, not a demo): SPECT, breast-cancer, colic, credit-approval,
heart-statlog, hepatitis, tic-tac-toe, vote; 5-fold, 10 rules, N-lists.
Mean accuracy / mean rank / total conditions / fit time: greedy 83.08% /
3.06 / 32.6 / 0.13s; beam3 82.56% / 3.75 / 36.9 / 0.14s; beam5 83.12% /
3.06 / 39.3 / 0.22s; beam10 83.64% / 2.00 / 41.0 / 0.43s; branch and
bound capped at 3 conditions 83.06% / 3.12 / 26.6 / 38.2s. Greedy won on
the three noisiest, smallest datasets (SPECT, breast-cancer,
hepatitis). Uncapped branch and bound: vote 2s, tic-tac-toe 118s,
breast-cancer over 300s per fit. Hence `ORB`'s default search is
`BeamSearch(beam_width=10)`.

## Batched counting on every representation

**Done (2026-10-09).** `batch_cover_counts` and `batch_cover_sums` --
which every rule search, `CPAR` and `ENDER` count with -- now exist for
all four representations, so none falls back to one `refine_cover` +
`cover_counts` per open condition any more. All count horizontally,
from the rows a rule covers to the features they have, with one sparse
product (CSR: Compressed Sparse Row, per row the features it has):

- `SparseDataRepresentation`: the covered rows (with a nonzero weight)
  of the CSR it already keeps; cost: their nonzeros.
- `NListRepresentation`/`PrePostNListRepresentation`: rows with the same
  feature set end at the same tree node, so the covered rows' values are
  first summed per distinct row pattern (`bincount`), and only the
  patterns that occur are multiplied, through a CSR of the distinct
  patterns built once from `_row_feats`; cost: distinct patterns, not
  rows. Reading counts off an ancestor node's path instead would miss
  the features below it -- the depth trap of both `chain_cover_counts`
  attempts.

A count column gives `keeps_all` exactly, so the rounding guard works on
every representation. Checked against the per-feature path on 8856
counts (every feature, dead ones included; weighted, masked) with no
mismatch. Every learner learns the same model on every representation
as on Boolean, and as before. **Measured** (best of 3, old vs. new, same
machine): NList 1.5-10x on `diabetes`, 1.8-7x on `kr-vs-kp`, 9-88x on
`sonar` (`ENDER` most); Sparse 3-8x, 3-8x, 9-35x; Boolean unchanged.
Now the other representations are as fast as Boolean or faster:
`kr-vs-kp`'s `CPAR` 0.27s on N-lists vs. 0.50s on Boolean, `Pypper`
0.17s vs. 0.43s, and on `sonar` every learner is faster than on
Boolean -- whose batch copies a dense block of covered rows x open
conditions at every step, where the CSR products touch only the true
entries. Decided (2026-10-09) not to do the same for Boolean: it is
the dense representation, N-lists and sparse data are the sparse ones,
and a CSR copy next to `X` would cost memory for what they already
offer. Worth re-running the representations demo for.

## Several rules per search: CPAR as a SeCo configuration

**Done (2026-10-09).** A `RuleSearch` can yield several rules
(`search_all`; default: `search`'s one), `SingleRuleLearner.learn_rules`
post-processes each, and `SeCo`'s covering loop accepts them in turn --
all judged on the scope they were found in, each updating the covering.
`GainAscentHillClimbing` got `min_gain=` (a gain of at least this,
instead of any positive gain) and `branch_similarity=` (also follow
every child within that factor of the best gain, as a copy grown on the
same way: CPAR's search). With those, `CPAR` is a `SeCo` preset --
`FoilGain`, the branching gain ascent, `WeightedCovering`, a top-k
combiner -- with no search or covering code of its own, and the
branching is available to every SeCo learner (e.g. `PFoil` with copies).
Same models as before (`CPAR` == `DenseCPAR` in `tests/test_cpar.py`;
the other SeCo learners unchanged). `max_rounds` now caps rules per
class, as `WeightedCovering` counts, not searches; `DenseCPAR` counts the
same way.

**First experiment** (3-fold CV, the 10 small datasets of the RIPPER
quick run): no accuracy effect anywhere -- every variant within about a
point (0.830-0.841 mean accuracy), well inside the noise of so few
folds. Branching mostly adds rules: `PFoil` 28 -> 67 (similarity 0.99)
-> 201 (0.9), since removal covering still accepts copies whose
examples the first rule already removed; in `CPAR`, branching at 0.99
rarely triggers (126 rules without, 135 with) and `CPAR` without it was
no worse (0.841 vs. 0.837); `Pypper` hardly changes, grow-then-prune
absorbs the copies (one outlier: `tic-tac-toe` 0.932 -> 0.981). So: a
clean generalization `CPAR` needs, no reason yet to make branching a
default anywhere -- a 10-fold run over many datasets would be the real
test (a demo).

Open: that demo; a `BeamSearch` counterpart (return the final beam
rather than its best rule).

## Optimal rules: branch and bound, OPUS

**Done (2026-10-09).** `seco.BranchAndBoundSearch(max_conditions, k)`:
the optimal rule (or the `k` best, distinct coverage) by the heuristic
among all conjunctions, after OPUS (Webb 1995) -- set-enumeration tree,
best-first by the `(tp, 0)` bound, OPUS's removal of a condition from
the whole subtree once it is bounded out, dead or changes nothing,
constraint propagation, one batched count per node (every
representation). Checked against brute-force enumeration (96 cases:
several heuristics, `k` 1 and 5, weighted data, threshold chains,
Boolean and N-lists). Learners in `learners.opus`: `Opus` (`SeCo` with
it; `Laplace(n_classes=c)` by default -- CN2's and Webb's Laplace, with
the number of classes in the denominator; `Laplace()` keeps 2 -- as in
Webb's experiments, Section 6 of the OPUS paper; `CN2` now defaults to
it too, as Clark & Boswell's)
and `OpusTopK` (the `k` best rules per class by `WRAcc`, no covering,
Magnum Opus style).

First observations (Titanic): `Opus` with `Laplace` is slow (40s, 142
rules) -- its optimum is a narrow pure rule and the bound prunes little;
with `WRAcc` 0.65s, 8 rules. `OpusTopK`'s top rules include near
duplicates (`sex = male` and `sex = male, parch != 6`, one row apart):
Webb's *cannotImprove* pruning (`dominance_pruning`, for `k = 1`
without filtering): conditions dropping no covered negative, and
conditions whose child a sibling dominates, leave the subtree.
Checked against brute force (120 cases) and against the search without
it.

Webb's filter for *productive* rules drops them: `pruning.ProductiveRule`
(precision strictly higher than every generalization, the empty rule
included), a `filtering=` criterion so that the top-k search finds `k`
productive rules -- on by default in `OpusTopK` (Titanic, k=5: 3.3s
instead of 1.4s). Still in the top-k: several negated values of one
attribute (`sibsp != 4, sibsp != 5, sibsp != 8`), each productive --
Webb's language of `≠` conditions is internal disjunction. Pretty
printing now shows them as one condition (`sibsp ∉ {4, 5, 8}`, or `∈`
the remaining values when that's shorter), and thresholds on one numeric
attribute as one interval; open: a value-set *feature* that also counts
as one condition, for rule-length measures and `max_conditions`.

## Objectives: what a search maximizes; ORB on the rule searches

**Done (2026-10-10).** Decision (a) of 2026-10-09, taken: the searches no
longer call a heuristic; they talk to an `Objective`
(`pyrulearn.learners.seco`), which owns counting (a rule's statistics,
and every open child's at once, from the representation's primitives),
scoring, optimistic bounds, and the "changes nothing" / "nothing left"
predicates. `HeuristicObjective` wraps a `RuleHeuristic` (statistics
`RuleStats`, bound at `(tp, 0)`, counting via `count_open_children`) --
every search wraps a heuristic it is given, so the API is unchanged.
`ValueSumObjective` is a function of sums of per-row values over the
covered rows (one `batch_cover_sums` per step; an exact integer count of
contributing rows decides "same" and "dead"), optionally with an
`exact_bound` from the covered rows that `BranchAndBoundSearch` applies
to every child it builds. Checked: all 120 models of 10 search-based
learners (CN2, AQR, PFoil, PFossil, Pypper, PyLORD, CPAR, Opus x2,
OpusTopK) on 6 datasets x Boolean/N-list identical to before; timing
below.

On top: `ORB` -- optimal rule boosting with any rule search on the
`XGBGain` objective, passed as a `RuleSearch` like in every SeCo learner
(default `BeamSearch(beam_width=10)`, see the dense vs. modular demo
item; `BranchAndBoundSearch` is realkd's exhaustive search,
`HillClimbing` its greedy one), on every representation; the old
`OptimalRuleBoosting` is `DenseORB`, the dense-matrix specialization.
Both reproduce realkd exactly (`tests/test_realkd_import.py`), and ORB
reproduces DenseORB model for model. `XGBGain(sign=)` restricts to
rules raising (`+1`) or lowering (`-1`) the scores -- the per-class
search a multi-class ORB would run, and the way to use class-dependent
filters; not used by `ORB` yet (binary, class from the weight's sign).

**Components by object, not by name (2026-10-10).** Every learner
component is now passed as an object, like a SeCo learner's search and
heuristic: ENDER/DenseENDER/Boomer `loss=` (`LogisticLoss()`,
`ExponentialLoss()`, `SigmoidLoss(beta)` -- `beta` moved onto the loss)
and `method=` (`ConstantStep(beta)`, `Gradient()`, ...; the learner's
`beta` is gone); ORB/DenseORB `loss=` (`LogisticMarginLoss()`,
`SquaredMarginLoss()`, new `MarginLoss`); ORB `search=` (a `RuleSearch`);
every `combiner=` (`HeuristicMaxCombiner()`, `ListCombiner()`,
`CountVoteCombiner()`, the Micro/Macro combiners; pairwise:
`MajorityVote()`, `WeightedVote()`, `AccuracyWeightedVote()` -- the
shortcut tables are gone); IDS `optimizer=` (`SmoothLocalSearch(samples,
max_restarts, final_samples)`, `GreedySelection()`; the `sls_*`
parameters moved onto the optimizer). Shorthands that only swapped a
criterion's role are gone too: `CN2`/`PFossil` `mode=` (pass the
criterion as `filtering=`/`stopping=`), `AQR` `seed_strategy=` (pass
`space_init=SeedExample(...)`). Left as names, on purpose: `DenseORB`
`search=` (a fixed pair of hand-written searches), `SeedExample`
`strategy=`, `MajorityVote` `tie_break=` (accepts a callable), the
multi-class `order=`/`positive=` (a label sequence or callable also work),
RuleFit's `solver=` (scikit-learn's own).

## LRI on the rule searches

**Done (2026-10-10).** The previous `LRI` is `DenseLRI`; `LRI` is built
from the components, as `ORB` is: per class, `n_rules` rounds of a
`WeightedCovering(LRIReweighting(), max_rounds=n_rules)`; a round's DNF
rule is up to `max_terms` terms, each found by a `SingleRuleLearner` --
`HillClimbing(max_conditions=5, stop_at_local_optimum=False)` (any search
works) over `LRIError`, the new objective for the paper's weighted
`FP + k*FN` with its adaptive `k` -- with the covered cases removed
between terms (scope = the round's weights times the remaining rows) and
`FeatureSubset` after `freeze_features_after` rules. `max_length`,
`power` and `max_errors` moved onto the search and `LRIReweighting`.
`LRIError` counts all conditions not yet in the term, the dead ones
included, to choose `k` as the paper does, and returns children only for
the conditions the search keeps open (constraint propagation).

Checked against `DenseLRI`: 54 configurations on plain Boolean features
(weighted, three classes, frozen, short rules; Boolean, N-list and sparse
data) give identical models -- the weights are integers, so every count is
exact whatever the summation order. With negation and threshold features
the models differ: the dense version can append a condition the term
already implies (72 of 160 terms of tic-tac-toe carry a redundant
condition, 42 for `LRI`, whose remaining ones are conditions that change
nothing on the training data -- the paper's "always add the best
condition" does that), and the reweighting cascades from there; test
accuracy on one 70/30 split of five datasets was about the same (`LRI`
ahead on three of them, by up to 3.5 points; single splits).

Speed (10 rules per class, best of one on `adult`, 242-256 features):
2,000 rows: `DenseLRI` 0.88s, `LRI` 0.46s on Boolean data, 0.29s on
N-lists; 8,000 rows: 5.5s / 2.0s / 0.67s; 32,000 rows: 47s / 7.0s / 2.5s.
Unlike `CPAR` and `ENDER`, the modular learner is the faster one here,
and the gap grows with the data: the dense term search multiplies the
whole data matrix per condition, while the modular one only counts the
rows its term still covers. On the small datasets of the demos (a few
hundred rows) `DenseLRI` is about twice as fast (0.05s vs 0.10s for 20
rules per class on vote).

To write up (report or paper): the objective abstraction -- searches as
pure strategies over an objective that owns counting, scoring and
bounding -- is central to this package's design; class-count heuristics
and boosting gains are its two instances so far. ENDER's
`ImpurityCriterion` is a third candidate (its own grower is a hill
climb on an objective of the same kind).

## 0.3.0: every native learner on any data representation

**Principle.** Learners are written against the `DataRepresentation`
interface and run on any representation; which representation is used is
decided when the data is prepared, not inside a learner. This was the
design from the start, but some later learners broke it.

**Why now.** The representations demo (`demos/representations.py`,
2026-09-30) found that N-lists take about half of Boolean's fit time for
the SeCo learners and PyLORD, and the gap grows with the data: at 32,000
examples of `adult`, CN2 finishes only on N-lists. `PrePostNList` was
only marginally faster than plain N-lists (51% vs. 53% of Boolean's time)
while taking longer to build, so plain `NListRepresentation` is the
intended default.

**Status.** Everything that goes through a rule's coverage
(`covers_data_packed`, which forwards to the data's own `coverage`) is
already representation-independent: the SeCo framework with CN2, AQR,
PFoil, PFossil and Pypper, PyLORD, the rule models, pruning and
evaluation. Some implementations turned the data into a dense matrix
(`data.X`) and worked on its columns with numpy instead: `CPAR`, `ENDER`
(with `Boomer`), `OptimalRuleBoosting` and `LRI`. All four now run on
every representation, built from the SeCo search primitives (`ORB`,
`LRI`: 2026-10-10, see "Objectives" and "LRI on the rule searches"
below), and the dense-matrix versions stay on purpose as `DenseORB`,
`DenseLRI`, `DenseCPAR` (same models, 2-4x
faster) and `DenseENDER`/`DenseBoomer` (same models, 1-8x faster) -- see
"Build only the children a search follows" and "ENDER on the
representation's primitives" below. (`ENDER` had been investigated on
2026-10-07/08 and kept dense: a representation-generic rewrite measured
10-500x slower -- before the lazy search, batch counting and the
primitives below existed.)

External learners reading `data.X` is fine: the external tools need a
matrix anyway.

**Shared mechanism (2026-10-08).** A learner that commits to one (or
more) representations' own storage instead of the universal
`coverage`/`initial_cover`/`refine_cover` interface declares
`NativeRuleLearner.NATIVE_REPRESENTATIONS` (`None`: any representation
works as given, unconverted -- the default, true of everything above
except the four listed here); its native fit method calls
`self.ensure_representation(data, self.max_auto_convert_cells,
purpose=...)` to convert anything else, bounded by
`max_auto_convert_cells` (raises past it -- no silent unbounded-memory
rebuild). Ties, when a learner's native set names more than one
representation and the data given matches none of them, break toward
`REPRESENTATION_PREFERENCE_ORDER` (`NListRepresentation` first, then
`HorizontalBitmapRepresentation`, then `SparseDataRepresentation`) -- so
absent a more specific reason, N-list is this library's default
conversion target, the same preference item 1 below gives data
*preparation*. `ENDER`/`Boomer` (`NATIVE_REPRESENTATIONS =
(HorizontalBitmapRepresentation,)`) and `pyrulearn.learners.associative.CARMiner`
(`(NListRepresentation,)`, converting in the *opposite* direction, for
mining's speed rather than `_grow`'s correctness) are the two current
users; `CARMiner`'s own `ensure_nlist` (which used to accept any other
representation as-is, just slower, since `generate_cars` only needs the
universal `coverage()`) was folded into this shared mechanism too, which
means that leniency is gone -- a `SparseDataRepresentation` now also
gets converted rather than used directly, since the mechanism doesn't
distinguish "needed" from "just much faster" and CARMiner no longer has
its own bespoke gate to make that distinction in.

**Work**, for the three remaining learners (`ENDER`/`Boomer` excluded,
see above -- expect them to need the same investigation before assuming
the work below is worth doing for them too):

1. *Data preparation*: a `representation=` choice where data is prepared
   (`pyrulearn.experiments.runner.run_cv`, the demos, the loading
   helpers), with N-lists as the default. **The primitive this needs is
   done (2026-10-08): `pyrulearn.data.io.encode`** (`binarize`'s
   sparse-intermediate cousin) builds whichever `DataRepresentation` is
   asked for -- default `NListRepresentation` -- directly from a
   `scipy.sparse` matrix, never forming the dense `(n_samples,
   n_features)` array `binarize` always does, even transiently;
   `NListRepresentation.__init__` now also accepts a `scipy.sparse`
   matrix directly (reading rows from its CSR structure instead of
   `np.flatnonzero` on a dense row -- `HorizontalBitmapRepresentation`/
   `SparseDataRepresentation` already did). **Done (2026-10-10)**:
   `REPRESENTATION_PREFERENCE_ORDER` moved to `pyrulearn.data.representation`
   (re-exported from `learners.base`), now with `PrePostNListRepresentation`
   after `NListRepresentation`; its first entry, `default_representation()`
   (read when called, so reassigning the module attribute changes the
   default library-wide), is what `encode`, `read_arff`/`read_csv`
   (new `representation=` parameter) and `run_cv` (same) build when no
   representation is given. `binarize` stays as the lower-level
   dense-matrix helper. `run_cv`'s fold cache key now includes the
   representation's class name, so a representation change can't
   silently replay a stale fit_time from a different one (the exact trap
   hit the same day in `seco_learners_comparison`'s cache).
   `NATIVE_REPRESENTATIONS`/`ensure_representation` moved up from
   `NativeRuleLearner` to `RuleLearner`, `max_auto_convert_cells=None`
   meaning unbounded; `ExternalRuleLearner` declares
   `(HorizontalBitmapRepresentation,)` and converts in `fit`, unbounded and
   silent (the tool needs the dense matrix either way) -- so every
   learner, external ones included, accepts every representation, and
   the demos need no `representation=` of their own. Checked with the
   `run_cv` demos' quick runs, not full reruns. (`LRI` and
   `OptimalRuleBoosting` read `data.X` directly, which every
   representation provides -- built on demand for N-lists and sparse data
   -- so they ran on any representation too; they are `DenseLRI` and
   `DenseORB` now, next to the representation-generic `LRI` and `ORB`.) The tutorial notebook asks for `HorizontalBitmapRepresentation`
   explicitly (it shows `data.X`), with a note that N-lists are the
   default.
2. *The learners*: **done (2026-10-10)** -- `CPAR`, `ENDER`, `ORB`,
   `LRI` score candidate conditions through the representation's
   (weighted) coverage functions instead of matrix columns; the speed is
   measured on every representation in each learner's section (numpy
   scores all features in one vectorized step, so the modular versions
   are slower than the `Dense*` ones on small Boolean data).
3. *Tests*: extend the check that all representations give identical
   rules (so far CN2/PFoil/PFossil/AQR, PyLORD, and `ENDER` (2026-10-08,
   checking its deliberate dense-conversion path rather than true
   representation independence), `tests/test_representations.py`) to
   every native learner, so the principle is enforced, not just
   intended.
4. *Optional*: build the N-list index faster (it is a Python loop per row:
   about 7 s for 32,000 rows of `adult`) and store feature indices as
   32-bit integers to halve its memory.

## Memory of the representations; N-list on top of a base storage

**Open (discussed 2026-10-13; start after the re-run demo reports from
the server are checked in).** The memory measurement in
`demos/representations.py` (`tracemalloc`, held after the build and peak
while building) shows the vertical representations far larger than the
packed Boolean matrix -- `adult`, n=32,000:

| | with negation | without negation | peak while building (with) |
|---|--:|--:|--:|
| Boolean (packed bits) | 1.1 MiB | 0.55 MiB | 1.1 MiB |
| Sparse | 50 MiB | 10.2 MiB | 113 MiB |
| NList | 166 MiB | 38.4 MiB | 528 MiB |
| PrePostNList | 190 MiB | 41.7 MiB | 706 MiB |
| unpacked `X` | 8.5 MiB | 4.3 MiB | |

*Why Boolean wins.* One bit per cell against at least one integer per
*true* cell: an `int32` index list only beats packed bits below a density
of 1/32 (1/64 with two copies, as `Sparse` keeps CSR and CSC). `adult`
without negation has density 0.136 (19 of 140 features per row), so any
index list is at least ~4x the packed matrix per copy. Sparse storage
saves memory against an *unpacked* matrix, not against packed bits.

*Where N-list's memory goes* (n=32,000, no negation; deep size of the
stored attributes, 23.7 MiB of the 38.4 MiB `tracemalloc` reports):
`_row_feats` 8.3 MiB (32,000 separate small `int64` arrays -- the whole
Boolean table a second time, in a far more expensive format; used by
`features_of` and `X`), `_row_idx` 4.7 and `_entry_node` 4.7 (`int64`,
one entry per true cell: a full tid-list, i.e. the information of
`Sparse`'s CSC), `_path_words` 2.4 (the per-node path bitmasks -- the
only part corresponding to the paper's N-lists), `_node_class_counts` 1.6
(`int64`), `_leaf_of` 1.6+ (assigned during the build, never read again,
and it keeps the trie's leaf nodes alive -- probably most of the
unaccounted 15 MiB; not verified). 105,271 tree nodes. `Sparse`: CSR 3.0
+ CSC 2.9 MiB (`int32` indices, `bool` data) against 10.2 MiB from
`tracemalloc` -- the gap was the Boolean source's cached `X` (see
"Done" below).

*The tree is the bottleneck, as in Huynh et al.* -- but only while
building: the Python `_PPCNode` trie (a dict of children and a Python list
of row ids per node) causes the ~3x peak, then is dropped. What is kept
is more than LORD's N-list: LORD only *counts* (per node pre/post code and
count), while our learners also need (1) *which* rows a rule covers
(removal in separate-and-conquer, predictions), (2) counts *under a
mask* (uncovered examples, grow/prune splits), and (3) counts under
*changing weights* (boosting, weighted covering, LRI). Hence the row ids
per node -- so our N-list is LORD's N-list plus a sparse matrix, and can
never be smaller than `Sparse`.

*Decided direction.* Base storage underneath, N-list as an index on top:

- Base: Boolean (packed bits) or Sparse (index lists, `int32` is enough
  -- scipy picks it when the sizes fit), chosen by density. The base
  answers `features_of`, `X` and "which rows does this rule cover"; the
  N-list holds no copy of the table.
- N-list is positioned as a *trade-off between memory and speed*, not as
  the obvious default. The case for it as the default has weakened: it
  is clearly faster for the SeCo learners on larger data (CN2, PFossil,
  Pypper 2-5x on `adult` at 16,000-32,000 rows), but even or slower for
  the small datasets and the boosting/CPAR learners (`dense_vs_modular`:
  N-list/Boolean 0.87-1.88x), at 70-190x the memory. Revisit
  `REPRESENTATION_PREFERENCE_ORDER` after the fixes below, on the new
  numbers.

*Options for the N-list index:*

- **C. Current layout, slimmed -- do first; only changes that don't cost
  time.** Drop `_leaf_of`; drop `_row_feats` (answer from the base);
  `int32` for `_row_idx`, `_entry_node`, `_node_class_counts`, `_y_idx`;
  replace `_entry_node` (one node id per true cell) by per-node offsets
  and `np.repeat(hit, sizes)` instead of `hit[entry_node]` (needs a
  timing check). Estimate: ~38 -> ~6-7 MiB here (still ~12x Boolean;
  `_row_idx` alone is ~4x at this density). Storage only, so identical rules (the equality tests cover it);
  benchmark old vs new as usual (worktree,
  sequential, best of 3), since `features_of` and coverage are hot paths.
- **B. Counts kept up to date -- prototype later.** Per node pre/post
  code and per-class (weighted) counts; per row the node ids along its
  path (one integer per true cell); removing or reweighting an example
  updates the counts along its path; "which rows" from the base. Drops
  `_row_idx`, `_entry_node`, `_row_feats` -- close to LORD's footprint.
  Removal is one walk down a path, a full reweighting one pass over the
  true cells; arbitrary changing masks (grow/prune splits) don't fit
  well. A redesign of how the search counts under masks and weights:
  prototype and full benchmark before deciding.
- **A. Count-only N-list** (pre/post + class counts, everything else from
  the base): smallest, but masked or weighted counts fall back to the
  base, which is what SeCo (masks after every rule) and boosting
  (weights every round) do all the time -- probably loses the speed gain.
- Peak while building: build the trie in numpy (sort the rows by their
  ranked features, group shared prefixes) instead of the Python
  `_PPCNode` trie -- lowers the peak and the build time (item 4 of the
  0.3.0 list above). The biggest of these changes.

`PrePostNListRepresentation` already stores pre/post codes, but
*alongside* the row ids, not instead of them.

*Done (2026-10-13, branch `memory-efficiency`): `from_boolean` without
the unpacked matrix.* `Sparse`/`NList`/`PrePostNList.from_boolean(b)` read
`b.X`, a cached property, so the Boolean source kept the unpacked `n x k`
matrix (8x its packed size) for as long as it lived -- and the demo's
`tracemalloc` counted it against the new representation. They now build
from `HorizontalBitmapRepresentation.to_csr()`, which unpacks a block of rows
at a time (2^16 cells: lower peak than larger blocks, and no slower) into
an `int32` CSR and caches nothing. `adult`, n=32,000
(held / peak while building, MiB):

| | without negation, before | after | with negation, before | after |
|---|--:|--:|--:|--:|
| Sparse | 10.2 / 18.8 | 5.9 / 5.9 | 50.4 / 113.0 | 41.9 / 41.9 |
| NList | 38.4 / 75.2 | 27.9 / 65.4 | 165.7 / 528.3 | 136.6 / 503.5 |
| PrePostNList | 41.7 / 97.5 | 31.2 / 87.7 | 190.3 / 706.3 | 161.2 / 681.5 |

N-list gains more than the 4.3 MiB of `X`, because its `_row_feats` are
now `int32` views into the CSR's indices instead of separate `int64`
arrays. Identical `features_of`, `X` and Pypper models; build times
unchanged within noise.

*Done (2026-10-14, branch `memory-efficiency`): the tree is the data
(option D).* Instead of slimming the layout (C), `NListRepresentation`
now stores the tree *instead of* the table, not next to it. The nodes
are numbered in pre-order and the rows kept in the order of the node they
end at, so the rows through a node are one contiguous range of that
order. Stored per node: the path bitmask (as before, for the search) and
its row range (2 `int32`); per row: its place in the order and its end
node (2 `int32`). `features_of`, `X` and the batched counting's row
patterns are decoded from the end nodes' path bitmasks; `select_rows`,
`relabel` and `without_negations` decode to a sparse matrix (no dense `X`
cached). Class counts are differences of cumulative per-class sums over
the rows in tree order -- under a mask or weights built once per search.
The tree is built in numpy (rows sorted by their ranked feature
sequences; nodes read off the shared prefixes of neighbouring rows), no
Python node objects. `PrePostNListRepresentation` coincides in storage:
it only keeps the two pre/post codes per node (which the base needs
during the build anyway) and overrides the re-anchoring strategy.

`adult`, n=32,000, held / peak while building, build time (against the
`from_boolean` fix above):

| | before | after |
|---|--:|--:|
| NList, no negation | 27.9 / 65.4 MiB, 0.88 s | 4.2 / 44.2 MiB, 0.14 s |
| PrePost, no negation | 31.2 / 87.7 MiB, 1.09 s | 5.0 / 44.2 MiB, 0.14 s |
| NList, with negation | 136.6 / 503.5 MiB, 5.13 s | 37.7 / 289.2 MiB, 0.82 s |
| PrePost, with negation | 161.2 / 681.5 MiB, 6.66 s | 43.8 / 289.2 MiB, 1.01 s |

Now slightly below `Sparse` (5.9 / 41.9 MiB); packed Boolean is still
far smaller (0.55 / 1.1 MiB). Fits on N-lists, `adult` n=8,000, new / old,
best of 3 (identical models in all 14 cases; a video stream ran during
some of it, so +-10% is noise): CN2 1.01 / 0.82 (without / with
negation), PFossil 1.05 / 0.77, ENDER 1.00 / 0.93, CPAR 0.96 / 0.91, LRI
0.96 / 0.68, Pypper 1.15-1.18 / 0.88-1.09, PyLORD 1.03 / 1.03-1.27. The
cost is in `coverage`: the old layout kept a row-id list per feature
(one integer per true cell -- the memory saved), the new one expands the
matching nodes' row ranges on every call (34 against 22 us per call; of
three formulations the cumulative "steps" one is the fastest), which
the coverage-heavy learners (Pypper's pruning, PyLORD) feel.

*Decided (2026-10-14), after the full re-run of
`demos/representations.py` with the memory a fit costs in total:*
`NListRepresentation` stays the default, for speed (the
separate-and-conquer learners are fastest on it on real data, and a fit
costs 1.3-1.7x Boolean's memory with negation features, about half
without); `HorizontalBitmapRepresentation` for dense and normal data when
memory matters; `SparseDataRepresentation` for sparse data (without
negation features). `PrePostNListRepresentation` stays an opt-in: equal
to N-lists in most of the report, faster on the largest dense data (CN2,
`adult` with negation, 32,000 rows: 22 s against 31 s). Written up in the
README ("Which one to use"). Worth a look separately: a fit on Boolean
allocates 30-90 MiB at 32,000 rows although the data holds 0.55-1.1 MiB
-- probably temporary unpacked or float matrices in the counting.

*Still open for N-lists:* on wide, sparse data the path bitmasks
dominate -- `k/64` words per node, and with little prefix sharing about
one node per true cell (estimate for the width sweep at k=5,000: ~11-12
MiB, against 0.6 for Boolean and 0.2 for `Sparse`). Replacing them by
ancestor tests through the pre/post codes (a binary search per feature,
as `PrePostNListRepresentation` already does for re-anchoring) would make
a node ~20-24 bytes whatever k, so N-lists could beat Boolean below about
1 true cell in 160-190 -- at a speed cost (the PrePost trade-off);
possibly only above some width.

*Sparse: what is left, and why it stays (for now).* The remaining
5.9 MiB is CSR 3.0 + CSC 2.9 (`int32` indices, `bool` data) -- close to the
floor for two index copies. Both copies are hot paths: CSC answers
`coverage` and `refine_cover`, CSR `features_of` and the batched counting
(`_row_sums`, one sparse product over the covered rows) -- dropping either
saves ~3 MiB but slows one of them; benchmark first if ever wanted.
`uint16` indices would mean leaving scipy (only `int32`/`int64`):
not worth it. The `bool` `data` arrays (all `True`, one byte per true
cell and copy: 1.2 MiB of the 5.9, 8.9 of 41.9 with negation) stay too:
dropping them means leaving scipy, and a numpy `bincount` version of the
batched counting was tried (2026-10-13) and lost -- 4-6x slower than
scipy's product on large covered sets (faster only below ~100 rows),
whole fits 1.02-2.03x slower (CN2, PFossil, Pypper, ENDER on `adult`,
identical models). Sharing one `data` array between CSR and CSC (half
the saving, no time cost) was left out: the two store the cells in
different orders, so it only works while every value is `True`.

## Deep rule networks: NNF networks and their learners

**In progress (branch `deep-models`, from 2026-11-07).** Learning
formulas in negation normal form as fully connected networks of
alternating AND and OR layers with 0/1 edge weights (Beck & Fürnkranz
2021, "An empirical investigation into deep and shallow rule learning";
Seip et al., "Towards SAT-based learning of NNF networks", SYNASC 2025, and
its unpublished long version for JAIR, with MaxSAT).

Done:

- `DeepModel` holds the weights (`layers` are the hidden sizes, like a
  neural network's; default one AND layer of `ceil(sqrt(n))` nodes);
  predicts by matrix products; its rules are a view for printing and
  export (`to_concepts`): every AND node a rule body, every OR node a
  concept with one rule per incoming edge, pass-through nodes removed.
- The datasets of both papers in the catalog, tags `deep` and
  `artificial`: Beck's 20 artificial truth tables and binary car and
  connect-4, read from his repository (`csv_url`, pinned to a commit).
- A `bool` column is a Boolean attribute (`x`, `not x`), so the
  artificial data comes in both papers' encodings: with negations (the
  SAT papers' 2d inputs) or without (Beck's).

Next:

- `DeepModel`: an AND node feeding several OR nodes as its own
  intermediate concept in the rule view (the long paper's step iii).
  (Done 2026-11-08: an AND output -- any number of hidden layers --, and
  `first="or"`, a first OR layer.)
- Three learners on a shared base (`DeepRuleLearner`):
  - `DeepLocalSearch`: Beck's stochastic hill climbing (his code:
    github.com/f-beck/scikit-learn-rule-network) -- per mini-batch of 50,
    the best of all single flips until none improves, 5 epochs, the best
    weights on the whole training set kept, attribute-aware flips in the
    first layer. The first version (one random flip per step) is far
    weaker than his published numbers and gets replaced.
  - `DeepSAT` (done 2026-11-08): Seip et al.'s encoding through PySAT
    (CaDiCaL), re-implemented with the known values as constants and
    checked against their code; `minimize="inputs"`/`"weights"` for
    MaxSAT, by an anytime linear search (default; trace, `max_seconds`,
    `max_conflicts`) or RC2. Conflicting rows keep the majority label.
    The `DataSpec`'s constraints as clauses on the weights
    (`constraints=True`): consistent rules, but on a quick test
    (tic-tac-toe, car-binary, monk-2, 20% training) neither faster nor
    more accurate -- plain SAT returns an arbitrary exact fit, which
    decides more. The linear search showed the Occam effect clearly:
    tic-tac-toe, 20% training, NNF(1), fewest conditions -- 146
    conditions test 0.68, 71 0.88, 45 0.96, 27 (after 15 s) 0.975, against
    Pypper's 0.859 on that split. PySAT can interrupt Glucose, MiniSat and
    MapleChrono but not CaDiCaL (both versions).
    Open:
    - KRK illegality as a test of hidden concepts (`pyrulearn.datasets`,
      added 2026-11-08): the coordinates alone, so "same file", "same
      rank", "adjacent" -- the ILP systems' background knowledge -- have
      to appear as hidden nodes (an OR over eight ANDs `wrf=v & bkf=v`,
      ...). Pypper, 1000 positions, 5-fold (majority 0.68): coordinates
      0.792, coordinates + relations 0.998, relations alone 0.998
      (`relations=True`, `coordinates=False`; the upper bound). Fossil
      used KRK with training sets up to 1000 (and 2000). To look at: the learned interior nodes (`to_concepts()`), and wider
      first layers than NNF(3)'s 32. kropt itself (28,056 positions, 18
      classes) is out of reach for the full demo; a binary version on a
      sample would be possible.
    - The output's class (`target_class`, done 2026-11-08). The night
      run (2026-11-08) had SAT time-outs on NNF(1) for six of the 20
      artificial datasets, where the paper had one; the cause: our output
      learned the less frequent class, the authors' always class 1 (the
      formula's). On artificial-nnf-107, fold 1: drl 4 s for class 1,
      still running after minutes for class 0; our DeepSAT 0.4 s (Glucose)
      / 2.5 s (CaDiCaL) for class 1, nothing in an hour for class 0 --
      with either solver. The demo now passes `target_class=True` in
      part 1. Local search's weak NNF(1) (66% against SHS's 94%) has
      another cause: on that fold 0.61 for class 1, 0.69 for class 0,
      after 0.2 s -- it stops far too early on a flat network; open.
    - Soft examples (done 2026-11-08, `size_penalty`): the objective
      `errors + lambda * size`, a regularized loss (0-1 loss, L0
      penalty); `None` is the exact fit. The linear search bounds it with
      two totalizers (edges, errors) instead of a pseudo-Boolean encoding
      (`pysat.pb` needs the extra `pindakaas`). Integer costs (lambda as
      a fraction p/q): with float weights RC2 took 25 s instead of 0.01 s
      on a small case -- 1 - 0.1 - 0.1 - ... never reaches 0 exactly, and
      the remainders (~1e-16) stay soft clauses RC2 keeps finding cores
      for; weights exact in binary (0.5, 0.125) were fast as floats. With
      soft examples `init` is the first solution (phases, and only
      cheaper networks wanted). Open: choosing lambda (by
      cross-validation, or relative to the data size), `sample_weight` as
      the examples' costs, and trying it on the demo's datasets -- noisy
      ones in particular, and started from Pypper.
    - Other objectives: distinct features used, rules used, costs per
      layer.
    - External MaxSAT solvers, like the Weka/LORD interfaces: write the
      formula as a `.wcnf` file and run an anytime solver with a time
      limit -- Loandra (core-boosted linear search), NuWLS or SATLike
      (local search) -- reading its `o <cost>` lines as the trace and the
      final `v` line as the weights. The strongest option for the larger
      datasets; needs the clauses built faster (numpy, or streamed to
      the file) than in Python lists.
    - Starting `DeepSAT` from another learner's network (e.g.
      `RuleSetInit(Pypper())`); `init` is ignored so far. As solver
      phases: the solver tries each weight first with the start's value.
      A first try (2026-11-08, tic-tac-toe, Glucose 4) gave exactly the
      same search path as without -- possibly Glucose ignores PySAT's
      phases, or the start's unconnected random filler rules (64
      conditions where Pypper has 26) got in the way; to check. Not as
      an upper bound: the start isn't an exact fit. With soft examples
      (`size_penalty`) it is: done, the start is the first solution.
  - `DeepNeuroFuzzy` (added 2026-11-08): the relaxed network (product
    AND/OR, weights in [0, 1]) trained by gradient descent in numpy,
    thresholded to 0/1 after every epoch, gradients checked against
    finite differences. Our alternating layers (the paper also puts AND
    and OR nodes side by side in a layer). Random binarization as
    described (binarized weights not updated) left the network empty
    from the paper's small start -- the OR layer's weights never got
    above 0.5 (tic-tac-toe: 0.65 test, 0 edges); with the
    straight-through estimator (now the default) tic-tac-toe 0.984 like
    Pypper, vote 0.920, breast-cancer 0.793 (one fold each, NNF(1)).
    KRK illegality (1000) stays near the majority (0.68; with more epochs
    0.80 train, 0.62 test) -- as for local search (0.69). Deep networks
    on small data need more steps. Open: the paper's hyperparameters
    (only a first guess here: 100 epochs, batch 64, Adam 0.01, L2 1e-4,
    rate 0.75), an epoch budget relative to the data size, the demo.
  - `size_penalty` for all four (done 2026-11-08): errors + lambda per
    backbone edge. Counting all edges made local search collapse to the
    empty network (removing a dead edge always "improved"); counting the
    backbone fixed that, but on mini-batches the penalty still stripped
    rules of conditions the other rows need (tic-tac-toe lambda 0.1: 112
    training errors); applied only on the whole training set, local
    search got better than without it (tic-tac-toe, one fold: 14 errors
    and 95 edges, test 0.964, against 34, 225, 0.917). DeepNeuroFuzzy's
    L1 removes few redundant edges -- in the fuzzy network they still
    lower the loss a little; open.
  - `DeepGenetic` (added 2026-11-07): a memetic algorithm with crossover
    of aligned networks: node-wise, subtree (a node with its cone),
    mixed, none. One seed, 10-fold, nine datasets, mean accuracy NNF(1) /
    NNF(3): nodes 0.925 / 0.932, subtree 0.907 / 0.929, mixed 0.911 /
    0.922, none 0.908 / 0.922 -- but the same node-wise configuration
    with other random numbers gave 0.916 / 0.925, and single datasets
    move by up to 0.06 (monk-2), so several seeds are needed to decide
    whether crossover helps. Not tried: deep rules (a top-level AND node
    with its cone) as genes, Pittsburgh style.
- Sanity checks against the papers' numbers, with their encodings;
  otherwise our framework's encoding comes first.
- Consistency of rules (start and moves) is logical: what the
  `DataSpec`'s constraints imply (`contradictions`). A `DataSpec` of bare
  feature names has none. Later, as a `DataSpec` method: inferring
  constraints from data (e.g. features never true together, or one
  implying another), so that such specs get them too.
- Results so far (2026-11-07, 10-fold, nine datasets; mean accuracy):
  Pypper 0.929; local search NNF(1) 0.811, NNF(3) 0.925; GA NNF(1) 0.925,
  NNF(3) 0.932; both started from Pypper's rules 0.948-0.952 -- the gain
  comes from the start, the GA adds little over local search there.
- A demo of the anytime behaviour (`trace=True`): test accuracy against
  effort, averaged over folds, for local search and the GA, random and
  Pypper-seeded, NNF(1) and NNF(3), on larger datasets where a deeper
  structure may pay off -- adult, kr-vs-kp, mushroom, connect-4-binary,
  kropt -- with the artificial datasets as a control. Needs budgets
  (`max_evaluations`, `max_seconds`) to compare on equal effort. Whether
  the largest ones can be run is open.
- `RuleSetInit` puts a rule set into the first layer and passes it up
  unchanged; a start that factors the rules (shared conditions as hidden
  concepts) would use the deeper levels from the start.
- Hidden concepts (full demo run on ab7a5b3 and follow-ups, 2026-11-08):
  no learner has built one yet. KRK (`[128, 16, 4]`): NeuroFuzzy from
  Pypper beats Pypper (0.858 vs 0.792) with one hidden concept, "rook on
  the black king's file or rank" (an OR of the 15 `eq` pairs), the layers
  above pass it through; with lambda 0.25 it adds flat fragments of "kings
  adjacent" (0.846), not "files close AND ranks close". Parity-4 is
  learnt flat by NNF(1) as well (8 rules fit `[20]`); parity-6-of-12
  (1000 rows) defeats every learner (no subset of the six features tells
  anything, so a group's parity earns nothing). `parity_groups` (at least
  2 of 4 groups odd, 1000 rows, lambda 1) rewards the groups one by one,
  but no learner finds them either: groups of three, everything near the
  majority (0.688), local search on `[32, 8, 8]` best with 0.733; groups
  of two (plus 4 irrelevant), local search on `[32, 8, 8]` 0.874 against
  Pypper 0.825 and flat local search 0.833 -- with 6-8 used OR nodes, but
  none of them a group's parity (each agrees with every group's on about
  half the rows): a tangle of partial terms, not the concepts.
- Next steps (the user, 2026-11-08), new algorithms rather than changes to
  the current ones:
  1. A better start than Pypper, whose rules are too simple (general,
     short): FOIL's, say.
  2. A search -- and possibly a start -- driven by examples (e.g. a
     misclassified example picks the moves, rules seeded from examples):
     a different algorithm.
- `DeepBlameRevision` (experimental, from 2026-11-09, branch
  `experimental/blame-revision`): the user's direction 2 -- targets for
  the hidden nodes from the errors, layer-wise, with recruitment of free
  nodes seeded from examples. What was needed: soft demands for "one
  input has to hold" (`soft_weight`; else every recruit grew into "always
  true"), only nodes in use as new inputs (a free AND node is true
  everywhere), and a child counted as cheap only if its repair breaks no
  protected row (else an old rule was stretched to examples it does not
  fit and nothing was recruited). First results (one split, 120 s, empty
  start, batch 256, soft weight 0.25): `parity_groups` 2-of-4x2 exactly
  (56 edges; layer 1 holds the eight odd terms, layer 2 mixes groups --
  not the clean 42-edge decomposition), 2-of-4x3 0.765, parity-4 exact in
  a second; artificial-nnf-36 0.912 (Pypper 0.946), tic-tac-toe 0.906
  (0.990), KRK 0.695 (0.830). Open: it stops within seconds in a local
  optimum (restarts or perturbation); seeds for one-hot data (the most
  specific seed is mostly negated values); starting from FOIL.
  Run time (2026-11-09): 6-35 ms per step (`[32, 8, 8]` to `[256, 64,
  64]`), a run ends by `patience` within 2-15 s; wider or deeper networks
  do not help (the same optima, sometimes none). Restarts (`restarts`,
  `-1` until `max_seconds`; 50-120 runs in 300 s) do where runs differ:
  2-of-4x2 exact with the clean concepts -- for the class "no", the four
  groups' even parities in layer 2, triples of them in layer 3, 40
  edges --, tic-tac-toe 0.99 (one run: 0.885); not where all runs end
  alike (KRK 0.69, 2-of-4x3 0.68).
- **Remove `car-binary` and `connect-4-binary` again** once those
  comparisons are done: they are binary versions of `car` and
  `connect-4`, only there to match the papers' setups (also adjust
  `test_deep_datasets_are_the_papers`, which counts 29 `deep` entries).

## Other planned work

- **Decision tables** (Kohavi, *The Power of Decision Tables*, ECML 1995),
  as the paper's rule-models section discusses them: a schema of attributes
  (chosen by a wrapper search) and one rule per cell that holds training
  examples, each testing every schema attribute -- an oblivious decision
  tree. The model is a `DisjointRuleSet` with a default (the majority class)
  for cells without training examples. Possible routes: a native learner,
  and/or an importer for Weka's `DecisionTable` next to `J48Importer`.
  Not started.

- **Measure the SeCo-search-vs-ENDER speed gap on `demos/covering_boosting.py`
  itself, now that the Boolean-representation threshold-counting speedup
  above has landed (2026-10-07).** Not done yet -- the chain-counting
  work was checked against CN2/PFoil/PFossil/Pypper on other datasets, not
  by rerunning this demo's own `WeightedCovering`+WRAcc configuration,
  so the gap this note describes hasn't actually been re-measured.
  `demos/covering_boosting.py`
  (2026-10-06/07) found `WeightedCovering` configs (plain `SeCo`,
  `BeamSearch(beam_width=5)`) taking tens of seconds per fit
  (`Weighted-Add+WRAcc`: 86.55s) against boosting's sub-3s
  (`Boomer@100`: 2.48s) on the same datasets. The likely reason:
  `ENDER`/`Boomer`'s `_grow` scores every candidate feature in one
  vectorized `(t * c).T @ Xf` matrix multiply per round, while
  `BeamSearch`/`HillClimbing` still count each candidate condition with
  its own pass (the "High priority" item above) -- not a difference
  between weighted and removal covering, or a beam-width effect (both
  use the same beam width here). But `ENDER`'s vectorization only works
  because it commits to a dense `data.X` matrix, exactly what the
  0.3.0 plan above calls out as *not* representation-independent for
  `ENDER` itself. So two numbers worth comparing once the chain-counting
  speedup lands: how much the SeCo searches close the gap, and
  (separately, from the 0.3.0 work) how much `ENDER` slows down once
  it's made representation-independent instead of assuming a matrix --
  i.e. whether the two converge from opposite directions.
- **`CoverageDifference` is ~3x faster than `Accuracy` for the same
  ranking** (measured 2026-10-06: 1,000,000 `.score()` calls each,
  0.07s vs. 0.21s) -- confirms the docstrings' claim that
  `Accuracy = (CoverageDifference + n_neg) / (n_pos + n_neg)` exactly
  (checked over 200,000 random stats, zero floating-point difference).
  For 0.3.0, check every built-in algorithm's default heuristic for a
  hot-path use of `Accuracy` swappable to `CoverageDifference` without
  changing which rule wins. None currently defaults to `Accuracy`
  (grepped) -- start with `Pypper` specifically, since it was asked
  about directly: its defaults are `Precision` (pruning and the IREP
  stopping criterion) and `FoilGain` (growing), neither with the same
  isometrics as `CoverageDifference`, so this needs checking whether a
  cheaper-but-equivalent swap exists there too, not assuming the same
  substitution applies.
- **Inverted heuristics** (Stecher, Janssen & Fürnkranz): a refinement
  evaluated in the coverage space of its parent rule instead of the empty
  rule. Implement in `pyrulearn.heuristics`, then add to the
  `heuristic_isometrics` and `heuristic_comparison` demos. (Also listed in
  the README.)
- **README**: subsections under *Native learning algorithms* for the
  learners that are only mentioned in tables so far -- CBA, CMAR, IDS and
  RuleFit.
- **Java LORD on large data**: it timed out on every fold of `adult` and
  `connect-4` in the SeCo learners comparison. The likely reason is the
  number of binary features (about 250-280 with negation features);
  check with one fit without negation features.
- **`pyrulearn/learners/seco.py`**: the module docstring still says that
  rule-set optimization is not implemented, although `SeCo` has
  `optimization=` and Pypper uses `ReplaceReviseOptimization`.
- **Error-correcting output codes** (Dietterich & Bakiri 1995) as a
  further multi-class decomposition, next to one-vs-rest, ordered
  one-vs-rest and pairwise: each class gets a code word, one binary
  model is learned per code bit, and an example is assigned the class
  whose code word is nearest to the models' predictions. It would fit as
  another model type of `DecomposingLearner` (`fit(data, model=...)`) and
  as another method in the multi-class decomposition demo. With ternary
  codes (a class may also be left out of a bit's problem), one-vs-rest
  and pairwise are themselves special cases, so they could be expressed
  as codes; the codes to try: exhaustive, random and dense/sparse random
  codes, decoded by Hamming distance or by the members' confidences. The
  paper reviews ECOC in its binarization section (2026-10-30), without
  an experiment so far.
- **Accuracy as percentages, four digits** (the user's preference,
  2026-10-10): every demo report, verdict sentence and plot axis shows
  accuracy as ``86.31%``, not ``0.863`` -- `experiments.report`'s
  `render_results_table` (its one `float_format` for all columns) and
  each demo's own summary tables and sentences. With the planned clean
  re-run of the demos.
- **Demo plots**: on logarithmic axes, the plain-number tick labels also
  label the minor ticks, which overlap where the axis spans several
  decades (e.g. `demos/representations_plots/size_index_build.png`).
- **Growth and the pairwise speed-up, controlled** (2026-10-15). The
  multi-class demo estimates each base learner's growth b from three nested
  subsamples of one training fold per dataset and sets the predicted one-vs-rest
  / pairwise ratio against the measured one: the points follow the line only
  roughly (PFossil, b about 0.5, never gains; PFoil and Pypper, b about 1.3,
  gain on some datasets; Pypper on kropt is far off, 0.34 measured vs. 4.36
  predicted). Rule learners' run times are noisy -- one rule more or less
  changes the time a lot and the quality hardly. A controlled experiment would
  vary the training-set size on a single dataset (several sizes, several
  repetitions) and compare the decompositions there. Open question as well:
  whether a covering learner's cost depends on the total size of a sub-problem
  at all, or rather on its number of positive examples (the rules it learns)
  times its size (the cost of each search) -- then one-vs-rest and pairwise
  would cost about the same, about n^2 in both cases.
- **The number of classes in CN2's (and Opus') Laplace heuristic under a
  decomposition** (open, 2026-10-15). `_ClassCountLaplace` sets
  `Laplace(n_classes=c)` once per `fit`, from the whole problem, so every
  sub-problem uses it: on `letter`, c = 26 in each pair of classes, although a
  pair's data has only two. Intended: c = the number of the original classes
  with examples in the local problem -- 26 for one-vs-rest, 26, 25, ..., 2 for
  the stages of ordered, 2 for a pair. Preferred way (not done yet): the
  class count as part of `RuleStats`, set by the search from the data, and an
  unset `Laplace()` using it -- which raises the general question whether
  statistics should describe the current problem (shrinking under covering)
  or the whole dataset. Alternative: c from the prior, (n_pos + n_neg) / n_pos,
  which needs no plumbing but changes during covering. Measured on one fold of
  `letter` (pairwise): c = 26 gives 89.75% test accuracy in 56 s, c = 2 78.65%
  in 14 s -- c = 2 leaves slightly more of each pair's class uncovered (96.4%
  vs. 99.8%), and with 26 classes the pairwise vote turns that into many
  errors (the true class leads the best wrong one by only 1.5-2 of 25 votes).
  The m-estimate with the pair's own prior (m = 26: 89.85%, m = 8: 89.65% in
  35 s) shows it is the heuristic's strength, not the class count, that
  matters for accuracy. Until this is settled, the multi-class demo uses PFoil
  instead of CN2 as its superlinear learner.
- **Per-class feature sets** (a perspective, from `demos/numeric_discretization.py`,
  2026-10-21; not planned yet). `PerClass` discretizes each class against
  the rest but pools the thresholds, so a learner working on one class
  searches every class's cuts, most of them placed for other classes. The
  demo's section 2 suggests that this pooling dilutes the gain of
  per-class cuts: for the decision tree and FUSINTER, per class is
  competitive and sometimes best (not significantly), at several times
  the features of a global discretization of similar accuracy. The idea:
  keep one feature space with every class's thresholds, but when a
  learner works on a dedicated target class (`ConceptCascade`,
  `ConceptSet`, a covering loop with `target_class`), mask out the
  features placed for other classes; a pairwise learner (class c vs. d)
  would see the features of c and d. Perhaps expressed as constraints
  derived from the classes on the `DataSpec`, so the learners need no
  code of their own. The same mechanism -- a feature set per learning
  task -- would also be a natural place for feature selection, which
  pyrulearn does not do at all yet.
- **Parallel learners** (not planned for now, 2026-10-14): every native
  learner runs on one thread. Some would parallelize well because they
  consist of independent searches: `PyLORD` runs one rule search per
  training example, and a random forest (`SKLRandomForest`, the candidate
  pool of `RuleFit` and of the pool distillers) grows independent trees;
  other ensembles and pool generators may qualify too. The reference
  implementations already do this: `mlrl-boomer` uses every core and the
  Java LORD half of them by default -- the demos limit both to one thread
  so that fit times compare like with like. A learner's own threads and
  `run_cv(n_jobs=...)` compete for the same cores, so the two would need
  to be coordinated (e.g. a thread budget per worker).

## Design decisions and known behaviour

Not to-dos: choices made deliberately, with the behaviour they imply, so
they aren't reopened by accident.

- **Negations are stored explicitly, never derived as complements
  (2026-10-13).** A negated feature is a feature of its own in every
  representation, not computed at query time as "all rows minus the
  positive feature's rows". The cost shows in `SparseDataRepresentation`
  with negation: every complement is stored, so half the cells are true
  (`adult`, n=32,000: 41.9 MiB against 5.9 without negation) -- for sparse
  data, build without negations (`without_negations()`). Note that a
  negation isn't always the exact complement anyway: with missing values
  (`MissingStrategy.NEVER_COVERS`) a row can satisfy neither `>=t` nor
  `<t`.
- **ENDER-style matrix-multiply counting, tried for `HorizontalBitmapRepresentation`
  and parked, not adopted (2026-10-07).** Following up on the numeric-
  threshold-counting item above, `HorizontalBitmapRepresentation.batch_cover_counts`
  and an experimental `pyrulearn.learners.seco._score_children_matmul`
  score *every* open feature of a round in one matrix multiply (`pos_w
  @ X[:, features]`), the way `ENDER`'s `_grow` does -- more general than
  `chain_cover_counts` (works across attributes and on nominal features
  too, not just one numeric chain), and simpler code (no monotonic-chain
  bookkeeping, no missing-value edge case -- a bug in the chain version's
  first draft, where `<t` was wrongly derived as `>=t`'s complement
  without accounting for `MissingStrategy.NEVER_COVERS` rows where
  neither holds).
  - **Where it wins cleanly**: `HillClimbing` (one lineage, one matmul
    per round, the clean ENDER-shaped case) -- consistently at or above
    `chain_cover_counts`'s speed (PFoil 1.07x, PFossil 1.19x, Pypper
    1.12x, Slipper 1.03x on `sonar`; PFossil 1.07x on `spambase`), and far
    ahead on nominal-heavy data regardless of search (`kr-vs-kp`: 1.74x
    over the old path vs. chain's 1.12x, since chain-counting can't help
    nominal features at all).
  - **Where it regresses**: `BeamSearch` with a real beam (>1) on a
    feature-rich dataset. Each of the (up to `beam_width`) surviving
    parents per round pays its own full-open-feature matmul, with no
    sharing across parents -- cost scales roughly with `beam_width x
    open_feature_count`, unlike `chain_cover_counts`'s smaller,
    per-attribute batches. Measured on `spambase` (its widest feature
    count): `beam_width=1` 1.08x (matmul wins), `beam_width=3` 0.98x
    (a wash), `beam_width=5` -- CN2's actual default -- 0.76x (matmul
    24% *slower*), `beam_width=10` 0.83x. `PyLORD`'s default
    `BeamSearch(beam_width=1)` lands in the good case; CN2/AQR/PFoil's
    beam variants, at their real default widths on feature-rich data,
    don't.
  - **GPU**: considered and rejected without prototyping, on the
    numbers already in hand. The matmul is a vector-times-matrix
    (GEMV, memory-bandwidth-bound, not the large compute-bound GEMM a
    GPU's parallelism actually pays off on), called once per beam
    member per round and synchronized back to Python immediately
    after (`tp == 0` and heuristic scoring need the result right away)
    -- exactly the per-call-overhead-dominated pattern the beam-width
    finding above already shows losing to CPU's smaller, more frequent
    calls. Would need batching across beam members into one bigger
    matrix-matrix product *and* keeping data device-resident across a
    whole search to plausibly pay off -- a real redesign, not a backend
    swap, not attempted.
  - **`batch_cover_counts`'s own fancy-indexing cost, found and then
    un-fixed (2026-10-08).** It slices `self.X[:, feature_indices]`
    before the matmul -- `feature_indices` is a list/array, so that's
    fancy indexing, which numpy always copies. Measured in isolation:
    26x slower than matmul-ing against the *whole* `self.X` and
    selecting `feature_indices` out of the small `(n_features,)` result
    afterward instead -- the same "score everything, mask the result"
    trick `pyrulearn.learners.cpar.DenseCPAR`'s (then `CPAR`'s) own native code already uses
    (see the CPAR entry below). Tried as a fix, measured *worse* in
    every real case re-checked (`spambase`, CN2/`BeamSearch` at
    `beam_width` 1/3/5/10, and `PFossil`/`HillClimbing` -- previously
    matmul's cleanest win at 1.07x, now 0.90x): the isolated benchmark
    used a case with ~98% of features still open, which is *not*
    representative of `BeamSearch`/`HillClimbing`'s own open set -- a
    `SeCo` search's open set shrinks fast, not slowly: `Rule.specialize`'s
    constraint-aware closure means fixing *one* numeric threshold forces
    every other threshold (and negation) of that same attribute out of
    the open set in one step, so most calls happen far narrower than
    98% open, where slicing-then-matmul's smaller FLOP count beats
    matmul-then-select's constant full-width cost despite the slice's
    own overhead. Reverted back to slicing; the 26x number is real and
    not wrong, just measured on a case unrepresentative of *this*
    caller, generalized from without checking the real workload first
    -- the exact mistake the "verify, don't assume" discipline elsewhere
    in this file is supposed to catch, caught here one step too late
    (after proposing the fix, not before).

    Checked directly *why* this doesn't contradict `CPAR`'s (now `DenseCPAR`'s) own native
    code using the identical "score everything, mask after" trick
    successfully (prompted by a direct question, not found
    unprompted): on `spambase` (n=4601, k=998), `CPAR`'s real
    full-matrix `_grow` is **15x faster** than a slice-first variant of
    the exact same search (102ms vs. 1547ms, identical 27 bodies) --
    the opposite result from `BeamSearch`. The difference isn't
    single-lineage-vs-branching or small-n-vs-large-n (both were
    candidate explanations, both wrong) -- it's *how fast the open set
    shrinks*. `CPAR` works directly on `data.X` with no constraint
    propagation at all: each condition removes exactly the one feature
    just used from consideration, nothing else, so with `max_length`
    typically ~5, its open set stays at roughly `k - 5` -- essentially
    full width -- for the entire search. That's genuinely the ~98%-open
    regime the isolated benchmark measured; it just isn't `SeCo`'s
    regime, where one condition can remove a dozen+ features at once.
    Same trick, opposite verdict, because the two callers' open sets
    behave completely differently -- not a property of the trick itself.
  - **Decision**: keep `chain_cover_counts` as `HorizontalBitmapRepresentation`'s
    shipped default (safe across beam widths); keep `batch_cover_counts`/
    `_score_children_matmul` in the codebase as a validated-but-not-wired-in
    alternative, not deleted -- revisit if a beam-width/feature-count-aware
    dispatch between the two (or restricting matmul to `HillClimbing`-based
    learners specifically) turns out to be worth the complexity.
    **Superseded (2026-10-08)**: all searches now count through
    `batch_cover_counts` after all, once they stopped building every
    child eagerly -- see "Build only the children a search follows".
    These measurements were taken on top of that eager construction.
- **A `chain_cover_counts` for `NListRepresentation`: first attempt was
  wrong, reverted before being committed (2026-10-07).** Tried porting
  the Boolean trick directly: scan the loosest (shallowest) open
  threshold's own tree nodes once, then check the stricter thresholds'
  bits against those same nodes' `path_mask`. Wrong, because the
  PPC-tree orders features most-frequent-first and a monotonic chain's
  loosest threshold is, by construction, the *most frequent* member --
  so it sits *shallower* than the stricter ones, and their bits are only
  added to the path *further down*, past that node, not yet present
  there. Silent failure mode, not a crash: the undercounted thresholds
  came back `tp == 0` and were dropped into the "dead" set before ever
  reaching the per-child brute-force check, so the check -- which only
  verified *surviving* children, never *why something didn't survive* --
  reported zero mismatches on a run that was actually dropping most of
  a chain's thresholds (caught by comparing old-vs-new *full fits*
  instead, which disagreed). Scanning the *deepest* (strictest) member
  instead doesn't fix it either: its own node list only has rows
  satisfying *every* member of the chain, missing everything that
  satisfies some but not all of it -- correct for nothing shallower than
  itself. Reverted rather than shipped; the representation's existing
  per-threshold `refine_cover` path is untouched.

  **Second attempt, correct but not a win, also reverted (same day).**
  Intersect the one scan's candidate rows against each stricter
  threshold's already-precomputed, unmasked `_row_idx[feature]` directly
  (no tree traversal, no dense matrix) -- `{parent ∧ f_k} = {parent ∧
  f_1} ∩ row_idx[f_k]` holds exactly, since `f_k` implies `f_1`. Also
  had to mirror `refine_cover`'s *own* two branches for `f_1` itself
  (cheap filter vs. re-anchor scan), not just assume the expensive one
  -- a second, narrower instance of the same mistake: `f_1` is only
  safe to re-anchor-scan when it's deeper than every condition already
  in the rule, same as any other feature. Verified correct this time
  with a stricter check than the first attempt's -- explicitly
  confirming every *dropped* ("dead") candidate was brute-force-checked
  too, not just survivors, the exact gap that let the first version's
  bug through unnoticed -- across 6 datasets, 1844 search-node calls,
  zero mismatches either way; full test suite (777 passed).

  Despite that, timing (CN2, one fit each, old vs. new) was mostly a
  regression: `segment` 0.73x, `diabetes` 0.62x, `ionosphere` 0.80x,
  `banknote-authentication` 0.68x, `sonar` 1.07x (even), `kr-vs-kp`
  1.60x (the one clear win, nominal-heavy -- same pattern as
  `HorizontalBitmapRepresentation`'s matmul vs. chain finding above). Likely
  cause: `np.isin(candidate_rows, row_idx[f])` for each of the `m - 1`
  remaining thresholds costs more than expected -- its cost scales with
  both array sizes, re-sorting `row_idx[f]` fresh on every call, which
  on a numeric-heavy dataset with many samples can exceed what it saves
  versus the tree re-scans it replaces. A pre-sorted `row_idx` (sorted
  once, cached, e.g. at `_build` time) plus `np.searchsorted` instead of
  `np.isin` might close the gap -- not tried. Reverted rather than left
  as an always-on regression (`_score_children` dispatches on whether
  `chain_cover_counts` exists at all, with no beam-width/feature-count
  gating the way the matmul decision got -- so, unlike that one, there
  was no way to "keep it but not wire it in" without also changing the
  dispatch).
- **How rules are annotated** (decided 2026-10-01). Every rule is
  annotated with the training examples its body covers, independently of
  the other rules (`annotate_rules`, `annotate_default_rule`) -- so a
  rule's counts mean the same in every model it belongs to, and don't
  change when a rule set is reordered or turned into a decision list. The
  default rule's body is empty, so it is counted over all training
  examples (Weka, by contrast, prints JRip's rules and default over the
  examples that reach them). Two alternatives were considered and not
  taken: counting each rule over the examples it actually decides in its
  model (more informative for decision lists, but model-dependent, so
  every conversion or reordering would have to recount -- needing the
  data), and counting it over the examples it was learned from (which
  would make the same rule set annotated differently depending on the
  order it was learned in).
- **The pairwise weighted vote does poorly** -- a result of the
  multi-class decomposition demo, explained in its report. It weights a
  pair model's vote by the Laplace estimate of the deciding rule; most
  pair models decide most examples by their default rule, whose counts
  (over all of its pair's examples) make its weight just its class's
  share of the pair -- about 0.5 for a balanced pair, below 0.5 when the
  default is the pair's smaller class. Counting the default over the
  examples it decides brings the weighted vote up to the plain vote, but
  not above (the demo's final check). Kept as it is, following the
  decision above.
- **The Prolog rendering of a decision list's default rule isn't
  faithful standalone Prolog** (noted 2026-10-03, building the
  tutorial). `died(X) :- true.` reads as "died holds for every X", but
  the model only means it for rows no earlier rule matched -- the
  faithful body would be `died(X) :- not(survived_rule_1(X)), not(...),
  ...`, and in general every non-first rule of a decision list would
  need the same growing prefix of negated earlier rules to be exactly
  right, which would make longer lists unreadable. A cleaner fix exists
  -- give every rule (default included) a shared head like `class(X,
  <target>)` and a trailing cut (`!`), which is faithful first-match-wins
  Prolog without any explicit negation -- but it only reads correctly if
  each rule's printed stats are also its cut-aware decided coverage, not
  its independent one, which is the "how rules are annotated" choice
  above, deliberately not reopened. No change: the current rendering
  stays informal/approximate for a decision list's non-first rules, most
  visibly its default.

## Pypper: a general overhaul against JRip

**Planned (2026-10-26).** Pypper and Weka's JRip are about equally
accurate on the existing benchmarks (`demos/seco_learners_comparison.py`,
27 datasets: 81.32% vs. 80.97%, multi-class 78.45% vs. 78.17%), but they get
there differently: Pypper learns fewer rules (4.8 vs. 5.6) and is about ten
times faster (0.09 vs. 0.97 s per fit). The speed is worth keeping; the
differences in behaviour need understanding. Known or suspected ones:

- **Stopping.** Pypper stops a class's covering loop at the first pruned
  rule with a precision below 0.5 on the covering scope (IREP's criterion)
  or a per-rule MDL test (`EncodingLengthRestriction`), instead of Cohen's
  "total description length more than 64 bits over the minimum seen" with
  its retroactive trim-back.
- **Decision lists with head search break down on many classes.** Here the
  benchmarks do *not* show Pypper doing well, and JRip hasn't been tested
  on the same task: on `kropt` (18 classes) Pypper's decision lists whose
  rules are the best of any class or of each class reach only 19-22%
  (10-45 rules), its seeded one 61% (989 rules); on `letter`, 68% against
  84% (`demos/rule_models_report.md`, 10-fold). Two suspects: the stopping
  criterion above -- in one loop over all classes the first weak rule ends
  the whole list, so a criterion made for one class against the rest may
  have to end only that class's search, or be judged against the class's
  prior instead of 0.5 -- and the head search under FOIL gain, which
  chooses the head at the first refinement and keeps it (see "Rules of any
  class"). First diagnosis: a single fit on `kropt` logging why the loop
  ends. For comparison, JRip on the same data -- its own class cascade, as
  it has no decision list over all classes.
- **Numeric attributes.** JRip chooses thresholds during the search, Pypper
  works on an a-priori discretization (`TreeDiscretizer(max_intervals=8)`
  in the demos); this alone may explain part of the differences, and
  should be separated from the others (e.g. by giving JRip the same
  binarized data).
- **The optimization phase.** `ReplaceReviseOptimization` and no
  post-optimization residual IREP\* (see `Pypper`'s docstring); to be
  compared with JRip's replacement/revision and its DL-based choice.
- **Pairwise with the larger class as target** learns no rules for many
  pairs (see "Open problems from the multi-class demo").

Plan: analyze the differences on **multi-class** datasets first, where they
show most (class order, rules per class, where the accuracies part), with
JRip and Pypper on the same binarized data, then trace single datasets rule
by rule against JRip's source.

## Open problems from the multi-class demo

- **Pypper with the larger class as a pair's target** learns no rules for
  many pairs (128 of 210 on `primary-tumor`), and every rule-less model
  votes for its pair's smaller class -- so the plain vote predicts the
  rarest classes and collapses (accuracy 0.003). Why Pypper learns
  nothing there is open.

## Release checklist

- The version number is in four places: `pyproject.toml`,
  `pyrulearn/__init__.py` (`__version__`), `CITATION.cff`, and the BibTeX
  entry in the README's *How to cite*. Grep for the old number before
  tagging.
- `CHANGELOG.md`: turn *Unreleased* into `## X.Y.Z (date)`.
- Tag `vX.Y.Z`.
