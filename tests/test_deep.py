import itertools
import time

import numpy as np
import pandas as pd
import pytest

from pyrulearn.data import DataSpec, DataSpecBuilder, HorizontalBitmapRepresentation, SparseDataRepresentation
from pyrulearn.data.io import build_dataspec, encode
from pyrulearn.learners.deep import (
    DeepBlameRevision, DeepGenetic, DeepLocalSearch, DeepNeuroFuzzy, DeepSAT, ModelInit, RandomInit, RuleSetInit, contradictions,
    nnf_forward,
)
from pyrulearn.models import ConceptModel, DeepModel
from pyrulearn.rule import Rule

SPEC = DataSpec(list("abcdefg"))
X = np.array(list(itertools.product([0, 1], repeat=7)), dtype=bool)
a, b, c, d, e, f, g = X.T
SEIP = (a | b) & (c | d) & (e | (f & g))
DATA = HorizontalBitmapRepresentation(SPEC, X, np.where(SEIP, "pos", "neg"))
# the same through `build_dataspec`: Boolean attributes, `x` and `not x`, mutually exclusive
FRAME = pd.DataFrame(X, columns=list("abcdefg"))
NEG_SPEC = build_dataspec(FRAME).build()
NEG_X = np.asarray(encode(NEG_SPEC, FRAME).X, dtype=bool)
NEG_DATA = HorizontalBitmapRepresentation(NEG_SPEC, NEG_X, DATA.y)


def _seip_weights():
    W1 = np.zeros((6, 7), dtype=int)
    for node, inputs in enumerate([[0], [1], [2], [3], [4], [5, 6]]):
        W1[node, inputs] = 1
    W2 = [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]]
    return [W1, W2, [[1, 1, 1]], [[1]]]


def test_nnf_forward_matches_the_model():
    dm = DeepModel(SPEC, ["pos"], weights=_seip_weights())
    values = nnf_forward(dm.weights, X)
    assert [v.shape for v in values] == [(6, 128), (3, 128), (1, 128), (1, 128)]
    assert values[-1][0].tolist() == SEIP.tolist()
    assert all((v == m).all() for v, m in zip(values, dm.node_values(DATA)))


def test_flip_errors_are_those_of_flipping():
    rng = np.random.default_rng(0)
    learner = DeepLocalSearch()
    for _ in range(10):
        shapes, below = [], 9
        for size in [int(x) for x in rng.integers(1, 6, size=rng.choice([1, 3]))] + [2]:
            shapes.append((size, below))
            below = size
        weights = [rng.random(s) < 0.4 for s in shapes]
        Xr, targets, sw = rng.random((40, 9)) < 0.5, rng.random((2, 40)) < 0.5, rng.random(40)
        conflicts = rng.random((9, 9)) < 0.3
        np.fill_diagonal(conflicts, False)
        values = nnf_forward(weights, Xr)
        for layer in range(len(weights)):
            cf = conflicts if layer == 0 else None
            got = learner._flip_errors(weights, values, Xr.T, layer, targets, sw, cf)
            for k, j in np.ndindex(weights[layer].shape):
                flipped = [w.copy() for w in weights]
                DeepLocalSearch._flip(flipped, layer, k, j, cf)
                want = ((nnf_forward(flipped, Xr)[-1] != targets) * sw).sum()
                assert got[k, j] == pytest.approx(want)


def test_local_search_learns_the_formula():
    m = DeepLocalSearch(layers=[6, 3, 1]).fit(NEG_DATA)
    assert isinstance(m, DeepModel) and m.layers == [6, 3, 1]
    assert m.outputs == ["pos"] and m.labels == ["neg", "pos"]  # two classes: the rarer one is the output
    assert (m.predict(NEG_DATA) == NEG_DATA.y).mean() > 0.9
    assert m.provenance.learner == "DeepLocalSearch"


def test_two_classes_of_equal_size_default_to_the_other_one():
    y = np.where(a, "x", "y")  # 64 each
    m = DeepLocalSearch(layers=[2]).fit(HorizontalBitmapRepresentation(SPEC, X, y))
    assert m.outputs == ["x"] and m.default_prediction == "y"
    assert (m.predict(HorizontalBitmapRepresentation(SPEC, X)) == y).all()


def test_consistent_moves_never_build_a_contradictory_rule():
    m = DeepLocalSearch(layers=[8]).fit(NEG_DATA)
    for row in m.weights[0]:
        names = {NEG_SPEC.feature_names[j] for j in np.flatnonzero(row)}
        assert not any(f"not {x}" in names and x in names for x in "abcdefg")


def test_contradictions_come_from_the_constraints():
    b = DataSpecBuilder()
    b.add_numeric("x", [3, 5])
    b.add_nominal("c", ["r", "g", "b"])
    b.add_boolean("f")
    spec = b.build()
    pairs = {tuple(sorted((spec.feature_names[i], spec.feature_names[j])))
             for i, j in zip(*np.nonzero(contradictions(spec)))}
    assert {("x<3", "x>=3"), ("x<3", "x>=5"), ("x<5", "x>=5"), ("c=g", "c=r"), ("c!=r", "c=r"),
            ("f", "not f")} <= pairs
    assert ("c!=g", "c!=r") not in pairs and ("c=r", "f") not in pairs
    assert not contradictions(DataSpec(["f", "not f"])).any()  # bare names: no constraints


def test_local_search_keeps_a_perfect_start():
    initial = DeepModel(SPEC, ["pos"], weights=_seip_weights())
    m = DeepLocalSearch(layers=[6, 3, 1], init=ModelInit(initial)).fit(DATA)
    assert all((w == i).all() for w, i in zip(m.weights, initial.weights))
    with pytest.raises(ValueError, match="initial model has layers"):
        DeepLocalSearch(layers=[4, 3, 1], init=ModelInit(initial)).fit(DATA)


def test_rule_set_init_starts_as_the_rules_dnf():
    rules = ConceptModel([Rule([0, 2, 4], target="pos", dataspec=SPEC), Rule([1, 3, 5, 6], target="pos", dataspec=SPEC)])
    expected = (a & c & e) | (b & d & f & g)
    for layers in ([4], [4, 3, 2], [5, 2, 3, 2, 1]):
        start = RuleSetInit(rules).weights(DATA, ["pos"], [w.shape for w in DeepModel(SPEC, ["pos"], layers).weights],
                                           np.random.default_rng(1))
        assert nnf_forward(start, X)[-1][0].tolist() == expected.tolist()
    with pytest.raises(ValueError, match="first layer of 1"):
        DeepLocalSearch(layers=[1], init=RuleSetInit(rules)).fit(DATA)


def test_rule_set_init_takes_a_learner():
    from pyrulearn.learners.seco import Pypper

    m = DeepLocalSearch(layers=[12], init=RuleSetInit(Pypper()), n_epochs=0).fit(NEG_DATA)
    pypper = Pypper().fit(NEG_DATA)
    assert (m.predict(NEG_DATA) == pypper.predict(NEG_DATA)).mean() >= 0.99


def test_a_subclass_can_override_the_start():
    class Empty(DeepLocalSearch):
        def _initialize(self, data, labels, shapes, rng):
            return [np.zeros(s, dtype=bool) for s in shapes]

    m = Empty(layers=[3], n_epochs=0, max_flips=0).fit(DATA)
    assert not any(w.any() for w in m.weights)


def test_local_search_is_reproducible_and_takes_any_representation():
    learner = DeepLocalSearch(layers=[4, 2, 2], n_epochs=1, random_state=3)
    m1, m2 = learner.fit(DATA), learner.fit(SparseDataRepresentation(SPEC, X, DATA.y))
    assert all((w1 == w2).all() for w1, w2 in zip(m1.weights, m2.weights))


def test_local_search_on_several_classes():
    y = np.where(a & b, "ab", np.where(c, "c", "other"))
    assert DeepLocalSearch(n_epochs=1).fit(HorizontalBitmapRepresentation(SPEC, X, y)).layers == [3]  # ceil(sqrt(7))
    m = DeepLocalSearch(layers=[6]).fit(HorizontalBitmapRepresentation(NEG_SPEC, NEG_X, y))
    assert m.outputs == ["ab", "c", "other"]
    assert (m.predict(HorizontalBitmapRepresentation(NEG_SPEC, NEG_X)) == y).mean() > 0.9


def test_beck_start_and_moves_are_available():
    m = DeepLocalSearch(layers=[6, 3, 1], moves="attribute", init=RandomInit(min_rule_length=0, consistent=False)).fit(NEG_DATA)
    assert (m.predict(NEG_DATA) == NEG_DATA.y).mean() > 0.8


def test_alignment_undoes_a_permutation_of_the_hidden_nodes():
    rng = np.random.default_rng(0)
    for sizes in ([6, 5, 4, 1], [7, 1]):
        shapes, below = [], 7
        for size in sizes:
            shapes.append((size, below))
            below = size
        a_ = [rng.random(s) < 0.4 for s in shapes]
        b_ = [w.copy() for w in a_]
        for layer in range(len(b_) - 1):
            perm = rng.permutation(b_[layer].shape[0])
            b_[layer], b_[layer + 1] = b_[layer][perm], b_[layer + 1][:, perm]
        back = DeepGenetic._aligned(a_, b_, X)
        assert all((x == y).all() for x, y in zip(nnf_forward(back, X), nnf_forward(a_, X)))


def test_genetic_learns_the_formula_and_is_reproducible():
    learner = DeepGenetic(layers=[6, 3, 1], population=8, generations=5)
    m = learner.fit(NEG_DATA)
    assert isinstance(m, DeepModel) and (m.predict(NEG_DATA) == NEG_DATA.y).mean() > 0.9
    assert all((w1 == w2).all() for w1, w2 in zip(m.weights, learner.fit(NEG_DATA).weights))


def test_subtree_crossover_takes_a_node_with_its_cone():
    rng = np.random.default_rng(3)
    w = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], bool),  # 4 rules over 3 inputs
         np.array([[1, 0, 0, 1], [0, 1, 1, 0]], bool),                  # OR 0 <- rules 0, 3; OR 1 <- rules 1, 2
         np.array([[1, 0], [0, 1]], bool), np.array([[1, 1]], bool)]
    cones = {tuple(tuple(np.flatnonzero(m).tolist()) for m in DeepGenetic._cone(w, 2, np.random.default_rng(s)))
             for s in range(20)}
    assert cones == {((0, 3), (0,), (0,), ()), ((1, 2), (1,), (1,), ())}
    a_ = [np.zeros_like(x) for x in w]
    child = DeepGenetic(crossover="subtree")._crossover(a_, w, X[:, :3], rng)
    taken = [np.flatnonzero(c.any(axis=1)).tolist() for c in child]
    assert taken[-1] == [] and len(taken[1]) == 1 and len(taken[0]) == 2  # one OR node and its two rules, or one rule


@pytest.mark.parametrize("crossover", ["subtree", "mixed", "none"])
def test_genetic_with_other_crossovers(crossover):
    m = DeepGenetic(layers=[6, 3, 1], population=8, generations=5, crossover=crossover).fit(NEG_DATA)
    assert (m.predict(NEG_DATA) == NEG_DATA.y).mean() > 0.8  # the majority class has 0.65


def test_genetic_keeps_a_perfect_seed():
    seed = ModelInit(DeepModel(SPEC, ["pos"], weights=_seip_weights()))
    m = DeepGenetic(layers=[6, 3, 1], population=4, generations=3, seeds=[seed]).fit(DATA)
    assert (m.predict(DATA) == DATA.y).all()


@pytest.mark.parametrize("learner", [DeepLocalSearch(layers=[6, 3, 1], trace=True),
                                     DeepGenetic(layers=[6, 3, 1], population=6, generations=4, trace=True)])
def test_trace_records_every_new_best(learner):
    m = learner.fit(NEG_DATA)
    trace = m.trace
    assert len(trace) >= 2
    assert all(a.train_error > b.train_error for a, b in zip(trace, trace[1:]))
    assert all(a.evaluations <= b.evaluations and a.seconds <= b.seconds for a, b in zip(trace, trace[1:]))
    assert trace[0].evaluations == 128  # the first start, evaluated on every row
    assert all((w == v).all() for w, v in zip(trace[-1].weights, m.weights))  # the last point is the model
    first = m.trace_model(0)
    assert (first.predict(NEG_DATA) != NEG_DATA.y).sum() == trace[0].train_error
    assert DeepLocalSearch(layers=[3]).fit(NEG_DATA).trace is None


def test_sat_fits_the_data_exactly():
    pytest.importorskip("pysat")
    for layers in ([6, 3, 1], [20]):
        m = DeepSAT(layers=layers).fit(DATA)
        assert (m.predict(DATA) == DATA.y).all()


def test_maxsat_finds_the_smallest_network():
    pytest.importorskip("pysat")
    m = DeepSAT(layers=[6, 3, 1], minimize="weights").fit(DATA)  # one first-layer node per condition
    assert (m.predict(DATA) == DATA.y).all()
    assert m.n_edges == 17  # (a | b) & (c | d) & (e | (f & g)): 16 edges, plus the output's
    # fewest conditions in a DNF: each of the 8 terms contains a prime implicant (a|c|e, ...), 4x3 + 4x4
    m = DeepSAT(layers=[8], minimize="inputs").fit(NEG_DATA)
    assert (m.predict(NEG_DATA) == NEG_DATA.y).all() and m.weights[0].sum() == 28


def test_sat_with_several_classes_and_conflicting_rows():
    pytest.importorskip("pysat")
    y = np.where(a & b, "ab", np.where(c, "c", "other"))
    m = DeepSAT(layers=[6]).fit(HorizontalBitmapRepresentation(NEG_SPEC, NEG_X, y))
    assert (m.predict(HorizontalBitmapRepresentation(NEG_SPEC, NEG_X)) == y).all()
    # the all-zero row three times more, once with the other label: the majority (its own) is kept
    other = "pos" if DATA.y[0] == "neg" else "neg"
    Xc = np.vstack([X, X[:1], X[:1], X[:1]])
    yc = np.concatenate([DATA.y, [DATA.y[0], DATA.y[0], other]])
    m = DeepSAT(layers=[6, 3, 1]).fit(HorizontalBitmapRepresentation(SPEC, Xc, yc))
    assert (m.predict(DATA) == DATA.y).all()
    with pytest.raises(ValueError, match="identical rows"):
        DeepSAT(layers=[6, 3, 1], conflicts="raise").fit(HorizontalBitmapRepresentation(SPEC, Xc, yc))
    with pytest.raises(ValueError, match="unsatisfiable"):
        DeepSAT(layers=[1]).fit(DATA)  # one rule can't express the formula


@pytest.mark.parametrize("first", ["and", "or"])
@pytest.mark.parametrize("make", [
    lambda layers, first: DeepLocalSearch(layers=layers, first=first),
    lambda layers, first: DeepGenetic(layers=layers, first=first, population=6, generations=3),
])
def test_search_learners_on_any_depth_and_first_layer(make, first):
    for layers in ([6, 6], [6, 4, 3], [8]):
        m = make(layers, first).fit(NEG_DATA)
        assert m.first == first and m.layers == layers
        out = nnf_forward(m.weights, NEG_X, first)[-1][0]
        assert ((m.predict(NEG_DATA) == "pos") == out).all()  # the model is the learner's network
        assert (m.predict(NEG_DATA) == NEG_DATA.y).mean() > 0.6


def test_sat_on_the_papers_architecture():
    pytest.importorskip("pysat")
    m = DeepSAT(layers=[6, 6], minimize="weights", max_conflicts=1000).fit(DATA)  # AND 6, OR 6, AND output
    assert (m.predict(DATA) == DATA.y).all() and m.n_edges == 16  # as in the long paper's Fig. 4
    for first, layers in (("or", [4]), ("or", [3, 3]), ("and", [])):
        y = np.where(a & (b | c), "pos", "neg")  # a CNF with two clauses, a DNF with two rules
        data = HorizontalBitmapRepresentation(SPEC, X, y)
        if not layers:
            with pytest.raises(ValueError, match="unsatisfiable"):  # one rule can't express it
                DeepSAT(layers=layers, first=first).fit(data)
            continue
        m = DeepSAT(layers=layers, first=first).fit(data)
        assert (m.predict(data) == y).all() and m.first == first


def test_maxsat_linear_search_is_anytime():
    pytest.importorskip("pysat")
    # the optimum (16 edges) comes early; proving it would take half a minute more, which a
    # budget of conflicts per solver call cuts off -- the same on every machine, unlike a time limit
    m = DeepSAT(layers=[6, 6], minimize="weights", max_conflicts=1000, trace=True).fit(DATA)
    sizes = [p.size for p in m.trace]
    assert sizes == sorted(sizes, reverse=True) and len(set(sizes)) == len(sizes)  # each one smaller
    assert sizes[-1] == 16 == m.n_edges and all(p.train_error == 0 for p in m.trace)
    assert all((m.trace_model(i).predict(DATA) == DATA.y).all() for i in range(len(m.trace)))
    assert [p.evaluations for p in m.trace] == sorted(p.evaluations for p in m.trace)  # solver calls
    y = np.where(a & (b | c), "pos", "neg")
    rc2 = DeepSAT(layers=[2, 2], minimize="weights", maxsat="rc2").fit(HorizontalBitmapRepresentation(SPEC, X, y))
    # two first-layer nodes can't hold a, b and c each: (a & b) | (a & c), 4 + 2 edges, plus the output's
    assert rc2.n_edges == 7
    small = DeepSAT(layers=[6, 6], minimize="weights", max_conflicts=100, trace=True).fit(DATA)
    assert [p.size for p in small.trace] == sizes[:len(small.trace)]  # the same path, cut off earlier
    # a time limit: whatever the machine's speed, an exact fit -- or no network yet
    try:
        quick = DeepSAT(layers=[8], minimize="inputs", max_seconds=0.5).fit(NEG_DATA)
        assert (quick.predict(NEG_DATA) == NEG_DATA.y).all()
    except TimeoutError:
        pass


def test_parameters_are_checked():
    with pytest.raises(ValueError):
        DeepLocalSearch(moves="random")
    with pytest.raises(ValueError):
        DeepLocalSearch(max_flips=-1)
    with pytest.raises(ValueError):
        RandomInit(init_prob=1.5)
    with pytest.raises(ValueError):
        DeepGenetic(population=3, elite=3)
    with pytest.raises(ValueError, match="first"):
        DeepLocalSearch(first="xor")
    with pytest.raises(ValueError, match="minimize"):
        DeepSAT(minimize="all")
    with pytest.raises(ValueError, match="maxsat"):
        DeepSAT(maxsat="lsu")
    with pytest.raises(ValueError, match="crossover"):
        DeepGenetic(crossover="uniform")
    with pytest.raises(ValueError, match="seeds"):
        DeepGenetic(population=2, elite=0, seeds=[RandomInit()] * 3)
    with pytest.raises(ValueError, match="two classes"):
        DeepLocalSearch().fit(HorizontalBitmapRepresentation(SPEC, X, np.array(["x"] * 128)))


def test_maxsat_with_soft_examples_trades_errors_for_edges():
    pytest.importorskip("pysat")
    y = np.where(a & (b | c), "pos", "neg")  # a DNF of two rules: 4 conditions, 2 output edges
    data = HorizontalBitmapRepresentation(SPEC, X, y)
    exact = DeepSAT(layers=[2], minimize="weights").fit(data)
    assert exact.n_edges == 6
    for maxsat in ("linear", "rc2"):
        # an edge costs little: the exact fit again
        m = DeepSAT(layers=[2], minimize="weights", size_penalty=0.1, maxsat=maxsat).fit(data)
        assert (m.predict(data) == y).all() and m.n_edges == 6
        # an edge costs more than all the errors: nothing at all, every positive example wrong
        m = DeepSAT(layers=[2], minimize="weights", size_penalty=100, maxsat=maxsat).fit(data)
        assert m.n_edges == 0 and (m.predict(data) != y).sum() == (y == "pos").sum()
        # in between: the rule a alone, 2 edges and 16 errors (36) beats nothing (48 errors, 48)
        m = DeepSAT(layers=[2], minimize="weights", size_penalty=10, maxsat=maxsat).fit(data)
        assert m.n_edges == 2 and (m.predict(data) != y).sum() == 16
        # lambda 0: the errors alone
        m = DeepSAT(layers=[2], size_penalty=0, maxsat=maxsat).fit(data)
        assert (m.predict(data) == y).all()
    with pytest.raises(ValueError, match="needs minimize"):
        DeepSAT(layers=[2], size_penalty=1)


def test_maxsat_with_soft_examples_takes_conflicting_rows():
    pytest.importorskip("pysat")
    Xc = np.vstack([X, X[:3]])
    y = np.where(a & b, "pos", "neg")
    yc = np.concatenate([y, np.where(y[:3] == "pos", "neg", "pos")])  # three rows twice, with both labels
    data = HorizontalBitmapRepresentation(SPEC, Xc, yc)
    with pytest.raises(ValueError, match="identical rows"):
        DeepSAT(layers=[1], conflicts="raise").fit(data)
    m = DeepSAT(layers=[1], minimize="weights", size_penalty=0.2).fit(data)
    assert m.n_edges == 3 and (m.predict(data) != yc).sum() == 3  # a & b; one of each pair wrong


def test_maxsat_with_soft_examples_is_anytime_from_the_start():
    pytest.importorskip("pysat")
    m = DeepSAT(layers=[6, 3, 1], minimize="weights", size_penalty=0.01, max_conflicts=1000,
                trace=True).fit(DATA)
    costs = [p.train_error + 0.01 * p.size for p in m.trace]
    assert m.trace[0].evaluations == 1  # the first solver call has a network: no example must fit
    assert costs == sorted(costs, reverse=True) and len(set(costs)) == len(costs)  # each one cheaper


def test_maxsat_with_soft_examples_starts_from_init():
    pytest.importorskip("pysat")
    # the formula itself as the start: the optimum already, nothing cheaper is found
    def fit(start):
        return DeepSAT(layers=[6, 3, 1], minimize="weights", size_penalty=0.1, max_conflicts=1000, trace=True,
                       init=ModelInit(DeepModel(SPEC, ["pos"], weights=start))).fit(DATA)

    m = fit(_seip_weights())
    assert all((w == np.asarray(s, dtype=bool)).all() for w, s in zip(m.weights, _seip_weights()))
    assert len(m.trace) == 1 and m.trace[0].evaluations == 0  # the start, before any solver call
    # a start with errors: the result is never worse
    start = [np.array(w) for w in _seip_weights()]
    start[0][0] = 0  # the rule a loses its condition: always true
    m = fit(start)
    costs = [p.train_error + 0.1 * p.size for p in m.trace]
    assert costs[0] > costs[-1] and m.trace[0].evaluations == 0


def test_target_class_picks_the_output():
    pytest.importorskip("pysat")
    y = np.where(a & b, "pos", "neg")  # "pos" is the less frequent class: the output by default
    data = HorizontalBitmapRepresentation(NEG_SPEC, NEG_X, y)
    assert DeepSAT(layers=[2]).fit(data).outputs == ["pos"]
    m = DeepSAT(layers=[2], target_class="neg", minimize="weights").fit(data)  # not a or not b
    assert m.outputs == ["neg"] and m.default_prediction == "pos" and (m.predict(data) == y).all()
    assert m.n_edges == 4
    for learner in (DeepLocalSearch(layers=[2], target_class="neg"), DeepGenetic(layers=[2], target_class="neg")):
        assert learner.fit(data).outputs == ["neg"]
    with pytest.raises(ValueError, match="not a class"):
        DeepSAT(layers=[2], target_class="maybe").fit(data)


def test_maxsat_start_counts_only_its_backbone():
    pytest.importorskip("pysat")
    w1, w2, w3, w4 = (np.array(w) for w in _seip_weights())
    w1 = np.vstack([w1, np.ones((1, 7), dtype=int)])  # a seventh rule, all conditions, feeding nothing
    w2 = np.hstack([w2, np.zeros((3, 1), dtype=int)])
    start = DeepModel(SPEC, ["pos"], weights=[w1, w2, w3, w4])
    m = DeepSAT(layers=[7, 3, 1], minimize="weights", size_penalty=0.1, max_conflicts=1000, trace=True,
                init=ModelInit(start)).fit(DATA)
    assert m.trace[0].evaluations == 0 and m.trace[0].size == 17  # not 24: the filler's edges are dropped
    assert m.n_edges == 17 and (m.predict(DATA) == DATA.y).all()


@pytest.mark.parametrize("first", ["and", "or"])
@pytest.mark.parametrize("loss", ["mse", "log"])
def test_neuro_fuzzy_gradients_match_finite_differences(first, loss):
    rng = np.random.default_rng(0)
    nf = DeepNeuroFuzzy(first=first, loss=loss)
    W = [rng.uniform(0.05, 0.95, s) for s in [(4, 6), (3, 4), (1, 3)]]
    Xf = (rng.random((10, 6)) < 0.5).astype(float)
    T = (rng.random((10, 1)) < 0.5).astype(float)

    def total_loss(W):
        h = Xf
        for l, w in enumerate(W):
            is_and = (l % 2 == 0) == (first == "and")
            f = 1 - w[None] * ((1 - h[:, None, :]) if is_and else h[:, None, :])
            p = np.prod(np.maximum(f, 1e-6), axis=2)
            h = p if is_and else 1 - p
        o = np.clip(h, 1e-6, 1 - 1e-6)
        return ((o - T) ** 2).mean() if loss == "mse" else (-(T * np.log(o) + (1 - T) * np.log(1 - o))).mean()

    grads = nf._gradients(W, Xf, T, np.ones(10))
    for l in range(3):
        for idx in np.ndindex(*W[l].shape):
            up = [w.copy() for w in W]
            down = [w.copy() for w in W]
            up[l][idx] += 1e-6
            down[l][idx] -= 1e-6
            assert grads[l][idx] == pytest.approx((total_loss(up) - total_loss(down)) / 2e-6, abs=1e-7)


def test_neuro_fuzzy_learns_the_formulas():
    y = np.where(a & (b | c), "pos", "neg")
    data = HorizontalBitmapRepresentation(SPEC, X, y)
    m = DeepNeuroFuzzy(layers=[2], trace=True).fit(data)
    assert (m.predict(data) == y).all() and m.pruned().n_edges == 6  # a & b, a & c
    errors = [p.train_error for p in m.trace]
    assert errors == sorted(errors, reverse=True) and errors[-1] == 0
    # deeper: more steps on 128 rows
    m = DeepNeuroFuzzy(layers=[6, 3, 1], epochs=300, batch_size=16).fit(DATA)
    assert (m.predict(DATA) == DATA.y).mean() >= 0.98
    again = DeepNeuroFuzzy(layers=[6, 3, 1], epochs=300, batch_size=16).fit(DATA)
    assert all((w == v).all() for w, v in zip(m.weights, again.weights))  # reproducible


def test_neuro_fuzzy_starts_from_init():
    start = DeepModel(SPEC, ["pos"], weights=_seip_weights())
    m = DeepNeuroFuzzy(layers=[6, 3, 1], epochs=5, init=ModelInit(start)).fit(DATA)
    assert (m.predict(DATA) == DATA.y).all()  # the start's edges begin above 0.5: exact from the first epoch
    with pytest.raises(ValueError, match="loss"):
        DeepNeuroFuzzy(loss="hinge")


@pytest.mark.parametrize("make, minimal", [
    (lambda lam: DeepLocalSearch(layers=[4], size_penalty=lam, trace=True), True),
    (lambda lam: DeepGenetic(layers=[4], generations=5, size_penalty=lam, trace=True), True),
    # the fuzzy network keeps redundant edges: in it they still lower the loss a little
    (lambda lam: DeepNeuroFuzzy(layers=[4], size_penalty=lam, trace=True), False),
])
def test_size_penalty_trades_errors_for_edges(make, minimal):
    y = np.where(a & (b | c), "pos", "neg")  # 48 positives; a & b, a & c: 6 edges
    data = HorizontalBitmapRepresentation(SPEC, X, y)
    small = make(0.01).fit(data)
    assert (small.predict(data) == y).all()
    if minimal:
        assert small.pruned().n_edges == 6  # the exact DNF, nothing extra
    costs = [p.train_error + 0.01 * p.size for p in small.trace]
    assert costs == sorted(costs, reverse=True)  # each trace point cheaper
    big = make(100.0).fit(data)
    assert big.pruned().n_edges == 0  # an edge costs more than all the errors together
    with pytest.raises(ValueError, match="size_penalty"):
        make(-1.0)


@pytest.mark.parametrize("make", [
    lambda: DeepLocalSearch(layers=[32, 8, 2], n_epochs=10 ** 6, max_seconds=1.0),
    lambda: DeepGenetic(layers=[32, 8, 2], generations=10 ** 6, max_seconds=1.0),
    lambda: DeepNeuroFuzzy(layers=[32, 8, 2], epochs=10 ** 6, max_seconds=1.0),
])
def test_time_budget_ends_the_search(make):

    start = time.perf_counter()
    m = make().fit(DATA)
    assert time.perf_counter() - start < 10  # the budget, plus one step and the model
    assert m.predict(DATA).shape == (128,)



# -- DeepBlameRevision -------------------------------------------------------------------

def _boolean_data(label):
    """All rows of eight Boolean features, `x` and `not x`, labeled by `label(x1, ..., x8)`."""
    bits = np.array(list(itertools.product([0, 1], repeat=8)), dtype=bool)
    frame = pd.DataFrame(bits, columns=[f"x{i}" for i in range(1, 9)])
    frame["class"] = np.where(label(*bits.T), "yes", "no")
    spec = build_dataspec(frame, target="class").build()
    return encode(spec, frame, frame["class"].to_numpy())


def test_blame_revision_builds_a_dnf_from_the_empty_network():
    data = _boolean_data(lambda x1, x2, x3, x4, *_: (x1 & x2) | (x3 & ~x4))
    for layers in ([8], [8, 4, 2]):
        model = DeepBlameRevision(layers=layers, size_penalty=0.1, target_class="yes").fit(data)
        assert (model.predict(data) == data.y).all()
        if layers == [8]:  # the two rules, nothing else
            assert model.pruned().n_edges == 6


def test_blame_revision_learns_parity():
    # a single condition tells nothing about parity, so greedy covering fails; the
    # examples seed the rules here
    data = _boolean_data(lambda x1, x2, x3, x4, *_: x1 ^ x2 ^ x3 ^ x4)
    model = DeepBlameRevision(layers=[20], size_penalty=1.0, target_class="yes").fit(data)
    assert (model.predict(data) == data.y).all()
    assert model.pruned().n_edges == 8 * 4 + 8


def test_blame_revision_stays_empty_when_nothing_pays_for_its_edges():
    data = _boolean_data(lambda x1, x2, x3, x4, *_: x1 & x2 & x3 & x4)  # 16 of 256 rows
    model = DeepBlameRevision(layers=[8, 4, 2], size_penalty=50.0, target_class="yes").fit(data)
    assert model.pruned().n_edges == 0


def test_blame_revision_is_reproducible_traced_and_bounded():
    data = _boolean_data(lambda x1, x2, x3, x4, x5, *_: (x1 ^ x2) & (x3 | x5))
    kw = dict(layers=[16, 4, 2], size_penalty=0.5, batch_size=32, trace=True, random_state=3)
    m1, m2 = DeepBlameRevision(**kw).fit(data), DeepBlameRevision(**kw).fit(data)
    assert [w.tolist() for w in m1.weights] == [w.tolist() for w in m2.weights]
    errors = [p.train_error for p in m1.trace]
    assert len(errors) >= 2 and m1.trace[0].train_error > m1.trace[-1].train_error
    assert DeepBlameRevision(max_steps=0, trace=True, **{k: v for k, v in kw.items() if k != "trace"}).fit(
        data).pruned().n_edges == 0  # no step: the empty start
    with pytest.raises(ValueError):
        DeepBlameRevision(soft_weight=0)
    with pytest.raises(ValueError):
        DeepBlameRevision(batch_size=0)


def test_blame_revision_local_edit_covers_a_row_at_once():
    learner = DeepBlameRevision()
    F = np.array([[1, 1, 0, 0],   # input 0
                  [0, 1, 1, 0],   # input 1
                  [0, 1, 0, 1]],  # input 2
                 dtype=bool)
    # an AND of all three is true on row 1 only; it should also hold on row 0 (where inputs
    # 1 and 2 are in the way) and not on rows 2, 3: dropping both at once, no single drop helps
    S = learner._local_edit(np.ones(3, dtype=bool), F, np.array([1, 1, 0, 0]), np.ones(4), True, None, 0.01)
    assert S.tolist() == [True, False, False]
    # an OR, by De Morgan: true on rows 0, 1 and not 2, 3 -- input 0 alone
    S = learner._local_edit(np.zeros(3, dtype=bool), F, np.array([1, 1, 0, 0]), np.ones(4), False, None, 0.01)
    assert S.tolist() == [True, False, False]


def test_blame_revision_restarts_keep_the_best_run():
    data = _boolean_data(lambda x1, x2, x3, x4, x5, x6, *_: (x1 ^ x2) & (x3 ^ x4) | (x5 & x6))
    kw = dict(layers=[16, 4, 4], size_penalty=1.0, batch_size=64, target_class="yes", random_state=1)

    def cost(model):
        return int((model.predict(data) != data.y).sum()) + model.pruned().n_edges

    one = DeepBlameRevision(**kw).fit(data)
    more = DeepBlameRevision(restarts=4, trace=True, **kw).fit(data)
    assert cost(more) <= cost(one)  # the first run is the same, the others can only do better
    assert [p.seconds for p in more.trace] == sorted(p.seconds for p in more.trace)
    timed = DeepBlameRevision(restarts=-1, max_seconds=1.0, **kw)
    t = time.perf_counter()
    timed.fit(data)
    assert time.perf_counter() - t < 3.0
    with pytest.raises(ValueError, match="needs max_seconds"):
        DeepBlameRevision(restarts=-1)
    with pytest.raises(ValueError):
        DeepBlameRevision(restarts=-2)
