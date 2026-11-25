import numpy as np
import pytest

from chebifier.hex_bounded import (
    bounded_marginals,
    brute_force_marginals,
    graph_from_edges,
    has_violation,
)


def random_graph(rng, n):
    hier = [(p, c) for c in range(n) for p in range(c) if rng.random() < 0.25]
    excl = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.10]
    g = graph_from_edges(n, hier, excl)
    for i, j in excl:
        if (g.succ[:, i] & g.succ[:, j]).any():
            return None
    return g


def test_golden_example():
    g = graph_from_edges(5, [(0, 2), (1, 3)], [(0, 1)], ["D", "C", "H", "S", "P"])
    f = np.log(np.array([4.0, 1.0, 3.0, 2.0, 0.5]))
    want = np.array([0.8, 0.15, 0.6, 0.10, 1 / 3])
    bf, n_states = brute_force_marginals(g, f)
    assert n_states == 10
    assert np.allclose(bf, want, atol=1e-9)
    bd = bounded_marginals(g, f, budget=10**5)
    assert bd["exhausted"]
    assert np.allclose(bd["lb"], want, atol=1e-9)
    assert np.allclose(bd["ub"], want, atol=1e-9)


def test_no_edges_is_independent_sigmoids():
    rng = np.random.default_rng(7)
    f = rng.normal(0, 1.5, size=8)
    g = graph_from_edges(8, [], [])
    want = 1 / (1 + np.exp(-f))
    assert np.allclose(brute_force_marginals(g, f)[0], want, atol=1e-12)
    bd = bounded_marginals(g, f, budget=10**5)
    assert np.allclose(bd["lb"], want, atol=1e-12)
    assert np.allclose(bd["ub"], want, atol=1e-12)


def test_complete_exclusion_is_softmax():
    rng = np.random.default_rng(7)
    n = 8
    f = rng.normal(0, 1.5, size=n)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    g = graph_from_edges(n, [], pairs)
    want = np.exp(f) / (1 + np.exp(f).sum())
    assert np.allclose(brute_force_marginals(g, f)[0], want, atol=1e-12)
    bd = bounded_marginals(g, f, budget=10**5)
    assert np.allclose(bd["lb"], want, atol=1e-12)
    assert np.allclose(bd["ub"], want, atol=1e-12)


@pytest.mark.parametrize("seed", range(40))
def test_partial_bounds_bracket_exact(seed):
    rng = np.random.default_rng(1000 + seed)
    g = random_graph(rng, int(rng.integers(2, 12)))
    if g is None:
        pytest.skip("inconsistent graph")
    f = rng.normal(0, 2.0, size=g.n)
    bf, _ = brute_force_marginals(g, f)
    full = bounded_marginals(g, f, budget=10**5)
    assert full["exhausted"]
    assert np.allclose(full["lb"], bf, atol=1e-9)
    assert np.allclose(full["ub"], bf, atol=1e-9)
    for budget in (1, 3, 8, 25):
        b = bounded_marginals(g, f, budget=budget)
        assert (b["lb"] <= bf + 1e-9).all()
        assert (b["ub"] >= bf - 1e-9).all()
        assert (b["lb"] >= -1e-12).all() and (b["ub"] <= 1 + 1e-9).all()
        assert b["z_lo"] <= b["z_hi"] + 1e-9


def test_forest_bound_is_exact_on_a_tree():
    g = graph_from_edges(4, [(0, 1), (1, 2), (2, 3)], [])
    f = np.full(4, 2.0)
    exact = bounded_marginals(g, f, budget=10**5)
    assert exact["exhausted"]
    root = bounded_marginals(g, f, budget=0)
    assert np.isclose(root["z_hi"], exact["z_hi"], atol=1e-9)
    assert root["z_hi"] < np.logaddexp(0.0, f).sum() - 0.3


def test_intervals_shrink_with_budget():
    rng = np.random.default_rng(3)
    g = random_graph(rng, 10)
    while g is None:
        g = random_graph(rng, 10)
    f = rng.normal(0, 2.0, size=10)
    prev = None
    for budget in (1, 2, 4, 8, 16, 32, 64, 128):
        width = float(
            (lambda b: (b["ub"] - b["lb"]).mean())(
                bounded_marginals(g, f, budget=budget)
            )
        )
        if prev is not None:
            assert width <= prev + 1e-12
        prev = width


@pytest.mark.parametrize("seed", range(30))
def test_lower_bound_at_neutral_is_consistent(seed):
    """Thresholding the lower bounds at 1/2 needs no repair pass to come out consistent.

    The marginals are taken over legal states only, so P(A) + P(B) <= 1 for disjoint A, B and
    P(A) <= P(B) for a subclass A of B. Both survive the bound: two disjoint classes cannot both
    have lb > 1/2 without their true marginals exceeding the total mass, and lb is monotone along
    subsumption. Neither inequality gives anything at a threshold below 1/2, which is why the
    ensemble's operating point is rescaled onto the neutral point instead of being thresholded at.
    """
    rng = np.random.default_rng(seed)
    g = random_graph(rng, 9)
    if g is None:
        pytest.skip("inconsistent graph")
    f = rng.normal(0, 2.0, size=9)
    res = bounded_marginals(g, f, budget=6, threshold=0.5)
    assert not has_violation(g, res["lb"] > 0.5)


def test_decision_survives_rescaling_round_trip():
    """A marginal just over 1/2 has to stay positive once it is mapped back and cast to float32.

    The pipeline decides `score > threshold` on float32, so a lower bound a hair above the neutral
    point must not land back exactly on the threshold - that would silently drop the label.
    """
    from chebifier.ensemble.level1 import rescale_from_threshold, rescale_to_threshold

    threshold = 0.2735091278070138
    scores = np.array([0.0, threshold, 0.5 + 1e-9, 1.0])
    assert np.allclose(rescale_to_threshold(scores, threshold)[1], 0.5, atol=1e-6)
    back = rescale_from_threshold(np.array([0.0, 0.5, 0.5 + 1e-9, 1.0]), threshold)
    assert np.isclose(back[1], threshold, atol=1e-6)
    nudged = np.where(
        np.array([False, False, True, True]),
        np.maximum(back, np.nextafter(np.float32(threshold), np.float32(1))),
        np.minimum(back, np.float32(threshold)),
    ).astype(np.float32)
    assert nudged[2] > np.float32(threshold)
    assert nudged[1] <= np.float32(threshold)
