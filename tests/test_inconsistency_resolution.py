import networkx as nx
import pytest
import torch

from chebifier.hex_bounded import BoundedHexSmoother
from chebifier.ilr import GodelILRSmoother, LukasiewiczILRSmoother
from chebifier.inconsistency_resolution import (
    NEUTRAL,
    ScoreBasedPredictionSmoother,
    densified_exclusion_pairs,
)
from chebifier.predict import apply_inconsistency_resolution

LABELS = ["1", "2", "3", "4", "5"]
SMOOTHERS = [
    ScoreBasedPredictionSmoother,
    GodelILRSmoother,
    LukasiewiczILRSmoother,
    BoundedHexSmoother,
]
EXTRA_ARGS = {BoundedHexSmoother: {"processes": 1}}


def make_graph(edges):
    g = nx.DiGraph()
    g.add_nodes_from(LABELS)
    for child, parent in edges:
        g.add_edge(child, parent, relation="is_a")
    return g


HIERARCHY = [("3", "1"), ("4", "1"), ("5", "2")]


@pytest.fixture
def disjoint_file(tmp_path):
    path = tmp_path / "disjoint.csv"
    path.write_text("1,2\n")
    return [str(path)]


@pytest.fixture
def empty_disjoint_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    return [str(path)]


def violations(decisions, smoother):
    succ = smoother.label_successors[0]
    bad = succ.unsqueeze(0) & decisions.unsqueeze(2) & ~decisions.unsqueeze(1)
    subsumption = int(bad.sum())
    pairs = densified_exclusion_pairs(
        LABELS, smoother.label_successors, smoother.disjoint_groups
    )
    disjointness = 0
    if pairs.shape[0]:
        disjointness = int(
            (decisions[:, pairs[:, 0]] & decisions[:, pairs[:, 1]]).sum()
        )
    return subsumption, disjointness


def test_godel_is_max_over_subclasses(empty_disjoint_file):
    graph = make_graph(HIERARCHY)
    scores = torch.tensor([[0.3, 0.6, 0.9, 0.35, 0.7]])
    godel = GodelILRSmoother(graph, LABELS, empty_disjoint_file)

    succ = godel.label_successors
    expected = (
        torch.where(torch.transpose(succ, 1, 2), scores.unsqueeze(1), -torch.inf)
        .max(dim=2)
        .values
    )
    assert torch.allclose(godel(scores.clone()), expected, atol=1e-3)


@pytest.mark.parametrize("cls", [GodelILRSmoother, LukasiewiczILRSmoother])
def test_output_stays_a_probability(cls, disjoint_file):
    graph = make_graph(HIERARCHY)
    scores = torch.tensor([[0.0, 0.2, 1.0, 0.45, 0.62], [0.95, 0.88, 0.4, 0.55, 0.3]])

    out = cls(graph, LABELS, disjoint_file)(scores.clone())
    assert bool(((out >= 0.0) & (out <= 1.0)).all())


def test_godel_raises_parent_to_child(empty_disjoint_file):
    graph = make_graph(HIERARCHY)
    godel = GodelILRSmoother(graph, LABELS, empty_disjoint_file)
    scores = torch.tensor([[0.1, 0.5, 0.95, 0.5, 0.5]])
    out = godel(scores)
    assert out[0, 0] == pytest.approx(float(scores[0, 2]), abs=1e-3)


def test_lukasiewicz_splits_disjointness(disjoint_file):
    graph = make_graph([])
    smoother = LukasiewiczILRSmoother(graph, LABELS, disjoint_file)
    refined = smoother(torch.tensor([[0.8, 0.7, 0.0, 0.0, 0.0]]))
    assert refined[0, 0].item() == pytest.approx(0.55, abs=1e-3)
    assert refined[0, 1].item() == pytest.approx(0.45, abs=1e-3)


def test_godel_zeroes_disjointness_loser(disjoint_file):
    graph = make_graph([])
    smoother = GodelILRSmoother(graph, LABELS, disjoint_file)
    refined = smoother(torch.tensor([[0.8, 0.7, 0.0, 0.0, 0.0]]))
    assert refined[0, 0].item() == pytest.approx(0.8, abs=1e-3)
    assert refined[0, 1].item() < 1e-3


@pytest.mark.parametrize("cls", [ScoreBasedPredictionSmoother, LukasiewiczILRSmoother])
def test_no_residual_violations(cls, disjoint_file):
    graph = make_graph(HIERARCHY)
    smoother = cls(graph, LABELS, disjoint_file)
    scores = torch.tensor(
        [
            [0.25, 0.25, 0.9, 0.4, 0.62],
            [0.85, 0.78, 0.45, 0.58, 0.4],
            [0.1, 0.62, 0.75, 0.75, 0.95],
        ]
    )
    out = smoother(scores.clone())
    subsumption, disjointness = violations(out > NEUTRAL, smoother)
    assert subsumption == 0
    assert disjointness == 0


@pytest.mark.parametrize("cls", SMOOTHERS)
def test_uncovered_label_is_filled_from_subclass(cls, empty_disjoint_file):
    graph = make_graph(HIERARCHY)
    smoother = cls(graph, LABELS, empty_disjoint_file, **EXTRA_ARGS.get(cls, {}))
    scores = torch.tensor([[0.12, 0.52, 0.97, 0.5, 0.5]])
    valid = torch.tensor([[False, True, True, True, True]])

    out = smoother(scores.clone(), valid)
    assert out[0, 0].item() > NEUTRAL


@pytest.mark.parametrize("cls", SMOOTHERS)
def test_uncovered_label_is_neither_evidence_nor_positive(cls, empty_disjoint_file):
    graph = make_graph(HIERARCHY)
    smoother = cls(graph, LABELS, empty_disjoint_file, **EXTRA_ARGS.get(cls, {}))
    scores = torch.tensor([[0.2, 0.5, 0.3, 0.9, 0.5]])
    valid = torch.tensor([[True, True, True, False, True]])

    out = smoother(scores.clone(), valid)
    assert out[0, 3].item() <= NEUTRAL
    assert out[0, 0].item() <= NEUTRAL


def test_score_based_weighs_confidence_not_raw_distance(empty_disjoint_file):
    graph = make_graph(HIERARCHY)
    smoother = ScoreBasedPredictionSmoother(
        graph, LABELS, empty_disjoint_file, threshold=0.2
    )
    # "3" is a subclass of "1": raising "1" to 0.55 spans 0.35 of the wide positive side, lowering
    # "3" to 0.05 spans 0.15 of the narrow negative side, but the latter is the confident one
    out = smoother(torch.tensor([[0.05, 0.05, 0.55, 0.05, 0.05]]))
    assert out[0, 0].item() <= 0.2
    assert out[0, 2].item() <= 0.2


def test_filled_label_becomes_covered(disjoint_file):
    graph = make_graph(HIERARCHY)
    smoother = ScoreBasedPredictionSmoother(graph, LABELS, disjoint_file)
    aggregated = {
        "net_score": torch.tensor([[0.12, 0.3, 0.97, 0.5, 0.5]]),
        "has_valid_predictions": torch.tensor([[False, True, True, False, True]]),
    }

    out = apply_inconsistency_resolution(smoother, LABELS, aggregated)
    assert bool(out["has_valid_predictions"][0, 0])
    assert not bool(out["has_valid_predictions"][0, 3])


@pytest.mark.parametrize("cls", SMOOTHERS)
def test_invalid_parent_does_not_suppress_child(cls, empty_disjoint_file):
    graph = make_graph(HIERARCHY)
    smoother = cls(graph, LABELS, empty_disjoint_file, **EXTRA_ARGS.get(cls, {}))
    scores = torch.tensor([[0.05, 0.5, 0.73, 0.5, 0.5]])
    valid = torch.tensor([[False, True, True, True, True]])

    out = smoother(scores.clone(), valid)
    assert out[0, 2].item() > NEUTRAL


def attribution_of(n_labels, assignment):
    a = torch.zeros((1, n_labels, 2))
    for label, weights in assignment.items():
        a[0, label] = torch.tensor(weights)
    return a


def test_attribution_rows_stay_normalised(disjoint_file):
    graph = make_graph(HIERARCHY)
    smoother = ScoreBasedPredictionSmoother(graph, LABELS, disjoint_file)
    scores = torch.tensor([[0.25, 0.7, 0.9, 0.4, 0.62], [0.85, 0.78, 0.45, 0.58, 0.4]])
    attribution = torch.rand((2, len(LABELS), 3))
    attribution = attribution / attribution.sum(dim=2, keepdim=True)

    _, out = smoother(scores.clone(), attribution=attribution)
    assert torch.allclose(out.sum(dim=2), torch.ones(2, len(LABELS)), atol=1e-5)


def test_attribution_passes_up_the_hierarchy(empty_disjoint_file):
    graph = make_graph(HIERARCHY)
    smoother = ScoreBasedPredictionSmoother(graph, LABELS, empty_disjoint_file)
    # "3" is a subclass of "1" and drags "1" up to its own score, so it also hands over its
    # attribution; "2" is untouched and keeps its own
    scores = torch.tensor([[0.5, 0.8, 0.95, 0.5, 0.5]])
    attribution = attribution_of(
        len(LABELS), {0: [0.5, 0.5], 1: [0.9, 0.1], 2: [1.0, 0.0]}
    )

    resolved, out = smoother(scores.clone(), attribution=attribution)
    assert resolved[0, 0].item() == pytest.approx(0.95)
    assert out[0, 0].tolist() == [1.0, 0.0]
    assert out[0, 1].tolist() == pytest.approx([0.9, 0.1], abs=1e-6)


def test_attribution_follows_disjointness_winner(disjoint_file):
    graph = make_graph([])
    smoother = ScoreBasedPredictionSmoother(graph, LABELS, disjoint_file)
    # "1" and "2" are disjoint, "1" wins and pushes "2" down, so "2" inherits "1"'s attribution
    scores = torch.tensor([[0.9, 0.7, 0.5, 0.5, 0.5]])
    attribution = attribution_of(len(LABELS), {0: [1.0, 0.0], 1: [0.0, 1.0]})

    resolved, out = smoother(scores.clone(), attribution=attribution)
    assert resolved[0, 1].item() <= NEUTRAL
    assert out[0, 0].tolist() == [1.0, 0.0]
    assert out[0, 1].tolist() == [1.0, 0.0]


def test_attribution_survives_batching(disjoint_file):
    graph = make_graph(HIERARCHY)
    smoother = ScoreBasedPredictionSmoother(graph, LABELS, disjoint_file)
    scores = torch.rand((7, len(LABELS)))
    attribution = torch.rand((7, len(LABELS), 4))
    attribution = attribution / attribution.sum(dim=2, keepdim=True)
    aggregated = {
        "net_score": scores.clone(),
        "has_valid_predictions": torch.ones_like(scores, dtype=torch.bool),
        "attribution": attribution,
    }

    out = apply_inconsistency_resolution(smoother, LABELS, aggregated, batch_size=3)
    expected, expected_attribution = smoother(scores.clone(), attribution=attribution)
    assert torch.allclose(out["net_score"], expected)
    assert torch.allclose(out["attribution"], expected_attribution)
