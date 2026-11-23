import numpy as np
import pytest
import torch

from chebifier.ensemble.dynamic_selection_ensemble import MetaChunk, aggregate
from chebifier.ensemble.level1 import (
    dense_from_pairs,
    fit_platt,
    noncandidate_probability,
)
from chebifier.ensemble.voting_ensemble import (
    MajorityVotingEnsemble,
    WMVwithConfidenceEnsemble,
)

THRESHOLD = 0.5


def make_ensemble(tmp_path, model_names, ensemble_class=WMVwithConfidenceEnsemble):
    ensemble = ensemble_class(str(tmp_path / "ensemble"))
    ensemble.prediction_thresholds = {name: THRESHOLD for name in model_names}
    return ensemble


def net_score(ensemble, columns):
    predictions = {
        name: torch.tensor(column, dtype=torch.float32).reshape(1, -1)
        for name, column in columns.items()
    }
    return ensemble.predict(predictions)["net_score"]


def test_net_score_is_a_probability_agreeing_with_the_vote(tmp_path):
    generator = torch.Generator().manual_seed(0)
    names = [f"m{i}" for i in range(4)]
    ensemble = make_ensemble(tmp_path, names)
    predictions = {name: torch.rand((32, 16), generator=generator) for name in names}
    aggregated = ensemble.predict(predictions)

    net = aggregated["net_score"]
    assert bool(((net >= 0.0) & (net <= 1.0)).all())
    vote = aggregated["positive_sum"] - aggregated["negative_sum"]
    assert torch.equal(net > THRESHOLD, vote > 0)


def test_net_score_is_invariant_to_the_number_of_covering_models(tmp_path):
    """A class every model agrees on scores the same whether one model or three cover it."""
    names = ["wide_a", "wide_b", "narrow"]
    ensemble = make_ensemble(tmp_path, names)
    nan = float("nan")
    # class 0 is covered by all three models, class 1 only by `narrow` - both unanimously positive
    scores = net_score(
        ensemble,
        {"wide_a": [0.9, nan], "wide_b": [0.9, nan], "narrow": [0.9, 0.9]},
    )

    assert scores[0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert scores[0, 1].item() == pytest.approx(scores[0, 0].item(), abs=1e-6)


def test_net_score_is_the_weighted_agreement(tmp_path):
    names = [f"m{i}" for i in range(4)]
    ensemble = make_ensemble(tmp_path, names, ensemble_class=MajorityVotingEnsemble)
    # three models positive, one negative -> 3/4 of the vote mass says yes
    scores = net_score(ensemble, {"m0": [0.9], "m1": [0.9], "m2": [0.9], "m3": [0.1]})

    assert scores[0, 0].item() == pytest.approx(0.75, abs=1e-6)


def test_models_without_a_prediction_do_not_contribute(tmp_path):
    names = ["covering", "absent"]
    ensemble = make_ensemble(tmp_path, names)
    nan = float("nan")
    scores = net_score(ensemble, {"covering": [0.8], "absent": [nan]})

    alone = make_ensemble(tmp_path, ["covering"])
    assert scores[0, 0].item() == pytest.approx(
        net_score(alone, {"covering": [0.8]})[0, 0].item(), abs=1e-6
    )


def test_unanimous_abstention_scores_neutral(tmp_path):
    """Every model sitting exactly on its threshold must not divide by zero."""
    ensemble = make_ensemble(tmp_path, ["m0", "m1"])
    scores = net_score(ensemble, {"m0": [THRESHOLD], "m1": [THRESHOLD]})

    assert scores[0, 0].item() == pytest.approx(0.5, abs=1e-6)


def test_unanimous_rejection_scores_zero(tmp_path):
    ensemble = make_ensemble(tmp_path, ["m0", "m1"])
    scores = net_score(ensemble, {"m0": [0.0], "m1": [0.0]})

    assert scores[0, 0].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("vote", ["plain", "confidence"])
def test_dynamic_selection_aggregate_is_bounded(vote):
    generator = np.random.default_rng(0)
    n_pairs, n_models = 64, 5
    delta = generator.random((n_pairs, n_models), dtype=np.float32)
    pool = generator.random((n_pairs, n_models)) > 0.3
    pool[~pool.any(axis=1), 0] = True
    # molecules without a usable region of competence drop out of selection but stay in the pool
    voting = pool & (generator.random((n_pairs, 1)) > 0.2)
    profiles = generator.random((n_pairs, n_models)).astype(np.float32)
    consensus = generator.uniform(0.5, 1.0, n_pairs)
    chunk = MetaChunk(None, voting, pool, profiles, consensus, None, None)

    net = aggregate(
        delta, chunk, competence_threshold=0.5, consensus_threshold=0.7, vote=vote
    )

    assert net.shape == (n_pairs,)
    assert np.all((net >= 0.0) & (net <= 1.0))


def test_dynamic_selection_agreed_pairs_survive_an_unusable_region():
    """A molecule RDKit cannot fingerprint has no region of competence, so no model can be
    selected - but a pair the whole pool agrees on never consults the meta-classifier anyway.
    """
    profiles = np.full((1, 3), 0.9, dtype=np.float32)
    delta = np.full((1, 3), 0.1, dtype=np.float32)
    pool = np.ones((1, 3), dtype=bool)
    no_region = np.zeros((1, 3), dtype=bool)
    chunk = MetaChunk(None, no_region, pool, profiles, np.array([1.0]), None, None)

    net = aggregate(
        delta, chunk, competence_threshold=0.5, consensus_threshold=0.7, vote="plain"
    )

    assert net[0] == pytest.approx(1.0)


def test_noncandidate_probability_matches_the_measured_validation_rate():
    # 723 of 418,197 validation positives fell outside the candidate set at candidate_k=70
    assert noncandidate_probability(36_244_081, 723) == pytest.approx(
        1.995e-05, rel=1e-3
    )


def test_noncandidate_probability_stays_non_zero_without_missed_positives():
    assert noncandidate_probability(1_000_000, 0) > 0.0
    assert noncandidate_probability(1_000_000, 0) < noncandidate_probability(
        1_000_000, 10
    )


def test_dense_from_pairs_honours_an_explicit_floor():
    molecule = np.array([0, 1])
    class_index = np.array([0, 2])
    net = np.array([0.9, 0.2], dtype=np.float32)

    dense = dense_from_pairs(molecule, class_index, net, (2, 3), floor=2e-05)
    assert dense[0, 1] == pytest.approx(2e-05)
    assert dense[0, 0] == pytest.approx(0.9)

    auto = dense_from_pairs(molecule, class_index, net, (2, 3))
    assert auto[0, 1] == pytest.approx(-1.0)


def test_fit_platt_recovers_a_known_calibration():
    generator = np.random.default_rng(0)
    scores = generator.normal(size=200_000)
    probabilities = 1.0 / (1.0 + np.exp(-(2.0 * scores - 1.0)))
    labels = generator.random(scores.shape) < probabilities

    slope, intercept = fit_platt(scores, labels)

    assert slope == pytest.approx(2.0, abs=0.05)
    assert intercept == pytest.approx(-1.0, abs=0.05)


def test_fit_platt_rejects_anti_correlated_scores():
    generator = np.random.default_rng(0)
    scores = generator.normal(size=5_000)
    labels = scores < 0

    with pytest.raises(ValueError, match="non-positive slope"):
        fit_platt(scores, labels)


def test_attribution_is_the_share_of_the_net_score_a_model_controls(tmp_path):
    generator = torch.Generator().manual_seed(1)
    names = [f"m{i}" for i in range(4)]
    ensemble = make_ensemble(tmp_path, names)
    predictions = {name: torch.rand((8, 6), generator=generator) for name in names}
    aggregated = ensemble.predict(predictions, attribution=True)

    attribution = aggregated["attribution"]
    assert aggregated["attribution_models"] == names
    assert torch.allclose(attribution.sum(dim=2), torch.ones(8, 6), atol=1e-5)
    # mirroring a model's score across the threshold flips its vote at unchanged confidence, so
    # the net score moves by exactly the share the attribution assigns to it
    flipped = dict(predictions, **{names[0]: 2 * THRESHOLD - predictions[names[0]]})
    moved = ensemble.predict(flipped)["net_score"] - aggregated["net_score"]
    assert torch.allclose(moved.abs(), attribution[:, :, 0], atol=1e-5)


def test_attribution_ignores_models_that_do_not_cover_the_class(tmp_path):
    names = ["wide", "narrow"]
    ensemble = make_ensemble(tmp_path, names)
    nan = float("nan")
    aggregated = ensemble.predict(
        {
            "wide": torch.tensor([[0.9, 0.9]]),
            "narrow": torch.tensor([[0.9, nan]]),
        },
        attribution=True,
    )

    assert aggregated["attribution"][0, 0].tolist() == pytest.approx([0.5, 0.5])
    assert aggregated["attribution"][0, 1].tolist() == pytest.approx([1.0, 0.0])
