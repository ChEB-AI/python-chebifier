import unittest

import numpy as np
import torch

from chebifier import modelwise_smiles_lru_cache
from chebifier.predict import collect_base_learner_predictions
from chebifier.prediction_models.nn_predictor import NNPredictor

LABELS = ["CHEBI:1", "CHEBI:2", "CHEBI:3"]


class StubInnerPredictor:
    """
    Stands in for chebai.result.prediction.Predictor, which would otherwise need a
    checkpoint on disk. predict_molecules is typed list[torch.Tensor | None] - the
    None entries are molecules the reader could not parse or that exceeded the
    model's token limit.
    """

    def __init__(self, preds: list[torch.Tensor | None]):
        self._classification_labels = LABELS
        self._preds = preds

    def predict_molecules(self, molecules) -> list[torch.Tensor | None]:
        return self._preds


def make_predictor(model_name: str, preds: list[torch.Tensor | None]) -> NNPredictor:
    """
    Build an NNPredictor without running __init__, which would load a checkpoint.
    """
    predictor = NNPredictor.__new__(NNPredictor)
    predictor.model_name = model_name
    predictor.predictor = StubInnerPredictor(preds)
    return predictor


class TestNNPredictorNoneHandling(unittest.TestCase):
    def tearDown(self) -> None:
        # predict_list goes through the module-level cache, so keep tests independent
        modelwise_smiles_lru_cache.clear()

    def test_predict_dense_partial_failure(self) -> None:
        """
        A None for a single molecule must not raise - that row becomes NaN while the
        molecules the model did handle keep their scores.
        """
        predictor = make_predictor(
            "partial",
            [torch.tensor([0.1, 0.2, 0.3]), None, torch.tensor([0.7, 0.8, 0.9])],
        )

        classes, scores = predictor.predict_dense(["C", "not-a-smiles", "CC"])

        self.assertListEqual(classes, LABELS)
        self.assertEqual(scores.shape, (3, len(LABELS)))
        # the failed molecule is excluded from the ensemble via NaN
        self.assertTrue(np.all(np.isnan(scores[1])))
        # the other molecules are unaffected (SCORE_DTYPE is float16, hence the tolerance)
        self.assertFalse(np.any(np.isnan(scores[0])))
        self.assertFalse(np.any(np.isnan(scores[2])))
        np.testing.assert_allclose(scores[0], [0.1, 0.2, 0.3], atol=1e-3)
        np.testing.assert_allclose(scores[2], [0.7, 0.8, 0.9], atol=1e-3)

    def test_predict_dense_all_failures(self) -> None:
        """
        If no molecule could be processed, the shape is still correct and every cell
        is NaN, so the model contributes nothing to the vote.
        """
        predictor = make_predictor("all_failed", [None, None])

        classes, scores = predictor.predict_dense(["not-a-smiles", "also-not"])

        self.assertListEqual(classes, LABELS)
        self.assertEqual(scores.shape, (2, len(LABELS)))
        self.assertTrue(np.all(np.isnan(scores)))

    def test_predict_dense_empty_input(self) -> None:
        """
        An empty molecule list yields an empty matrix rather than raising.
        """
        predictor = make_predictor("empty", [])

        classes, scores = predictor.predict_dense([])

        self.assertListEqual(classes, LABELS)
        self.assertEqual(scores.shape, (0, len(LABELS)))

    def test_predict_list_partial_failure(self) -> None:
        """
        predict_list documents that each element is either None (=failure) or a dict
        of classes and predicted values.
        """
        predictor = make_predictor(
            "list_partial", [torch.tensor([0.1, 0.2, 0.3]), None]
        )

        preds = predictor.predict_list(["C", "not-a-smiles"])

        self.assertEqual(len(preds), 2)
        self.assertIsNone(preds[1])
        self.assertEqual(len(preds[0]), len(LABELS))
        self.assertListEqual(list(preds[0]), LABELS)

    def test_partial_failure_survives_ensemble_collection(self) -> None:
        """
        The reported crash surfaced in chebifier.predict, so check that the output of
        predict_dense actually flows through the ensemble's collection step: the NaN
        row must be preserved and the model must still contribute for the rest.
        """
        predictor = make_predictor(
            "collected",
            [torch.tensor([0.1, 0.2, 0.3]), None, torch.tensor([0.7, 0.8, 0.9])],
        )
        molecules = ["C", "not-a-smiles", "CC"]

        collected, ensemble_classes = collect_base_learner_predictions(
            {"collected": predictor.predict_dense(molecules)}
        )

        tensor = collected["collected"]
        self.assertListEqual(ensemble_classes, LABELS)
        self.assertEqual(tuple(tensor.shape), (3, len(LABELS)))
        # this is what VotingEnsemble turns into valid_predictions / has_valid_predictions
        valid = ~torch.isnan(tensor)
        self.assertFalse(bool(valid[1].any()))
        self.assertTrue(bool(valid[0].all()))
        self.assertTrue(bool(valid[2].all()))


if __name__ == "__main__":
    unittest.main()
