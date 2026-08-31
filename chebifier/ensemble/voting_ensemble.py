from pathlib import Path

import torch
import yaml
from torchmetrics import F1Score

from chebifier.ensemble.base_ensemble import BaseEnsemble
from chebifier.inconsistency_resolution import NEUTRAL, confidence


class VotingEnsemble(BaseEnsemble):
    """Base class for the voting ensembles. Subclasses define the vote weights by overriding
    calculate_confidence (self-reported confidence of a model) and calculate_trust (measured
    reliability of a model)."""

    def __init__(
        self,
        ensemble_dir: str,
    ):
        super().__init__(ensemble_dir)
        self.classwise_f1 = None
        self.prediction_thresholds = None

    def find_best_threshold(self, predictions, val_labels_tensor):
        best_threshold = 0.5
        best_f1 = 0.0
        for threshold in range(1, 100):
            threshold_value = threshold / 100
            macro_f1_score = self.classwise_f1(
                predictions > threshold_value, val_labels_tensor
            ).mean()
            if macro_f1_score > best_f1:
                best_f1 = macro_f1_score.item()
                best_threshold = threshold_value
        return best_threshold

    def calibrate(self, validation_predictions, validation_data, validation_labels):
        print(
            f"Calibrating {self.ensemble_name} with {len(validation_predictions)} base learners..."
        )
        self.classwise_f1 = F1Score(
            task="multilabel", num_labels=validation_labels.shape[1], average=None
        )
        self._save_prediction_thresholds(
            self._fit_prediction_thresholds(validation_predictions, validation_labels)
        )

    def _fit_prediction_thresholds(self, predictions, labels) -> dict[str, float]:
        return {
            model_name: self.find_best_threshold(model_predictions, labels)
            for model_name, model_predictions in predictions.items()
        }

    def _save_prediction_thresholds(self, thresholds: dict[str, float]):
        thresholds_path = Path(self.ensemble_dir) / "prediction_thresholds.yaml"
        with open(thresholds_path, "w+", encoding="utf-8") as f:
            yaml.dump(thresholds, f)
        print(f"Saved prediction thresholds to {thresholds_path}: {thresholds}")

    def _load_prediction_thresholds(self) -> dict[str, float]:
        if self.prediction_thresholds is not None:
            return self.prediction_thresholds
        thresholds_path = Path(self.ensemble_dir) / "prediction_thresholds.yaml"
        if thresholds_path.exists():
            with open(thresholds_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(
                f"Prediction thresholds file not found in ensemble directory: {self.ensemble_dir}. Please calibrate the ensemble first."
            )

    def calculate_confidence(
        self, predictions_tensor: torch.Tensor, thresholds: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def calculate_trust(self, predictions: dict[str, torch.Tensor]) -> torch.Tensor:
        # No trust unless a subclass measures it (e.g. WMVwithF1Ensemble)
        return 1

    def predict(
        self,
        test_predictions: dict[str, torch.Tensor],
        molecules=None,
        attribution: bool = False,
    ):
        """
        Aggregates predictions from multiple models by voting, each vote weighted by
        calculate_confidence * calculate_trust.

        With `attribution`, the share each base learner holds of the weight mass cast for a
        (molecule, class) pair is reported as well. That share is exactly the fraction of the net
        score the model controls: the score is a weight-normalised average of the votes, so moving
        one model's vote from reject to assert moves the score by its share.

        The net score is the weighted agreement among the models that voted, mapped onto [0, 1]:
        1 if they unanimously predict the class, 0 if they unanimously reject it, 0.5 if they are
        evenly split. Normalising by the weight mass that was cast (rather than summing over models)
        keeps classes covered by different numbers of base learners comparable, which is what
        inconsistency resolution needs - it compares scores across classes. The agreement fraction
        is a well calibrated probability on its own, so no further calibration is applied.
        """
        predictions_tensor = torch.stack(
            list(test_predictions.values()), dim=2
        )  # Shape: (num_molecules, num_classes, num_models)
        # Get predictions for all classes
        valid_predictions = ~torch.isnan(predictions_tensor)
        valid_counts = valid_predictions.sum(dim=2)  # Sum over models dimension

        thresholds = self._load_prediction_thresholds()
        threshold_mask = torch.tensor(
            [
                thresholds.get(model_name, NEUTRAL)
                for model_name in test_predictions.keys()
            ],
            dtype=predictions_tensor.dtype,
            device=predictions_tensor.device,
        )

        # Skip classes with no valid predictions
        has_valid_predictions = valid_counts > 0

        # Calculate positive and negative predictions for all classes at once
        positive_mask = (
            predictions_tensor > threshold_mask.unsqueeze(0).unsqueeze(0)
        ) & valid_predictions
        negative_mask = (
            predictions_tensor < threshold_mask.unsqueeze(0).unsqueeze(0)
        ) & valid_predictions

        confidence = self.calculate_confidence(
            predictions_tensor, threshold_mask.unsqueeze(0).unsqueeze(0)
        )

        trust = self.calculate_trust(test_predictions)
        # Calculate weighted predictions using broadcasting
        # predictions shape: (num_molecules, num_classes, num_models)
        # weights shape: (num_classes, num_models)
        positive_weighted = positive_mask.float() * confidence * trust
        negative_weighted = negative_mask.float() * confidence * trust

        # Sum over models dimension
        positive_sum = positive_weighted.sum(
            dim=2
        )  # Shape: (num_molecules, num_classes)
        negative_sum = negative_weighted.sum(
            dim=2
        )  # Shape: (num_molecules, num_classes)

        # Determine which classes to include for each molecule
        net_score = (
            0.5
            + (positive_sum - negative_sum)
            / (positive_sum + negative_sum).clamp(min=1e-6)
            / 2
        ).clamp(
            0.0, 1.0
        )  # Shape: (num_molecules, num_classes)
        result = {
            "net_score": net_score,
            "has_valid_predictions": has_valid_predictions,
            "positive_sum": positive_sum,
            "negative_sum": negative_sum,
            "confidence": confidence,
            "positive_mask": positive_mask,
            "negative_mask": negative_mask,
        }
        if attribution:
            weights = positive_weighted + negative_weighted
            result["attribution"] = weights / weights.sum(dim=2, keepdim=True).clamp(
                min=1e-12
            )
            result["attribution_models"] = list(test_predictions.keys())
        return result


class MajorityVotingEnsemble(VotingEnsemble):
    """Plain majority voting: every model that votes counts the same, no weights at all."""

    def calculate_confidence(
        self, predictions_tensor: torch.Tensor, thresholds: torch.Tensor
    ) -> torch.Tensor:
        return torch.ones_like(predictions_tensor)


class WMVwithConfidenceEnsemble(VotingEnsemble):
    """WMV ensemble that weights each vote by the model's self-reported confidence, i.e. how far
    its prediction sits from its decision threshold, scaled separately on each side so that a
    maximally confident negative and a maximally confident positive both count 1."""

    def calculate_confidence(
        self, predictions_tensor: torch.Tensor, thresholds: torch.Tensor
    ) -> torch.Tensor:
        return confidence(predictions_tensor.nan_to_num(), thresholds)
