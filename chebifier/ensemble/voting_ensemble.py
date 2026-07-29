from pathlib import Path

import torch
import yaml
from torchmetrics import F1Score

from chebifier.ensemble.base_ensemble import BaseEnsemble


class VotingEnsemble(BaseEnsemble):
    def __init__(
        self,
        ensemble_dir: str,
        use_confidence: bool = True,
    ):
        super().__init__(ensemble_dir)
        self.use_confidence = use_confidence
        self.macro_f1 = None

    def find_best_threshold(self, predictions, val_labels_tensor):

        best_threshold = 0.5
        best_f1 = 0.0
        for threshold in range(0, 100):
            threshold_value = threshold / 100
            macro_f1_score = self.macro_f1(
                predictions > threshold_value, val_labels_tensor
            )
            if macro_f1_score > best_f1:
                best_f1 = macro_f1_score.item()
                best_threshold = threshold_value
        return best_threshold

    def calibrate(self, validation_predictions, validation_data, validation_labels):
        self.macro_f1 = F1Score(
            task="multilabel", num_labels=len(validation_labels), average="macro"
        )

        prediction_thresholds = {
            model_name: self.find_best_threshold(predictions, validation_labels)
            for model_name, predictions in validation_predictions.items()
        }
        self._save_prediction_thresholds(prediction_thresholds)

    def _save_prediction_thresholds(self, thresholds: dict[str, float]):
        thresholds_path = Path(self.ensemble_dir) / "prediction_thresholds.yaml"
        with open(thresholds_path, "w+", encoding="utf-8") as f:
            yaml.dump(thresholds, f)

    def _load_prediction_thresholds(self) -> dict[str, float]:
        thresholds_path = Path(self.ensemble_dir) / "prediction_thresholds.yaml"
        if thresholds_path.exists():
            with open(thresholds_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(
                f"Prediction thresholds file not found in ensemble directory: {self.ensemble_dir}. Please calibrate the ensemble first."
            )

    def predict(self, test_predictions: dict[str, torch.Tensor]):
        """
        Aggregates predictions from multiple models using weighted majority voting.
        weights are only the self-reported confidence (=difference between prediction and threshold). If set to false, all models are weighted equally.
        """
        predictions_tensor = torch.stack(
            list(test_predictions.values()), dim=2
        )  # Shape: (num_molecules, num_classes, num_models)
        # Get predictions for all classes
        valid_predictions = ~torch.isnan(predictions_tensor)
        valid_counts = valid_predictions.sum(dim=2)  # Sum over models dimension

        thresholds = self._load_prediction_thresholds()
        if any(model_name not in thresholds for model_name in test_predictions.keys()):
            raise ValueError(
                "Prediction thresholds not found for all models. Please calibrate the ensemble first. Models missing thresholds: "
                + ", ".join(
                    model_name
                    for model_name in test_predictions.keys()
                    if model_name not in thresholds
                )
            )
        threshold_mask = torch.tensor(
            [thresholds[model_name] for model_name in test_predictions.keys()],
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

        if self.use_confidence:
            confidence = 2 * torch.abs(
                predictions_tensor.nan_to_num()
                - threshold_mask.unsqueeze(0).unsqueeze(0)
            )
        else:
            confidence = torch.ones_like(predictions_tensor)

        # Calculate weighted predictions using broadcasting
        # predictions shape: (num_molecules, num_classes, num_models)
        # weights shape: (num_classes, num_models)
        positive_weighted = positive_mask.float() * confidence
        negative_weighted = negative_mask.float() * confidence

        # Sum over models dimension
        positive_sum = positive_weighted.sum(
            dim=2
        )  # Shape: (num_molecules, num_classes)
        negative_sum = negative_weighted.sum(
            dim=2
        )  # Shape: (num_molecules, num_classes)

        # Determine which classes to include for each molecule
        net_score = positive_sum - negative_sum  # Shape: (num_molecules, num_classes)
        return {
            "net_score": net_score,
            "has_valid_predictions": has_valid_predictions,
            "positive_sum": positive_sum,
            "negative_sum": negative_sum,
            "confidence": confidence,
            "positive_mask": positive_mask,
            "negative_mask": negative_mask,
        }
