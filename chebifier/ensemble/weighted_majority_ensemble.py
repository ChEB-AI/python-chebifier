from pathlib import Path

import torch

from chebifier.ensemble.voting_ensemble import VotingEnsemble


class WMVwithF1Ensemble(VotingEnsemble):

    def __init__(
        self,
        ensemble_dir: str,
        use_confidence: bool = True,
        weighting_strength=1,
        weighting_exponent=1,
        **kwargs,
    ):
        """WMV ensemble that weights models based on their class-wise F1 scores. For each class, the weight is calculated as:
        weight = model_weight * (weighting_strength * F1 + (1 - weighting_strength)) ** weighting_exponent
        where F1 is the class-specific F1 score ("trust") of the model on the validation set.
        """
        super().__init__(ensemble_dir, use_confidence, **kwargs)
        self.weighting_strength = weighting_strength
        self.weighting_exponent = weighting_exponent

    def calibrate(self, validation_predictions, validation_data, validation_labels):
        super().calibrate(validation_predictions, validation_data, validation_labels)
        self._save_classwise_f1(validation_predictions, validation_labels)

    def _save_classwise_f1(self, validation_predictions, validation_labels):
        thresholds = self._load_prediction_thresholds()
        for model_name, predictions in validation_predictions.items():
            f1 = self.classwise_f1(
                predictions > thresholds[model_name], validation_labels
            )
            f1_path = Path(self.ensemble_dir) / f"{model_name}_classwise_f1.txt"
            with open(f1_path, "w+") as f:
                f.write("\n".join(f1.tolist()))
            print(
                f"Saved class-wise F1 scores to {f1_path}: {len(f1.tolist())} classes (macro-f1: {f1.mean().item():.4f})."
            )

    def _load_classwise_f1(self, model_name: str) -> torch.Tensor:
        classwise_f1_path = Path(self.ensemble_dir) / f"{model_name}_classwise_f1.txt"
        if classwise_f1_path.exists():
            with open(classwise_f1_path, "r", encoding="utf-8") as f:
                return torch.tensor([float(x) for x in f.read().splitlines()])
        else:
            raise FileNotFoundError(
                f"Class-wise F1 scores file not found for model {model_name} in ensemble directory: {self.ensemble_dir}. Please calibrate the ensemble first."
            )

    def calculate_trust(self, predictions: dict[str, torch.Tensor]) -> torch.Tensor:
        # Calculate trust based on class-wise F1 scores for each model
        # target shape: (num_molecules, num_classes, num_models)
        num_models = len(predictions)
        num_molecules = list(predictions.values())[0].shape[0]
        num_classes = list(predictions.values())[0].shape[1]
        trust_tensor = torch.ones(
            (num_molecules, num_classes, num_models), dtype=torch.float32
        )
        for model_idx, (model_name, prediction_tensor) in enumerate(
            predictions.items()
        ):
            classwise_f1 = self._load_classwise_f1(model_name)
            assert (
                classwise_f1.shape[0] == num_classes
            ), f"Class-wise F1 scores for model {model_name} do not match number of classes in predictions."
            # Expand classwise_f1 to match the shape of trust_tensor for broadcasting
            classwise_f1 = classwise_f1.unsqueeze(0).expand(num_molecules, -1)
            trust_tensor[:, :, model_idx] = (
                self.weighting_strength * classwise_f1 + (1 - self.weighting_strength)
            ) ** self.weighting_exponent
        return trust_tensor
