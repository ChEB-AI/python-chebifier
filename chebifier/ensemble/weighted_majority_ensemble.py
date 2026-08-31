from pathlib import Path

import pandas as pd
import torch

from chebifier.ensemble.voting_ensemble import WMVwithConfidenceEnsemble

N_FOLDS = 5
WEIGHTING_STRENGTH_GRID = [0, 0.25, 0.5, 0.75, 1]


class WMVwithF1Ensemble(WMVwithConfidenceEnsemble):

    def __init__(
        self,
        ensemble_dir: str,
        weighting_strength=None,
        weighting_exponent=None,
        model_weights=None,
        **kwargs,
    ):
        """WMV ensemble that weights models based on their class-wise F1 scores, on top of the
        confidence weighting of WMVwithConfidenceEnsemble. For each class, the weight is calculated as:
        weight = model_weight * (weighting_strength * F1 + (1 - weighting_strength)) ** weighting_exponent
        where F1 is the class-specific F1 score ("trust") of the model on the validation set.

        weighting_strength and weighting_exponent default to the optimal values determined during
        calibration (best_hyperparameters.csv in the ensemble directory), falling back to 1 if the
        ensemble has not been calibrated. Values passed here take precedence over both.

        model_weights is a per-model multiplier set manually in the configuration (see the
        model_weight config key), defaulting to 1 for models without an explicit entry.
        """
        super().__init__(ensemble_dir, **kwargs)
        self.weighting_strength = weighting_strength
        self.weighting_exponent = weighting_exponent
        self.model_weights = dict(model_weights or {})
        self.model_f1_scores = None

    def model_weight(self, model_name: str) -> float:
        return float(self.model_weights.get(model_name, 1))

    def calibrate(self, validation_predictions, validation_data, validation_labels):
        super().calibrate(validation_predictions, validation_data, validation_labels)
        self._save_classwise_f1(validation_predictions, validation_labels)
        self._optimize_hyperparameters(validation_predictions, validation_labels)

    def _fit_classwise_f1(self, predictions, labels, thresholds):
        return {
            model_name: self.classwise_f1(
                model_predictions > thresholds[model_name], labels
            )
            for model_name, model_predictions in predictions.items()
        }

    def _save_classwise_f1(self, validation_predictions, validation_labels):
        thresholds = self._load_prediction_thresholds()
        classwise_f1 = self._fit_classwise_f1(
            validation_predictions, validation_labels, thresholds
        )
        for model_name, f1 in classwise_f1.items():
            f1_path = Path(self.ensemble_dir) / f"{model_name}_classwise_f1.txt"
            with open(f1_path, "w+") as f:
                f.writelines(f"{x}\n" for x in f1.tolist())
            print(
                f"Saved class-wise F1 scores to {f1_path}: {len(f1.tolist())} classes (macro-f1: {f1.mean().item():.4f})."
            )

    def _load_classwise_f1(self, model_name: str) -> torch.Tensor:
        if self.model_f1_scores is not None:
            return self.model_f1_scores[model_name]
        classwise_f1_path = Path(self.ensemble_dir) / f"{model_name}_classwise_f1.txt"
        if classwise_f1_path.exists():
            with open(classwise_f1_path, "r", encoding="utf-8") as f:
                return torch.tensor([float(x) for x in f.read().splitlines()])
        else:
            raise FileNotFoundError(
                f"Class-wise F1 scores file not found for model {model_name} in ensemble directory: {self.ensemble_dir}. Please calibrate the ensemble first."
            )

    def _load_hyperparameters(self) -> tuple[float, int]:
        """Hyperparameters set explicitly take precedence, otherwise the optimal values found during
        calibration are used (falling back to 1 if the ensemble has not been calibrated).
        """
        best = {}
        best_path = Path(self.ensemble_dir) / "best_hyperparameters.csv"
        if best_path.exists():
            best = pd.read_csv(best_path).iloc[0].to_dict()
        strength = self.weighting_strength
        if strength is None:
            strength = float(best.get("weighting_strength", 1))
        exponent = self.weighting_exponent
        if exponent is None:
            exponent = int(best.get("weighting_exponent", 1))
        return strength, exponent

    def calculate_trust(self, predictions: dict[str, torch.Tensor]) -> torch.Tensor:
        # Calculate trust based on class-wise F1 scores for each model
        # target shape: (num_molecules, num_classes, num_models)
        weighting_strength, weighting_exponent = self._load_hyperparameters()
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
                weighting_strength * classwise_f1 + (1 - weighting_strength)
            ) ** weighting_exponent * self.model_weight(model_name)
        return trust_tensor

    def _build_folds(self, validation_predictions, validation_labels):
        """Split the validation set into N_FOLDS folds and calibrate prediction thresholds and
        class-wise F1 scores on the training part of each fold. Returns one
        (thresholds, class-wise F1 scores, held-out indices) tuple per fold."""
        permutation = torch.randperm(
            validation_labels.shape[0], generator=torch.Generator().manual_seed(0)
        )
        fold_indices = [permutation[i::N_FOLDS] for i in range(N_FOLDS)]
        folds = []
        for fold, test_idx in enumerate(fold_indices):
            print(f"Calibrating fold {fold + 1}/{N_FOLDS}...")
            train_idx = torch.cat(
                [idx for other, idx in enumerate(fold_indices) if other != fold]
            )
            train_predictions = {
                model_name: model_predictions[train_idx]
                for model_name, model_predictions in validation_predictions.items()
            }
            train_labels = validation_labels[train_idx]
            thresholds = self._fit_prediction_thresholds(
                train_predictions, train_labels
            )
            classwise_f1 = self._fit_classwise_f1(
                train_predictions, train_labels, thresholds
            )
            folds.append((thresholds, classwise_f1, test_idx))
        return folds

    def _score_hyperparameters(
        self,
        folds,
        validation_predictions,
        validation_labels,
        weighting_strength,
        weighting_exponent,
    ):
        """Macro F1 of the aggregated predictions on each held-out fold, averaged only over
        classes that have positive labels in that fold (a fold is too small for every class to
        be present, and absent classes would otherwise contribute a hard 0)."""
        self.weighting_strength = weighting_strength
        self.weighting_exponent = weighting_exponent
        scores = []
        for thresholds, classwise_f1, test_idx in folds:
            self.prediction_thresholds = thresholds
            self.model_f1_scores = classwise_f1
            aggregated = self.predict(
                {
                    model_name: model_predictions[test_idx]
                    for model_name, model_predictions in validation_predictions.items()
                }
            )
            # the net score is a probability, so the operating point is the one the ensemble
            # reports - scoring at > 0 asks "did any covering learner vote positive", which is
            # almost independent of the weights the search is supposed to compare
            decisions = (
                aggregated["net_score"] > self.decision_threshold
            ) & aggregated["has_valid_predictions"]
            fold_labels = validation_labels[test_idx]
            fold_f1 = self.classwise_f1(decisions, fold_labels)
            scores.append(fold_f1[fold_labels.sum(dim=0) > 0].mean().item())
        self.prediction_thresholds = None
        self.model_f1_scores = None
        return scores

    def _optimize_hyperparameters(self, validation_predictions, validation_labels):
        print(
            f"Optimizing hyperparameters with {N_FOLDS}-fold cross-validation on the validation set..."
        )
        weighting_strength = self.weighting_strength
        weighting_exponent = self.weighting_exponent
        folds = self._build_folds(validation_predictions, validation_labels)
        results = []

        def score(stage, strength, exponent):
            scores = self._score_hyperparameters(
                folds, validation_predictions, validation_labels, strength, exponent
            )
            mean_score = sum(scores) / len(scores)
            results.append(
                {
                    "stage": stage,
                    "weighting_strength": strength,
                    "weighting_exponent": exponent,
                    "mean_macro_f1": mean_score,
                    "std_macro_f1": torch.tensor(scores).std().item(),
                    **{f"fold_{i}_macro_f1": s for i, s in enumerate(scores)},
                }
            )
            print(
                f"weighting_strength={strength}, weighting_exponent={exponent}: macro-f1 {mean_score:.4f}"
            )
            return mean_score

        strength_scores = {
            strength: score("weighting_strength", strength, 1)
            for strength in WEIGHTING_STRENGTH_GRID
        }
        best_strength = max(strength_scores, key=strength_scores.get)

        best_exponent = 1
        best_score = strength_scores[best_strength]
        exponent = 2
        while True:
            exponent_score = score("weighting_exponent", best_strength, exponent)
            if exponent_score <= best_score:
                break
            best_score = exponent_score
            best_exponent = exponent
            exponent += 1

        self.weighting_strength = weighting_strength
        self.weighting_exponent = weighting_exponent
        self._save_hyperparameter_results(
            results, best_strength, best_exponent, best_score
        )

    def _save_hyperparameter_results(
        self, results, best_strength, best_exponent, best_score
    ):
        results_path = Path(self.ensemble_dir) / "hyperparameter_search.csv"
        pd.DataFrame(results).to_csv(results_path, index=False)
        best_path = Path(self.ensemble_dir) / "best_hyperparameters.csv"
        pd.DataFrame(
            [
                {
                    "weighting_strength": best_strength,
                    "weighting_exponent": best_exponent,
                    "mean_macro_f1": best_score,
                }
            ]
        ).to_csv(best_path, index=False)
        print(
            f"Saved hyperparameter search results to {results_path}. Recommended parameters (saved to {best_path}): "
            f"weighting_strength={best_strength}, weighting_exponent={best_exponent} (cross-validated macro-f1: {best_score:.4f})."
        )
