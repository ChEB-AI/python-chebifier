import torch

from chebifier.ensemble.base_ensemble import BaseEnsemble


class WMVwithPPVNPVEnsemble(BaseEnsemble):

    def __init__(
        self, config_path=None, weighting_strength=0.5, weighting_exponent=1.0, **kwargs
    ):
        """WMV ensemble that weights models based on their class-wise positive / negative predictive values. For each class, the weight is calculated as:
        weight = (weighting_strength * PPV + (1 - weighting_strength)) ** weighting_exponent
        where PPV is the class-specific positive predictive value of the model on the validation set
        or (if the prediction is negative):
        weight = (weighting_strength * NPV + (1 - weighting_strength)) ** weighting_exponent
        where NPV is the class-specific negative predictive value of the model on the validation set.
        """
        super().__init__(config_path, **kwargs)
        self.weighting_strength = weighting_strength
        self.weighting_exponent = weighting_exponent

    def calculate_classwise_weights(self, predicted_classes):
        """
        Given the positions of predicted classes in the predictions tensor, assign weights to each class. The
        result is two tensors of shape (num_predicted_classes, num_models). The weight for each class is the model_weight
        (default: 1) multiplied by the class-specific positive / negative weight (default 1).
        """
        positive_weights = torch.ones(len(predicted_classes), len(self.models))
        negative_weights = torch.ones(len(predicted_classes), len(self.models))
        for j, model in enumerate(self.models):
            positive_weights[:, j] *= model.model_weight
            negative_weights[:, j] *= model.model_weight
            if model.classwise_weights is None:
                continue
            for cls, weights in model.classwise_weights.items():
                positive_weights[predicted_classes[cls], j] *= (
                    weights["PPV"] * self.weighting_strength
                    + (1 - self.weighting_strength)
                ) ** self.weighting_exponent
                negative_weights[predicted_classes[cls], j] *= (
                    weights["NPV"] * self.weighting_strength
                    + (1 - self.weighting_strength)
                ) ** self.weighting_exponent

        if self.verbose_output:
            print(
                "Calculated model weightings. The averages for positive / negative weights are:"
            )
            for i, model in enumerate(self.models):
                print(
                    f"{model.model_name}: {positive_weights[:, i].mean().item():.3f} / {negative_weights[:, i].mean().item():.3f}"
                )

        return positive_weights, negative_weights


class WMVwithF1Ensemble(BaseEnsemble):

    def __init__(
        self, config_path=None, weighting_strength=0.5, weighting_exponent=1.0, **kwargs
    ):
        """WMV ensemble that weights models based on their class-wise F1 scores. For each class, the weight is calculated as:
        weight = model_weight * (weighting_strength * F1 + (1 - weighting_strength)) ** weighting_exponent
        where F1 is the class-specific F1 score ("trust") of the model on the validation set.
        """
        super().__init__(config_path, **kwargs)
        self.weighting_strength = weighting_strength
        self.weighting_exponent = weighting_exponent

    def calculate_classwise_weights(self, predicted_classes):
        """
        Given the positions of predicted classes in the predictions tensor, assign weights to each class. The
        result is two tensors of shape (num_predicted_classes, num_models). The weight for each class is the model_weight
        (default: 1) multiplied by (1 + the class-specific validation-f1 (default 1)).
        """
        weights_by_cls = torch.ones(len(predicted_classes), len(self.models))
        for j, model in enumerate(self.models):
            weights_by_cls[:, j] *= model.model_weight
            if model.classwise_weights is None:
                continue
            for cls, weights in model.classwise_weights.items():
                if cls in predicted_classes:
                    if (2 * weights["TP"] + weights["FP"] + weights["FN"]) > 0:
                        f1 = (
                            2
                            * weights["TP"]
                            / (2 * weights["TP"] + weights["FP"] + weights["FN"])
                        )
                        weights_by_cls[predicted_classes[cls], j] *= (
                            self.weighting_strength * f1 + 1 - self.weighting_strength
                        ) ** self.weighting_exponent
        if self.verbose_output:
            print("Calculated model weightings. The average weights are:")
            for i, model in enumerate(self.models):
                print(f"{model.model_name}: {weights_by_cls[:, i].mean().item():.3f}")

        return weights_by_cls, weights_by_cls
