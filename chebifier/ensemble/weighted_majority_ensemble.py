import torch

from chebifier.ensemble.base_ensemble import BaseEnsemble



class WMVwithPPVNPVEnsemble(BaseEnsemble):

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
                positive_weights[predicted_classes[cls], j] *= weights["PPV"]
                negative_weights[predicted_classes[cls], j] *= weights["NPV"]

        print(f"Calculated model weightings. The averages for positive / negative weights are:")
        for i, model in enumerate(self.models):
            print(f"{model.model_name}: {positive_weights[:, i].mean().item():.3f} / {negative_weights[:, i].mean().item():.3f}")

        return positive_weights, negative_weights


class WMVwithF1Ensemble(BaseEnsemble):

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
                if (2 * weights["TP"] + weights["FP"] + weights["FN"]) > 0:
                    f1 = 2 * weights["TP"] / (2 * weights["TP"] + weights["FP"] + weights["FN"])
                    weights_by_cls[predicted_classes[cls], j] *= 1 + f1

        print(f"Calculated model weightings. The average weights are:")
        for i, model in enumerate(self.models):
            print(f"{model.model_name}: {weights_by_cls[:, i].mean().item():.3f}")

        return weights_by_cls, weights_by_cls