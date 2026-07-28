import os

import torch


class EnsembleBuilder:
    """
    A class to build an ensemble model from base learners and validation data.

    Attributes:
        base_learners (dict[str, BasePredictor]): A dictionary of base learner models.
        ensemble_model (BaseEnsemble): An instance of a BaseEnsemble model.
        validation_data (list[Chem.Mol]): Validation data for calibration.
        validation_labels (torch.Tensor): Validation labels for calibration.
        prediction_cache_dir (str): Directory to cache predictions.
    """

    def __init__(
        self,
        base_learners,
        ensemble_model,
        validation_data,
        validation_labels,
        prediction_cache_dir,
    ):
        self.base_learners = base_learners
        self.ensemble_model = ensemble_model
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.prediction_cache_dir = prediction_cache_dir

    def build_ensemble(self):
        """
        Build an ensemble model from base learners and validation data.

        Base learner predictions are cached to avoid recomputation.
        """

        # Step 1: Get predictions from base learners on validation data
        validation_predictions = {}
        # get cached predictions if available, otherwise compute and cache them
        for model_name, model in self.base_learners.items():
            cache_path = os.path.join(
                self.prediction_cache_dir, f"{model_name}_validation_predictions.pt"
            )
            if os.path.exists(cache_path):
                validation_predictions[model_name] = torch.load(
                    cache_path, weights_only=False
                )
            else:
                validation_predictions[model_name] = model.predict_list(
                    self.validation_data
                )
                torch.save(validation_predictions[model_name], cache_path)

        # Step 2: Calibrate the ensemble model using validation predictions
        self.ensemble_model.calibrate(
            validation_predictions, self.validation_data, self.validation_labels
        )

        return self.ensemble_model
