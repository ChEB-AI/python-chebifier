import os

import torch

from chebifier.predict import (
    base_learner_cache_path,
    collect_base_learner_predictions,
    load_dense_predictions,
    save_dense_predictions,
)


class EnsembleBuilder:
    """
    A class to build an ensemble model from base learners and validation data.

    Attributes:
        base_learners (dict[str, BasePredictor]): A dictionary of base learner models.
        ensemble_model (BaseEnsemble): An instance of a BaseEnsemble model.
        validation_data (list[Chem.Mol]): Validation data for calibration.
        validation_labels (pd.DataFrame): Validation labels for calibration, one column per class.
            The column names define the label set the base learner predictions are mapped onto.
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
        os.makedirs(self.prediction_cache_dir, exist_ok=True)

    def build_ensemble(self):
        """
        Build an ensemble model from base learners and validation data.

        Base learner predictions are cached to avoid recomputation.
        """

        # Step 1: Get predictions from base learners on validation data
        validation_predictions = {}
        classes = {}
        # get cached predictions if available, otherwise compute and cache them
        for model_name, model in self.base_learners.items():
            cache_path = base_learner_cache_path(
                self.prediction_cache_dir, model_name, "validation"
            )
            if os.path.exists(cache_path):
                print(f"{model_name} validation predictions found in cache, loading...")
                validation_predictions[model_name] = load_dense_predictions(cache_path)
            else:
                print(f"Computing {model_name} validation predictions...")
                validation_predictions[model_name] = model.predict_dense(
                    self.validation_data
                )
                save_dense_predictions(cache_path, *validation_predictions[model_name])

        # Base learners may be trained on different label sets (e.g. ChEBI25 vs. ChEBI25_3_STAR),
        # so their union does not match the labels we calibrate against. Map every base learner
        # onto the label set of the validation data instead.
        label_classes = [str(cls) for cls in self.validation_labels.columns]
        validation_predictions, classes = collect_base_learner_predictions(
            validation_predictions, classes=label_classes
        )
        validation_labels = torch.from_numpy(
            self.validation_labels.to_numpy(dtype=bool)
        )

        print(
            f"Collected validation predictions from {len(validation_predictions)} base learners with {len(classes)} unique classes. Calibrating ensemble model..."
        )
        # Step 2: Calibrate the ensemble model using validation predictions
        self.ensemble_model.calibrate(
            validation_predictions, self.validation_data, validation_labels
        )

        return self.ensemble_model
