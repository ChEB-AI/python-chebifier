import os

import torch
from rdkit import Chem


class BaseEnsemble:
    """Base class for ensemble models.
    Each ensemble has to perform the following tasks:
    1. Calibration (e.g. calculating weights for WMV or fitting a meta-model) on validation data
    2. Prediction on test data (i.e., turning base learner predictions into aggregated predictions)

    Each ensemble gets a directory where it can store its calibration results (e.g. weights for WMV or meta-model parameters).

    Not part of the ensemble are
    - getting predictions from base learners
    - resolving inconsistencies in the aggregated predictions
    """

    def __init__(self, ensemble_dir: str):
        os.makedirs(ensemble_dir, exist_ok=True)
        self.ensemble_dir = ensemble_dir

    @property
    def ensemble_name(self):
        return self.__class__.__name__

    @property
    def decision_threshold(self):
        """Probability a class has to exceed to be predicted.

        Ensembles report probabilities, so the default is the neutral point: predict a class when
        the ensemble believes it more likely than not. Ensembles that tune their own operating
        point (to maximise macro-F1, say) override this.
        """
        return 0.5

    def calibrate(
        self,
        validation_predictions: dict[str, torch.Tensor],
        validation_data: list[Chem.Mol],
        validation_labels: torch.Tensor,
    ):
        """Calibrate the ensemble model using validation predictions and labels.
        At the end, save the calibration results (e.g. weights for WMV or meta-model parameters) to self.ensemble_dir.
        """
        pass

    def predict(
        self,
        test_predictions: dict[str, torch.Tensor],
        molecules: list[Chem.Mol] | None = None,
    ):
        """Aggregate base learner predictions. `molecules` is only required by ensembles that
        depend on the molecules themselves (e.g. dynamic selection, which looks up a region of
        competence for each of them)."""
        raise NotImplementedError()
