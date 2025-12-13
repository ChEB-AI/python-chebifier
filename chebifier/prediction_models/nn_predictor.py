from abc import ABC
from typing import TYPE_CHECKING

from chebai.result.prediction import Predictor

from chebifier import modelwise_smiles_lru_cache

from .base_predictor import BasePredictor

if TYPE_CHECKING:
    from torch import Tensor


class NNPredictor(BasePredictor, ABC):
    def __init__(
        self,
        model_name: str,
        ckpt_path: str,
        target_labels_path: str,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.batch_size = kwargs.get("batch_size", None)
        # If batch_size is not provided, it will be set to default batch size used during training in Predictor
        self._predictor: Predictor = Predictor(ckpt_path, self.batch_size)
        self.target_labels = [
            line.strip() for line in open(target_labels_path, encoding="utf-8")
        ]

        # Sanity check - ensure that the number of classes predicted by the model matches the number of target labels
        # TODO: In future, we can include the target labels in the model metadata and avoid this.
        raw_preds = self._predictor.predict_smiles(["CO"])
        assert len(raw_preds[0]) == len(
            self.target_labels
        ), "Number of predicted classes does not match number of target labels."

    @modelwise_smiles_lru_cache.batch_decorator
    def predict_smiles_list(self, smiles_list: list[str]) -> list:
        """
        Returns a list with the length of smiles_list, each element is
        either None (=failure) or a dictionary of classes and predicted values.
        """
        raw_preds: Tensor = self._predictor.predict_smiles(smiles_list)
        if raw_preds is not None:
            preds = [
                (
                    {
                        label: pred
                        for label, pred in zip(
                            self.target_labels, raw_preds[i].tolist()
                        )
                    }
                )
                for i in range(len(smiles_list))
            ]
            return preds
        else:
            return [None for _ in smiles_list]
