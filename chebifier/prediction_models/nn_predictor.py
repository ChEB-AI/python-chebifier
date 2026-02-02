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
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.batch_size = kwargs.get("batch_size", None)
        # If batch_size is not provided, it will be set to default batch size used during training in Predictor
        self.predictor: Predictor = Predictor(ckpt_path, self.batch_size)

    @modelwise_smiles_lru_cache.batch_decorator
    def predict_smiles_list(self, smiles_list: list[str]) -> list:
        """
        Returns a list with the length of smiles_list, each element is
        either None (=failure) or a dictionary of classes and predicted values.
        """
        raw_preds: Tensor = self.predictor.predict_smiles(smiles_list)
        if raw_preds is not None:
            preds = [
                (
                    {
                        label: pred
                        for label, pred in zip(
                            self.predictor._classification_labels, raw_preds[i].tolist()
                        )
                    }
                )
                for i in range(len(smiles_list))
            ]
            return preds
        else:
            return [None for _ in smiles_list]

    def calculate_results(self, batch):
        collator = self.predictor._dm.reader.COLLATOR()
        dat = self.predictor._model._process_batch(
            collator(batch).to(self.predictor.device), 0
        )
        return self.predictor._model(dat, **dat["model_kwargs"])
