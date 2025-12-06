from abc import ABC

from chebai.result.prediction import Predictor

from chebifier import modelwise_smiles_lru_cache

from .base_predictor import BasePredictor


class NNPredictor(BasePredictor, ABC):
    def __init__(
        self,
        model_name: str,
        ckpt_path: str,
        **kwargs,
    ):
        self.batch_size = kwargs.get("batch_size", None)
        # If batch_size is not provided, it will be set to default batch size used during training in Predictor
        self._predictor: Predictor = Predictor(ckpt_path, self.batch_size)

        super().__init__(model_name, **kwargs)

    @modelwise_smiles_lru_cache.batch_decorator
    def predict_smiles_list(self, smiles_list: list[str]) -> list:
        """
        Returns a list with the length of smiles_list, each element is
        either None (=failure) or a dictionary of classes and predicted values.
        """
        return self._predictor.predict_smiles(smiles_list)
