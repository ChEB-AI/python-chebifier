from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
from chebai.result.prediction import Predictor
from rdkit import Chem

from chebifier import modelwise_smiles_lru_cache

from .base_predictor import SCORE_DTYPE, BasePredictor

if TYPE_CHECKING:
    pass


class NNPredictor(BasePredictor, ABC):
    def __init__(
        self,
        model_name: str,
        ckpt_path: str,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.batch_size = kwargs.get("batch_size", None)
        # compile_model will run the model in eager mode, which gives better performance, but does not return intermediate states
        # such as attention weights. Therefore, ELECTRA attention graphs will only work with compile_model=False.
        compile_model = kwargs.get("compile_model", True)
        # If batch_size is not provided, it will be set to default batch size used during training in Predictor
        self.predictor: Predictor = Predictor(
            ckpt_path, self.batch_size, compile_model=compile_model
        )

    @modelwise_smiles_lru_cache.batch_decorator
    def predict_list(self, smiles_list: list[str]) -> list:
        """
        Returns a list with the length of smiles_list, each element is
        either None (=failure) or a dictionary of classes and predicted values.
        """
        raw_preds = self.predictor.predict_smiles(smiles_list)
        if raw_preds is not None:
            preds = [
                {
                    label: pred
                    for label, pred in zip(
                        self.predictor._classification_labels, pred_tensor.tolist()
                    )
                }
                for pred_tensor in raw_preds
            ]
            return preds
        else:
            return [None for _ in smiles_list]

    def predict_dense(
        self, molecule_list: list[str | Chem.Mol]
    ) -> tuple[list[str], np.ndarray]:
        raw_preds = self.predictor.predict_smiles(molecule_list)
        if raw_preds is None:
            return [], np.full((len(molecule_list), 0), np.nan, dtype=SCORE_DTYPE)
        classes = [str(label) for label in self.predictor._classification_labels]
        scores = np.stack([pred.detach().cpu().numpy() for pred in raw_preds]).astype(
            SCORE_DTYPE
        )
        return classes, scores

    def calculate_results(self, batch):
        collator = self.predictor._dm.reader.COLLATOR()
        dat = self.predictor._model._process_batch(
            collator(batch).to(self.predictor.device), 0
        )

        return self.predictor._model(dat, **dat["model_kwargs"])
