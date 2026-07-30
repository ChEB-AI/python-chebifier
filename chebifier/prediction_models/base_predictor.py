from abc import ABC

import numpy as np
from rdkit import Chem

from .._custom_cache import modelwise_smiles_lru_cache

SCORE_DTYPE = np.float16


def dicts_to_dense(predictions: list[dict | None]) -> tuple[list[str], np.ndarray]:
    class_set = set()
    previous_keys = None
    for pred in predictions:
        if pred:
            keys = tuple(pred)
            if keys != previous_keys:
                class_set.update(keys)
                previous_keys = keys
    classes = sorted(class_set)
    cls_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    scores = np.full((len(predictions), len(classes)), np.nan, dtype=SCORE_DTYPE)
    previous_keys = None
    columns = None
    for i, pred in enumerate(predictions):
        if pred:
            keys = tuple(pred)
            if keys != previous_keys:
                columns = np.fromiter(
                    (cls_to_idx[cls] for cls in keys), dtype=np.intp, count=len(keys)
                )
                previous_keys = keys
            scores[i, columns] = np.fromiter(
                pred.values(), dtype=SCORE_DTYPE, count=len(pred)
            )
    return classes, scores


class BasePredictor(ABC):
    def __init__(
        self,
        model_name: str,
        model_weight: int = 1,
        **kwargs,
    ):
        self.model_name = model_name
        self.model_weight = model_weight

        self._description = kwargs.get("description", None)

    @modelwise_smiles_lru_cache.batch_decorator
    def predict_list(self, molecule_list: list[str | Chem.Mol]) -> list[dict | None]:
        raise NotImplementedError()

    def predict(self, molecule: str | Chem.Mol) -> dict | None:
        # by default, use list-based prediction
        return self.predict_list([molecule])[0]

    def predict_dense(
        self, molecule_list: list[str | Chem.Mol]
    ) -> tuple[list[str], np.ndarray]:
        return dicts_to_dense(self.predict_list(molecule_list))

    @property
    def info_text(self):
        if self._description is None:
            return "No description is available for this model."
        return self._description

    def explain_smiles(self, smiles):
        return None
