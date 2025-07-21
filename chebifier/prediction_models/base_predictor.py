import json
from abc import ABC

from functools import lru_cache


class BasePredictor(ABC):
    def __init__(
        self,
        model_name: str,
        model_weight: int = 1,
        classwise_weights_path: str = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.model_weight = model_weight
        if classwise_weights_path is not None:
            self.classwise_weights = json.load(
                open(classwise_weights_path, encoding="utf-8")
            )
        else:
            self.classwise_weights = None

        self._description = kwargs.get("description", None)

    def predict_smiles_list(self, smiles_list: list[str]) -> dict:
        # list is not hashable, so we convert it to a tuple (useful for caching)
        return self.predict_smiles_tuple(tuple(smiles_list))

    @lru_cache(maxsize=100)
    def predict_smiles_tuple(self, smiles_tuple: tuple[str]) -> dict:
        raise NotImplementedError()

    def predict_smiles(self, smiles: str) -> dict:
        # by default, use list-based prediction
        return self.predict_smiles_tuple((smiles,))[0]

    @property
    def info_text(self):
        if self._description is None:
            return "No description is available for this model."
        return self._description

    def explain_smiles(self, smiles):
        return None
