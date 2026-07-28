from abc import ABC

from rdkit import Chem

from .._custom_cache import modelwise_smiles_lru_cache


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
    def predict_list(self, molecule_list: list[str | Chem.Mol]) -> dict:
        raise NotImplementedError()

    def predict(self, molecule: str | Chem.Mol) -> dict:
        # by default, use list-based prediction
        return self.predict_list([molecule])[0]

    @property
    def info_text(self):
        if self._description is None:
            return "No description is available for this model."
        return self._description

    def explain_smiles(self, smiles):
        return None
