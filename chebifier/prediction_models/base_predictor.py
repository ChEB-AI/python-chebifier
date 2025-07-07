from abc import ABC
import json

class BasePredictor(ABC):

    def __init__(self, model_name: str, model_weight: int = 1, classwise_weights_path: str = None, **kwargs):
        self.model_name = model_name
        self.model_weight = model_weight
        if classwise_weights_path is not None:
            self.classwise_weights = json.load(open(classwise_weights_path, encoding="utf-8"))
        else:
            self.classwise_weights = None

        self._description = kwargs.get("description", None)

    def predict_smiles_list(self, smiles_list: list[str]) -> dict:
        raise NotImplementedError

    @property
    def info_text(self):
        if self._description is None:
            return "No description is available for this model."
        return self._description

    def explain_smiles(self, smiles):
        return None
