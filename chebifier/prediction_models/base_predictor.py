from abc import ABC


class BasePredictor(ABC):

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    def predict_smiles_list(self, smiles_list: list[str]) -> dict:
        raise NotImplementedError