from typing import Optional, List
from pathlib import Path

from c3p import classifier as c3p_classifier

from chebifier.prediction_models import BasePredictor


class C3PPredictor(BasePredictor):
    """
    Wrapper for C3P (url).
    """

    def __init__(self, model_name: str, program_directory: Optional[Path]=None, chemical_classes: Optional[List[str]]=None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.program_directory = program_directory
        self.chemical_classes = chemical_classes

    def predict_smiles_list(self, smiles_list: list[str]) -> list:
        result_list = c3p_classifier.classify(smiles_list, self.program_directory, self.chemical_classes, strict=False)
        result_reformatted = [dict() for _ in range(len(smiles_list))]
        for result in result_list:
            result_reformatted[smiles_list.index(result.input_smiles)][result.class_id.split(":")[1]] = result.is_match
        print(f"C3P predictions for {len(smiles_list)} SMILES strings:")
        for i, smiles in enumerate(smiles_list):
            print(f"{smiles}: {result_reformatted[i]}")
        return result_reformatted