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
        self.chebi_graph = kwargs.get("chebi_graph", None)

    def predict_smiles_list(self, smiles_list: list[str]) -> list:
        result_list = c3p_classifier.classify(smiles_list, self.program_directory, self.chemical_classes, strict=True)
        result_reformatted = [dict() for _ in range(len(smiles_list))]
        for result in result_list:
            chebi_id = result.class_id.split(":")[1]
            result_reformatted[smiles_list.index(result.input_smiles)][chebi_id] = result.is_match
            if result.is_match and self.chebi_graph is not None:
                for parent in list(self.chebi_graph.predecessors(int(chebi_id))):
                    result_reformatted[smiles_list.index(result.input_smiles)][str(parent)] = 1
        return result_reformatted