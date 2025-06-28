import tqdm
from chemlog.cli import CLASSIFIERS, _smiles_to_mol, strategy_call

from chebifier.prediction_models.base_predictor import BasePredictor


class ChemLogPredictor(BasePredictor):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.strategy = "algo"
        self.classifier_instances = {
            k: v() for k, v in CLASSIFIERS[self.strategy].items()
        }
        # fmt: off
        self.peptide_labels = [
            "15841", "16670", "24866", "25676", "25696", "25697", "27369", "46761", "47923",
            "48030", "48545", "60194", "60334", "60466", "64372", "65061", "90799", "155837"
        ]
        # fmt: on
        print(f"Initialised ChemLog model {self.model_name}")

    def predict_smiles_list(self, smiles_list: list[str]) -> list:
        results = []
        for i, smiles in tqdm.tqdm(enumerate(smiles_list)):
            mol = _smiles_to_mol(smiles)
            if mol is None:
                results.append(None)
            else:
                results.append(
                    {
                        label: (
                            1
                            if label
                            in strategy_call(
                                self.strategy, self.classifier_instances, mol
                            )["chebi_classes"]
                            else 0
                        )
                        for label in self.peptide_labels
                    }
                )

        for classifier in self.classifier_instances.values():
            classifier.on_finish()

        return results
