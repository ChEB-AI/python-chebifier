import os
from abc import ABC
import torch
import tqdm
from chebai.preprocessing.datasets.chebi import ChEBIOver50
from chebai.result.analyse_sem import PredictionSmoother

from chebifier.prediction_models.base_predictor import BasePredictor
from chebifier.prediction_models.chemlog_predictor import ChemLogPredictor
from chebifier.prediction_models.electra_predictor import ElectraPredictor
from chebifier.prediction_models.gnn_predictor import ResGatedPredictor

MODEL_TYPES = {
    "electra": ElectraPredictor,
    "resgated": ResGatedPredictor,
    "chemlog": ChemLogPredictor
}

class BaseEnsemble(ABC):

    def __init__(self, model_configs: dict, chebi_version: int = 241):
        self.models = []
        self.positive_prediction_threshold = 0.5
        for model_name, model_config in model_configs.items():
            model_cls = MODEL_TYPES[model_config["type"]]
            model_instance = model_cls(model_name, **model_config)
            assert isinstance(model_instance, BasePredictor)
            self.models.append(model_instance)

        self.smoother = PredictionSmoother(ChEBIOver50(chebi_version=chebi_version), disjoint_files=[
            os.path.join("data", "disjoint_chebi.csv"),
            os.path.join("data", "disjoint_additional.csv")
        ])


    def gather_predictions(self, smiles_list):
        # get predictions from all models for the SMILES list
        # order them by alphabetically by label class
        model_predictions = []
        predicted_classes = set()
        for model in self.models:
            model_predictions.append(model.predict_smiles_list(smiles_list))
            for logits_for_smiles in model_predictions[-1]:
                if logits_for_smiles is not None:
                    for cls in logits_for_smiles:
                        predicted_classes.add(cls)
        print(f"Sorting predictions...")
        predicted_classes = sorted(list(predicted_classes))
        predicted_classes_dict = {cls: i for i, cls in enumerate(predicted_classes)}
        ordered_logits = torch.zeros(len(smiles_list), len(predicted_classes), len(self.models)) * torch.nan
        for i, model_prediction in enumerate(model_predictions):
            for j, logits_for_smiles in tqdm.tqdm(enumerate(model_prediction),
                                                 total=len(model_prediction),
                                                 desc=f"Sorting predictions for {self.models[i].model_name}"):
                if logits_for_smiles is not None:
                    for cls in logits_for_smiles:
                        ordered_logits[j, predicted_classes_dict[cls], i] = logits_for_smiles[cls]

        return ordered_logits, predicted_classes


    def consolidate_predictions(self, predictions, classwise_weights, **kwargs):
        """
        Aggregates predictions from multiple models using weighted majority voting.
        Optimized version using tensor operations instead of for loops.
        """
        num_smiles, num_classes, num_models = predictions.shape

        # Get predictions for all classes
        valid_predictions = ~torch.isnan(predictions)
        valid_counts = valid_predictions.sum(dim=2)  # Sum over models dimension

        # Skip classes with no valid predictions
        has_valid_predictions = valid_counts > 0

        # Calculate positive and negative predictions for all classes at once
        positive_mask = (predictions > self.positive_prediction_threshold) & valid_predictions
        negative_mask = (predictions < self.positive_prediction_threshold) & valid_predictions

        if "use_confidence" in kwargs and kwargs["use_confidence"]:
            confidence = 2 * torch.abs(predictions.nan_to_num() - self.positive_prediction_threshold)
        else:
            confidence = torch.ones_like(predictions)

        # Extract positive and negative weights
        pos_weights = classwise_weights[0]  # Shape: (num_classes, num_models)
        neg_weights = classwise_weights[1]  # Shape: (num_classes, num_models)

        # Calculate weighted predictions using broadcasting
        # predictions shape: (num_smiles, num_classes, num_models)
        # weights shape: (num_classes, num_models)
        positive_weighted = positive_mask.float() * confidence * pos_weights.unsqueeze(0)
        negative_weighted = negative_mask.float() * confidence * neg_weights.unsqueeze(0)

        # Sum over models dimension
        positive_sum = positive_weighted.sum(dim=2)  # Shape: (num_smiles, num_classes)
        negative_sum = negative_weighted.sum(dim=2)  # Shape: (num_smiles, num_classes)

        # Determine which classes to include for each SMILES
        net_score = positive_sum - negative_sum  # Shape: (num_smiles, num_classes)
        class_decisions = (net_score > 0) & has_valid_predictions  # Shape: (num_smiles, num_classes)



        return class_decisions

    def calculate_classwise_weights(self, predicted_classes):
        """No weights, simple majority voting"""
        positive_weights = torch.ones(len(predicted_classes), len(self.models))
        negative_weights = torch.ones(len(predicted_classes), len(self.models))

        return positive_weights, negative_weights

    def predict_smiles_list(self, smiles_list, load_preds_if_possible=True) -> list:
        preds_file = f"predictions_by_model_{'_'.join(model.model_name for model in self.models)}.pt"
        predicted_classes_file = f"predicted_classes_{'_'.join(model.model_name for model in self.models)}.txt"
        if not load_preds_if_possible or not os.path.isfile(preds_file):
            ordered_predictions, predicted_classes = self.gather_predictions(smiles_list)
            # save predictions
            torch.save(ordered_predictions, preds_file)
            with open(predicted_classes_file, "w") as f:
                for cls in predicted_classes:
                    f.write(f"{cls}\n")
            predicted_classes = {cls: i for i, cls in enumerate(predicted_classes)}
        else:
            print(f"Loading predictions from {preds_file} and label indexes from {predicted_classes_file}")
            ordered_predictions = torch.load(preds_file)
            with open(predicted_classes_file, "r") as f:
                predicted_classes = {line.strip(): i for i, line in enumerate(f.readlines())}

        classwise_weights = self.calculate_classwise_weights(predicted_classes)
        class_decisions = self.consolidate_predictions(ordered_predictions, classwise_weights)
        # Smooth predictions
        class_names = list(predicted_classes.keys())
        self.smoother.label_names = class_names
        class_decisions = self.smoother(class_decisions)

        class_names = list(predicted_classes.keys())
        class_indices = {predicted_classes[cls]: cls for cls in class_names}
        result = [
            [class_indices[idx.item()] for idx in torch.nonzero(i, as_tuple=True)[0]]
            for i in class_decisions
        ]

        return result

if __name__ == "__main__":
    ensemble = BaseEnsemble({"resgated_0ps1g189":{
  "type": "resgated",
  "ckpt_path": "data/0ps1g189/epoch=122.ckpt",
  "target_labels_path": "data/chebi_v241/ChEBI50/processed/classes.txt",
 "molecular_properties": [
      "chebai_graph.preprocessing.properties.AtomType",
      "chebai_graph.preprocessing.properties.NumAtomBonds",
      "chebai_graph.preprocessing.properties.AtomCharge",
      "chebai_graph.preprocessing.properties.AtomAromaticity",
      "chebai_graph.preprocessing.properties.AtomHybridization",
      "chebai_graph.preprocessing.properties.AtomNumHs",
      "chebai_graph.preprocessing.properties.BondType",
      "chebai_graph.preprocessing.properties.BondInRing",
      "chebai_graph.preprocessing.properties.BondAromaticity",
      "chebai_graph.preprocessing.properties.RDKit2DNormalized",
    ],
  #"classwise_weights_path" : "../python-chebai/metrics_0ps1g189_80-10-10.json"
    },

"electra_14ko0zcf": {
  "type": "electra",
  "ckpt_path": "data/14ko0zcf/epoch=193.ckpt",
  "target_labels_path": "data/chebi_v241/ChEBI50/processed/classes.txt",
  #"classwise_weights_path": "../python-chebai/metrics_electra_14ko0zcf_80-10-10.json",
}
    })
    r = ensemble.predict_smiles_list(["[NH3+]CCCC[C@H](NC(=O)[C@@H]([NH3+])CC([O-])=O)C([O-])=O"], load_preds_if_possible=False)
    print(len(r), r[0])