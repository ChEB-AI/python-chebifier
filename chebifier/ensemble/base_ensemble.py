import os
from abc import ABC
import torch
import tqdm
from rdkit import Chem

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

    def __init__(self, model_configs: dict):
        self.models = []
        for model_name, model_config in model_configs.items():
            model_cls = MODEL_TYPES[model_config["type"]]
            model_instance = model_cls(**model_config)
            assert isinstance(model_instance, BasePredictor)
            self.models.append(model_instance)

    def gather_predictions(self, smiles_list):
        model_predictions = []
        predicted_classes = set()
        for model in self.models:
            model_predictions.append(model.predict_smiles_list(smiles_list))
            for predicted_labels_for_smiles in model_predictions[-1]:
                if predicted_labels_for_smiles is not None:
                    for cls in predicted_labels_for_smiles:
                        predicted_classes.add(cls)
        print(f"Sorting predictions...")
        predicted_classes = sorted(list(predicted_classes))
        predicted_classes = {cls: i for i, cls in enumerate(predicted_classes)}
        ordered_predictions = torch.zeros(len(smiles_list), len(predicted_classes), len(self.models)) * torch.nan
        for i, model_prediction in enumerate(model_predictions):
            for j, predicted_labels_for_smiles in tqdm.tqdm(enumerate(model_prediction),
                                                 total=len(model_prediction),
                                                 desc=f"Sorting predictions for {self.models[i].model_name}"):
                if predicted_labels_for_smiles is not None:
                    for cls in predicted_labels_for_smiles:
                        ordered_predictions[j, predicted_classes[cls], i] = predicted_labels_for_smiles[cls]
        return ordered_predictions, predicted_classes


    def consolidate_predictions(self, predictions, predicted_classes, classwise_weights, **kwargs):
        """
        Aggregates predictions from multiple models using weighted majority voting.
        Optimized version using tensor operations instead of for loops.
        """
        num_smiles, num_classes, num_models = predictions.shape

        # Create a mapping from class indices to class names for faster lookup
        class_names = list(predicted_classes.keys())
        class_indices = {predicted_classes[cls]: cls for cls in class_names}

        # Get predictions for all classes
        valid_predictions = ~torch.isnan(predictions)
        valid_counts = valid_predictions.sum(dim=2)  # Sum over models dimension

        # Skip classes with no valid predictions
        has_valid_predictions = valid_counts > 0

        # Calculate positive and negative predictions for all classes at once
        positive_mask = (predictions > 0.5) & valid_predictions
        negative_mask = (predictions < 0.5) & valid_predictions

        # Extract positive and negative weights
        pos_weights = classwise_weights[0]  # Shape: (num_classes, num_models)
        neg_weights = classwise_weights[1]  # Shape: (num_classes, num_models)

        # Calculate weighted predictions using broadcasting
        # predictions shape: (num_smiles, num_classes, num_models)
        # weights shape: (num_classes, num_models)
        positive_weighted = positive_mask.float() * (predictions.nan_to_num() - 0.5) * pos_weights.unsqueeze(0)
        negative_weighted = negative_mask.float() * (0.5 - predictions.nan_to_num()) * neg_weights.unsqueeze(0)

        # Sum over models dimension
        positive_sum = positive_weighted.sum(dim=2)  # Shape: (num_smiles, num_classes)
        negative_sum = negative_weighted.sum(dim=2)  # Shape: (num_smiles, num_classes)

        # Determine which classes to include for each SMILES
        net_score = positive_sum - negative_sum  # Shape: (num_smiles, num_classes)
        class_decisions = (net_score > 0) & has_valid_predictions  # Shape: (num_smiles, num_classes)

        # Convert tensor decisions to result list using list comprehension for efficiency
        result = [
            [class_indices[idx.item()] for idx in torch.nonzero(class_decisions[i], as_tuple=True)[0]]
            for i in range(num_smiles)
        ]

        return result

    def normalize_smiles_list(self, smiles_list):
        new = []
        print(f"Normalizing SMILES strings...")
        for smiles in tqdm.tqdm(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                canonical_smiles = Chem.MolToSmiles(mol)
            except Exception as e:
                print(f"Failed to parse SMILES '{smiles}': {e}")
                canonical_smiles = None
            new.append(canonical_smiles)
        return new

    def calculate_classwise_weights(self, predicted_classes):
        """No weights, simple majority voting"""
        positive_weights = torch.ones(len(predicted_classes), len(self.models))
        negative_weights = torch.ones(len(predicted_classes), len(self.models))

        return positive_weights, negative_weights

    def predict_smiles_list(self, smiles_list, load_preds_if_possible=True) -> list:
        preds_file = f"predictions_by_model_{'_'.join(model.model_name for model in self.models)}.pt"
        predicted_classes_file = f"predicted_classes_{'_'.join(model.model_name for model in self.models)}.txt"
        if not load_preds_if_possible or not os.path.isfile(preds_file):
            #smiles_list = self.normalize_smiles_list(smiles_list)
            ordered_predictions, predicted_classes = self.gather_predictions(smiles_list)
            # save predictions
            torch.save(ordered_predictions, preds_file)
            with open(predicted_classes_file, "w") as f:
                for cls in predicted_classes:
                    f.write(f"{cls}\n")
        else:
            print(f"Loading predictions from {preds_file} and label indexes from {predicted_classes_file}")
            ordered_predictions = torch.load(preds_file)
            with open(predicted_classes_file, "r") as f:
                predicted_classes = {line.strip(): i for i, line in enumerate(f.readlines())}

        classwise_weights = self.calculate_classwise_weights(predicted_classes)
        aggregated_predictions = self.consolidate_predictions(ordered_predictions, predicted_classes, classwise_weights)
        return aggregated_predictions
