from abc import ABC
import torch
import tqdm
from rdkit import Chem

from chebifier.prediction_models.base_predictor import BasePredictor
from chebifier.prediction_models.electra_predictor import ElectraPredictor

MODEL_TYPES = {
    "electra": ElectraPredictor,
    # todo add other model types here
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
        """

        :param smiles_list: list of SMILES strings to predict
        :return: 
            ordered_predictions: torch.Tensor of shape (num_smiles, num_classes, num_models)
            predicted_classes: list of ChEBI IDs predicted by the models
        """
        model_predictions = []
        predicted_classes = set()
        for model in self.models:
            model_predictions.append(model.predict_smiles_list(smiles_list))
            for predicted_smiles in model_predictions[-1]:
                if predicted_smiles is not None:
                    for cls in predicted_smiles:
                        predicted_classes.add(cls)
        print(f"Sorting predictions...")
        predicted_classes = sorted(list(predicted_classes))
        ordered_predictions = torch.zeros(len(smiles_list), len(predicted_classes), len(self.models)) * torch.nan
        for i, model_prediction in enumerate(model_predictions):
            for j, predicted_smiles in tqdm.tqdm(enumerate(model_prediction),
                                                 total=len(model_prediction),
                                                 desc=f"Sorting predictions for {self.models[i].model_name}"):
                if predicted_smiles is not None:
                    for cls in predicted_smiles:
                        ordered_predictions[j, predicted_classes.index(cls), i] = predicted_smiles[cls]
        return ordered_predictions, predicted_classes


    def aggregate_predictions(self, predictions, predicted_classes, **kwargs):
        """
        Aggregates predictions from multiple models using majority voting.

        :param predictions: torch.Tensor of shape (num_smiles, num_classes, num_models)
        :param predicted_classes: list of ChEBI IDs predicted by the models
        :param kwargs: Additional arguments
        :return: list of lists, where each inner list contains the class IDs that received
                 positive predictions from the majority of models for a given SMILES
        """
        num_smiles, num_classes, num_models = predictions.shape
        result = []

        for i in tqdm.tqdm(range(num_smiles), total=num_smiles, desc="Aggregating predictions"):
            smiles_result = []
            for j in range(num_classes):
                # Get predictions for this SMILES and class across all models
                class_predictions = predictions[i, j, :]

                # Count models that made a prediction (not NaN)
                valid_predictions = ~torch.isnan(class_predictions)
                num_valid_predictions = valid_predictions.sum().item()

                # If no valid predictions, skip this class
                if num_valid_predictions == 0:
                    continue

                # Count positive predictions (assuming positive is > 0)
                positive_predictions = class_predictions > 0
                num_positive = (positive_predictions & valid_predictions).sum().item()

                # If majority of models that made a prediction are positive, add this class
                if num_positive > num_valid_predictions / 2:
                    smiles_result.append(predicted_classes[j])

            result.append(smiles_result)

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

    def predict_smiles_list(self, smiles_list) -> list:
        #smiles_list = self.normalize_smiles_list(smiles_list)
        ordered_predictions, predicted_classes = self.gather_predictions(smiles_list)
        aggregated_predictions = self.aggregate_predictions(ordered_predictions, predicted_classes)
        return aggregated_predictions
