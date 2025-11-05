import importlib
import time
from pathlib import Path
from typing import Union

import torch
import tqdm
import yaml

from chebifier.check_env import check_package_installed
from chebifier.hugging_face import download_model_files
from chebifier.inconsistency_resolution import ScoreBasedPredictionSmoother
from chebifier.prediction_models.base_predictor import BasePredictor
from chebifier.utils import (
    get_default_configs,
    get_disjoint_files,
    load_chebi_graph,
    process_config,
)


class BaseEnsemble:
    def __init__(
        self,
        model_configs: Union[str, Path, dict, None] = None,
        resolve_inconsistencies: bool = True,
        verbose_output: bool = False,
        use_confidence: bool = True,
    ):
        # Deferred Import: To avoid circular import error
        from chebifier.model_registry import MODEL_TYPES

        # Load configuration from YAML file
        if not model_configs:
            config = get_default_configs()
        elif isinstance(model_configs, dict):
            config = model_configs
        else:
            print(f"Loading ensemble configuration from {model_configs}")
            with open(model_configs, "r") as f:
                config = yaml.safe_load(f)

        with (
            importlib.resources.files("chebifier")
            .joinpath("model_registry.yml")
            .open("r") as f
        ):
            model_registry = yaml.safe_load(f)

        processed_configs = process_config(config, model_registry)
        self.verbose_output = verbose_output
        self.use_confidence = use_confidence

        self.chebi_graph = load_chebi_graph()
        self.disjoint_files = get_disjoint_files()

        self.models = []
        self.positive_prediction_threshold = 0.5
        for model_name, model_config in processed_configs.items():
            model_cls = MODEL_TYPES[model_config["type"]]
            if "hugging_face" in model_config:
                hugging_face_kwargs = download_model_files(model_config["hugging_face"])
            else:
                hugging_face_kwargs = {}
            if "package_name" in model_config:
                check_package_installed(model_config["package_name"])

            model_instance = model_cls(
                model_name,
                **model_config,
                **hugging_face_kwargs,
                chebi_graph=self.chebi_graph,
            )
            assert isinstance(model_instance, BasePredictor)
            self.models.append(model_instance)

        if resolve_inconsistencies:
            self.smoother = ScoreBasedPredictionSmoother(
                self.chebi_graph,
                label_names=None,
                disjoint_files=self.disjoint_files,
                verbose=self.verbose_output,
            )
        else:
            self.smoother = None

    def gather_predictions(self, smiles_list):
        # get predictions from all models for the SMILES list
        # order them alphabetically by label class
        model_predictions = []
        predicted_classes = set()
        for model in self.models:
            model_predictions.append(model.predict_smiles_list(smiles_list))
            for logits_for_smiles in model_predictions[-1]:
                if logits_for_smiles is not None:
                    for cls in logits_for_smiles:
                        predicted_classes.add(cls)
        if self.verbose_output:
            print(f"Sorting predictions from {len(model_predictions)} models...")
        predicted_classes = sorted(list(predicted_classes))
        predicted_classes_dict = {cls: i for i, cls in enumerate(predicted_classes)}
        ordered_logits = (
            torch.zeros(len(smiles_list), len(predicted_classes), len(self.models))
            * torch.nan
        )
        for i, model_prediction in enumerate(model_predictions):
            for j, logits_for_smiles in tqdm.tqdm(
                enumerate(model_prediction),
                total=len(model_prediction),
                desc=f"Sorting predictions for {self.models[i].model_name}",
            ):
                if logits_for_smiles is not None:
                    for cls in logits_for_smiles:
                        ordered_logits[j, predicted_classes_dict[cls], i] = (
                            logits_for_smiles[cls]
                        )

        return ordered_logits, predicted_classes

    def consolidate_predictions(
        self,
        predictions,
        classwise_weights,
        return_intermediate_results=False,
        **kwargs,
    ):
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
        positive_mask = (
            predictions > self.positive_prediction_threshold
        ) & valid_predictions
        negative_mask = (
            predictions < self.positive_prediction_threshold
        ) & valid_predictions

        # if use_confidence is passed in kwargs, it overrides the ensemble setting
        use_confidence = kwargs.get("use_confidence", self.use_confidence)
        if use_confidence:
            confidence = 2 * torch.abs(
                predictions.nan_to_num() - self.positive_prediction_threshold
            )
        else:
            confidence = torch.ones_like(predictions)

        # Extract positive and negative weights
        pos_weights = classwise_weights[0]  # Shape: (num_classes, num_models)
        neg_weights = classwise_weights[1]  # Shape: (num_classes, num_models)

        # Calculate weighted predictions using broadcasting
        # predictions shape: (num_smiles, num_classes, num_models)
        # weights shape: (num_classes, num_models)
        positive_weighted = (
            positive_mask.float() * confidence * pos_weights.unsqueeze(0)
        )
        negative_weighted = (
            negative_mask.float() * confidence * neg_weights.unsqueeze(0)
        )

        # Sum over models dimension
        positive_sum = positive_weighted.sum(dim=2)  # Shape: (num_smiles, num_classes)
        negative_sum = negative_weighted.sum(dim=2)  # Shape: (num_smiles, num_classes)

        # Determine which classes to include for each SMILES
        net_score = positive_sum - negative_sum  # Shape: (num_smiles, num_classes)
        if return_intermediate_results:
            return (
                net_score,
                has_valid_predictions,
                {
                    "positive_mask": positive_mask,
                    "negative_mask": negative_mask,
                    "confidence": confidence,
                    "positive_sum": positive_sum,
                    "negative_sum": negative_sum,
                },
            )

        return net_score, has_valid_predictions

    def apply_inconsistency_resolution(
        self, net_score, class_names, has_valid_predictions
    ):
        # Smooth predictions
        start_time = time.perf_counter()
        if self.smoother is not None:
            self.smoother.set_label_names(class_names)
            smooth_net_score = self.smoother(net_score)
            class_decisions = (
                smooth_net_score > 0
            ) & has_valid_predictions  # Shape: (num_smiles, num_classes)
        else:
            class_decisions = (
                net_score > 0
            ) & has_valid_predictions  # Shape: (num_smiles, num_classes)
        end_time = time.perf_counter()
        if self.verbose_output:
            print(f"Prediction smoothing took {end_time - start_time:.2f} seconds")

        complete_failure = torch.all(~has_valid_predictions, dim=1)
        return class_decisions, complete_failure

    def calculate_classwise_weights(self, predicted_classes):
        """No weights, simple majority voting"""
        positive_weights = torch.ones(len(predicted_classes), len(self.models))
        negative_weights = torch.ones(len(predicted_classes), len(self.models))

        return positive_weights, negative_weights

    def predict_smiles_list(
        self, smiles_list, return_intermediate_results=False, **kwargs
    ) -> list:
        ordered_predictions, predicted_classes = self.gather_predictions(smiles_list)
        if len(predicted_classes) == 0:
            print("Warning: No classes have been predicted for the given SMILES list.")
        predicted_classes = {cls: i for i, cls in enumerate(predicted_classes)}

        classwise_weights = self.calculate_classwise_weights(predicted_classes)
        if return_intermediate_results:
            net_score, has_valid_predictions, intermediate_results_dict = (
                self.consolidate_predictions(
                    ordered_predictions,
                    classwise_weights,
                    return_intermediate_results=return_intermediate_results,
                )
            )
        else:
            net_score, has_valid_predictions = self.consolidate_predictions(
                ordered_predictions, classwise_weights
            )
        class_decisions, is_failure = self.apply_inconsistency_resolution(
            net_score, list(predicted_classes.keys()), has_valid_predictions
        )

        class_names = list(predicted_classes.keys())
        class_indices = {predicted_classes[cls]: cls for cls in class_names}
        result = [
            (
                [
                    class_indices[idx.item()]
                    for idx in torch.nonzero(i, as_tuple=True)[0]
                ]
                if not failure
                else None
            )
            for i, failure in zip(class_decisions, is_failure)
        ]
        if return_intermediate_results:
            intermediate_results_dict["predicted_classes"] = predicted_classes
            intermediate_results_dict["classwise_weights"] = classwise_weights
            intermediate_results_dict["net_score"] = net_score
            return result, intermediate_results_dict

        return result


if __name__ == "__main__":
    ensemble = BaseEnsemble(
        {
            "resgated_0ps1g189": {
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
                # "classwise_weights_path" : "../python-chebai/metrics_0ps1g189_80-10-10.json"
            },
            "electra_14ko0zcf": {
                "type": "electra",
                "ckpt_path": "data/14ko0zcf/epoch=193.ckpt",
                "target_labels_path": "data/chebi_v241/ChEBI50/processed/classes.txt",
                # "classwise_weights_path": "../python-chebai/metrics_electra_14ko0zcf_80-10-10.json",
            },
        }
    )
    r = ensemble.predict_smiles_list(
        [
            "[NH3+]CCCC[C@H](NC(=O)[C@@H]([NH3+])CC([O-])=O)C([O-])=O",
            "C[C@H](N)C(=O)NCC(O)=O#",
            "",
        ],
        load_preds_if_possible=False,
    )
    print(len(r), r[0])
