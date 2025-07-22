import os
import time

import torch
import tqdm
from chebifier.inconsistency_resolution import PredictionSmoother
from chebifier.utils import load_chebi_graph, get_disjoint_files

from chebifier.check_env import check_package_installed
from chebifier.prediction_models.base_predictor import BasePredictor


class BaseEnsemble:

    def __init__(
        self,
        model_configs: dict,
        chebi_version: int = 241,
        resolve_inconsistencies: bool = True,
    ):
        # Deferred Import: To avoid circular import error
        from chebifier.model_registry import MODEL_TYPES

        self.chebi_graph = load_chebi_graph()
        self.disjoint_files = get_disjoint_files()

        self.models = []
        self.positive_prediction_threshold = 0.5
        for model_name, model_config in model_configs.items():
            model_cls = MODEL_TYPES[model_config["type"]]
            if "hugging_face" in model_config:
                from chebifier.hugging_face import download_model_files

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
            self.smoother = PredictionSmoother(
                self.chebi_graph,
                label_names=None,
                disjoint_files=self.disjoint_files,
            )
        else:
            self.smoother = None

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
        self, predictions, classwise_weights, predicted_classes, **kwargs
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

        if "use_confidence" in kwargs and kwargs["use_confidence"]:
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

        # Smooth predictions
        start_time = time.perf_counter()
        class_names = list(predicted_classes.keys())
        if self.smoother is not None:
            self.smoother.set_label_names(class_names)
            smooth_net_score = self.smoother(net_score)
            class_decisions = (
                smooth_net_score > 0.5
            ) & has_valid_predictions  # Shape: (num_smiles, num_classes)
        else:
            class_decisions = (
                net_score > 0
            ) & has_valid_predictions  # Shape: (num_smiles, num_classes)
        end_time = time.perf_counter()
        print(f"Prediction smoothing took {end_time - start_time:.2f} seconds")

        complete_failure = torch.all(~has_valid_predictions, dim=1)
        return class_decisions, complete_failure

    def calculate_classwise_weights(self, predicted_classes):
        """No weights, simple majority voting"""
        positive_weights = torch.ones(len(predicted_classes), len(self.models))
        negative_weights = torch.ones(len(predicted_classes), len(self.models))

        return positive_weights, negative_weights

    def predict_smiles_list(
        self, smiles_list, load_preds_if_possible=False, **kwargs
    ) -> list:
        preds_file = f"predictions_by_model_{'_'.join(model.model_name for model in self.models)}.pt"
        predicted_classes_file = f"predicted_classes_{'_'.join(model.model_name for model in self.models)}.txt"
        if not load_preds_if_possible or not os.path.isfile(preds_file):
            ordered_predictions, predicted_classes = self.gather_predictions(
                smiles_list
            )
            if len(predicted_classes) == 0:
                print(
                    "Warning: No classes have been predicted for the given SMILES list."
                )
            # save predictions
            if load_preds_if_possible:
                torch.save(ordered_predictions, preds_file)
                with open(predicted_classes_file, "w") as f:
                    for cls in predicted_classes:
                        f.write(f"{cls}\n")
            predicted_classes = {cls: i for i, cls in enumerate(predicted_classes)}
        else:
            print(
                f"Loading predictions from {preds_file} and label indexes from {predicted_classes_file}"
            )
            ordered_predictions = torch.load(preds_file)
            with open(predicted_classes_file, "r") as f:
                predicted_classes = {
                    line.strip(): i for i, line in enumerate(f.readlines())
                }

        classwise_weights = self.calculate_classwise_weights(predicted_classes)
        class_decisions, is_failure = self.consolidate_predictions(
            ordered_predictions, classwise_weights, predicted_classes, **kwargs
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
