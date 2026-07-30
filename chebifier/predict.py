# Get end-to-end predictions (from SMILES / molecule list via base learners + ensemble + inconsistency resolution to ChEBI classes)


import os
from typing import Optional

import numpy as np
import torch
from rdkit import Chem

from chebifier.ensemble.base_ensemble import BaseEnsemble
from chebifier.inconsistency_resolution import ScoreBasedPredictionSmoother
from chebifier.prediction_models.base_predictor import BasePredictor
from chebifier.utils import get_disjoint_files, load_chebi_graph


def apply_inconsistency_resolution(smoother, class_names, aggregated_predictions):
    smoother.set_label_names(class_names)
    smooth_net_score = smoother(aggregated_predictions["net_score"])
    aggregated_predictions["net_score"] = smooth_net_score
    return aggregated_predictions


def collect_base_learner_predictions(
    predictions: dict[str, list[dict | None]],
    classes: Optional[list[str]] = None,
) -> (dict[str, torch.Tensor], list[str]):
    """
    Collect predictions from base learners into a single dictionary.

    Args:
        predictions (dict): A dictionary where keys are model names and values are lists of predictions.
            Assumes those lists have the same length and each entry in the list is either None or a dict
            mapping class labels to predicted values.
        classes (Optional[list]): Column space to map the predictions onto. If None (the default), the
            union of all classes reported by the base learners is used. Pass an explicit list to align
            the predictions with a fixed label set (e.g. the labels of an evaluation dataset) - base
            learners may be trained on different label sets, so the union does not necessarily match the
            labels you want to compare against. Classes predicted by a base learner but missing from
            `classes` are dropped, classes that no base learner covers stay NaN.

    Returns:
        dict: A dictionary where keys are model names and values are tensors of predictions with shape (num_samples, num_classes).
            If a prediction is None, it will be replaced with NaN.
        ensemble_classes (list): A list of class labels that are present in the predictions.
    """
    collected_predictions = {}
    ensemble_classes = set()
    n_samples = -1
    print(f"Collecting base learner predictions from {len(predictions)} models...")
    # step 1: collect classes
    # Base learners typically return the same label set for every sample, so we only
    # touch the class set when the label set actually changes from one sample to the next.
    for model_name, model_predictions in predictions.items():
        if classes is None:
            previous_keys = None
            for pred in model_predictions:
                if pred:
                    keys = tuple(pred)
                    if keys != previous_keys:
                        ensemble_classes.update(keys)
                        previous_keys = keys
        if n_samples == -1:
            n_samples = len(model_predictions)
        else:
            assert n_samples == len(
                model_predictions
            ), f"All prediction lists must have the same length. Model {model_name} has {len(model_predictions)} predictions, expected {n_samples}."
    if classes is None:
        ensemble_classes = sorted(ensemble_classes)  # Sort for consistent ordering
    else:
        ensemble_classes = list(classes)
    cls_to_idx = {cls: idx for idx, cls in enumerate(ensemble_classes)}
    # step 2: map predictions to tensors
    # Filled row-wise via numpy fancy indexing: one vectorised write per sample instead
    # of one Python-level tensor assignment per predicted class. The column indices are
    # reused as long as consecutive samples share the same label set (see step 1).
    for model_name, model_predictions in predictions.items():
        # Samples without a prediction (and classes the model does not cover) stay NaN
        predictions_array = np.full(
            (n_samples, len(ensemble_classes)), np.nan, dtype=np.float32
        )
        previous_keys = None
        columns = None
        # positions of the kept values within pred.values(), None if nothing is dropped
        kept_positions = None
        for i, pred in enumerate(model_predictions):
            if pred:
                keys = tuple(pred)
                if keys != previous_keys:
                    kept = [
                        (position, cls_to_idx[cls])
                        for position, cls in enumerate(keys)
                        if cls in cls_to_idx
                    ]
                    columns = np.fromiter(
                        (column for _, column in kept), dtype=np.intp, count=len(kept)
                    )
                    kept_positions = (
                        None
                        if len(kept) == len(keys)
                        else np.fromiter(
                            (position for position, _ in kept),
                            dtype=np.intp,
                            count=len(kept),
                        )
                    )
                    previous_keys = keys
                values = np.fromiter(pred.values(), dtype=np.float32, count=len(pred))
                predictions_array[i, columns] = (
                    values if kept_positions is None else values[kept_positions]
                )
        collected_predictions[model_name] = torch.from_numpy(predictions_array)

    return collected_predictions, list(ensemble_classes)


def predict(
    base_learners: dict[str, BasePredictor],
    ensemble_model: BaseEnsemble,
    molecules: list[str | Chem.Mol],
    prediction_cache_dir: Optional[str] = None,
    resolve_inconsistencies: bool = True,
    decision_threshold: float = 0,
) -> dict:
    """
    Get end-to-end predictions from base learners and an ensemble model.

    Args:
        base_learners (dict[str, BasePredictor]): A dictionary of base learner models.
        ensemble_model (BaseEnsemble): An instance of a BaseEnsemble model.
        molecules (list[str | Chem.Mol]): List of molecules for prediction (either SMILES strings or molecule objects).
        prediction_cache_dir (Optional[str]): Directory to cache predictions. If None, no caching is performed. If provided,
            predictions from base learners will be cached to avoid recomputation (warning: not checked against the molecules provided
            -> if the molecules change, you have to empty the cache or provide a new cache directory).
        resolve_inconsistencies (bool): Whether to resolve inconsistencies in the aggregated predictions.
        decision_threshold (float): Threshold for class decisions based on net score. Default is 0.

    Returns:
        dict: A dictionary containing the final predictions and optionally the smoothed predictions.
    """

    # Step 1: Get predictions from base learners on test data
    test_predictions = {}
    for model_name, model in base_learners.items():
        if prediction_cache_dir is None:
            test_predictions[model_name] = model.predict_list(molecules)
        else:
            cache_path = os.path.join(
                prediction_cache_dir, f"{model_name}_test_predictions.pt"
            )
            if os.path.exists(cache_path):
                test_predictions[model_name] = torch.load(
                    cache_path, weights_only=False
                )
            else:
                test_predictions[model_name] = model.predict_list(molecules)
                torch.save(test_predictions[model_name], cache_path)

    test_predictions, predicted_classes = collect_base_learner_predictions(
        test_predictions
    )

    # Step 2: Get aggregated predictions from the ensemble model
    aggregated_predictions = ensemble_model.predict(test_predictions)
    # net_score, has_valid_predictions, intermediate_results_dict

    # Step 3: Optionally resolve inconsistencies in the aggregated predictions
    if resolve_inconsistencies:
        chebi_graph = load_chebi_graph()
        disjoint_files = get_disjoint_files()
        smoother = ScoreBasedPredictionSmoother(
            chebi_graph=chebi_graph, label_names=None, disjoint_files=disjoint_files
        )
        aggregated_predictions = apply_inconsistency_resolution(
            smoother, predicted_classes, aggregated_predictions
        )

    class_decisions = (
        aggregated_predictions["net_score"] > decision_threshold
    ) & aggregated_predictions[
        "has_valid_predictions"
    ]  # Shape: (num_smiles, num_classes)

    complete_failure = torch.all(
        ~aggregated_predictions["has_valid_predictions"], dim=1
    )
    aggregated_predictions["class_decisions"] = class_decisions
    aggregated_predictions["complete_failure"] = complete_failure

    aggregated_predictions["predicted_classes"] = predicted_classes

    return aggregated_predictions
