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


def apply_inconsistency_resolution(
    smoother, class_names, aggregated_predictions, batch_size: int = 16
):
    """Resolve inconsistencies in batches - the smoother materialises a
    (batch_size, n_classes, n_classes) tensor, which does not fit into memory for a whole dataset split.
    """
    smoother.set_label_names(class_names)
    net_score = aggregated_predictions["net_score"]
    aggregated_predictions["net_score"] = torch.cat(
        [
            smoother(net_score[start : start + batch_size])
            for start in range(0, net_score.shape[0], batch_size)
        ]
    )
    return aggregated_predictions


def save_dense_predictions(path: str, classes: list[str], scores: np.ndarray) -> None:
    np.savez_compressed(path, classes=np.array(classes), scores=scores)


def load_dense_predictions(path: str) -> tuple[list[str], np.ndarray]:
    with np.load(path) as data:
        return [str(cls) for cls in data["classes"]], data["scores"]


def collect_base_learner_predictions(
    predictions: dict[str, tuple[list[str], np.ndarray]],
    classes: Optional[list[str]] = None,
) -> (dict[str, torch.Tensor], list[str]):
    """
    Collect predictions from base learners into a single dictionary.

    Args:
        predictions (dict): A dictionary where keys are model names and values are
            (class labels, score matrix) pairs as returned by BasePredictor.predict_dense.
            Assumes all score matrices have the same number of rows.
        classes (Optional[list]): Column space to map the predictions onto. If None (the default), the
            union of all classes reported by the base learners is used. Pass an explicit list to align
            the predictions with a fixed label set (e.g. the labels of an evaluation dataset) - base
            learners may be trained on different label sets, so the union does not necessarily match the
            labels you want to compare against. Classes predicted by a base learner but missing from
            `classes` are dropped, classes that no base learner covers stay NaN.

    Returns:
        dict: A dictionary where keys are model names and values are tensors of predictions with shape (num_samples, num_classes).
            If a prediction is missing, it will be NaN.
        ensemble_classes (list): A list of class labels that are present in the predictions.
    """
    print(f"Collecting base learner predictions from {len(predictions)} models...")
    n_samples = -1
    for model_name, (_, scores) in predictions.items():
        if n_samples == -1:
            n_samples = scores.shape[0]
        else:
            assert (
                n_samples == scores.shape[0]
            ), f"All prediction matrices must have the same length. Model {model_name} has {scores.shape[0]} predictions, expected {n_samples}."

    if classes is None:
        ensemble_classes = sorted(
            {cls for model_classes, _ in predictions.values() for cls in model_classes}
        )
    else:
        ensemble_classes = list(classes)
    cls_to_idx = {cls: idx for idx, cls in enumerate(ensemble_classes)}

    collected_predictions = {}
    for model_name, (model_classes, scores) in predictions.items():
        if model_classes == ensemble_classes:
            collected_predictions[model_name] = torch.from_numpy(
                scores.astype(np.float32)
            )
            continue
        shared = [
            (source, cls_to_idx[cls])
            for source, cls in enumerate(model_classes)
            if cls in cls_to_idx
        ]
        mapped = np.full((n_samples, len(ensemble_classes)), np.nan, dtype=np.float32)
        if shared:
            source_idx = np.fromiter(
                (source for source, _ in shared), dtype=np.intp, count=len(shared)
            )
            target_idx = np.fromiter(
                (target for _, target in shared), dtype=np.intp, count=len(shared)
            )
            mapped[:, target_idx] = scores[:, source_idx]
        collected_predictions[model_name] = torch.from_numpy(mapped)

    return collected_predictions, ensemble_classes


def predict(
    base_learners: dict[str, BasePredictor],
    ensemble_model: BaseEnsemble,
    molecules: list[str | Chem.Mol],
    prediction_cache_dir: Optional[str] = None,
    resolve_inconsistencies: bool = True,
    decision_threshold: float = 0,
    classes: Optional[list[str]] = None,
    split: str = "test",
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
        classes (Optional[list[str]]): Column space to map the base learner predictions onto, see
            collect_base_learner_predictions. If None (the default), the union of all classes is used.
        split (str): Name of the dataset split, used to separate cached base learner predictions of
            different splits within the same cache directory.

    Returns:
        dict: A dictionary containing the final predictions and optionally the smoothed predictions.
    """

    # Step 1: Get predictions from base learners on test data
    test_predictions = {}
    for model_name, model in base_learners.items():
        if prediction_cache_dir is None:
            test_predictions[model_name] = model.predict_dense(molecules)
        else:
            cache_path = os.path.join(
                prediction_cache_dir, f"{model_name}_{split}_predictions.npz"
            )
            if os.path.exists(cache_path):
                test_predictions[model_name] = load_dense_predictions(cache_path)
            else:
                test_predictions[model_name] = model.predict_dense(molecules)
                save_dense_predictions(cache_path, *test_predictions[model_name])

    test_predictions, predicted_classes = collect_base_learner_predictions(
        test_predictions, classes=classes
    )

    # Step 2: Get aggregated predictions from the ensemble model
    aggregated_predictions = ensemble_model.predict(test_predictions, molecules)
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
