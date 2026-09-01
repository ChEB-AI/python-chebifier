# Get end-to-end predictions (from SMILES / molecule list via base learners + ensemble + inconsistency resolution to ChEBI classes)


import os
from typing import Optional

import numpy as np
import torch
from rdkit import Chem

from chebifier.ensemble.base_ensemble import BaseEnsemble
from chebifier.inconsistency_resolution import NEUTRAL, get_smoother_class
from chebifier.prediction_models.base_predictor import BasePredictor
from chebifier.utils import get_disjoint_files, load_chebi_graph


def apply_inconsistency_resolution(
    smoother,
    class_names,
    aggregated_predictions,
    batch_size: int = 16,
    decision_threshold: float = NEUTRAL,
):
    """Resolve inconsistencies in batches - the smoother materialises a
    (batch_size, n_classes, n_classes) tensor, which does not fit into memory for a whole dataset split.
    """
    if getattr(smoother, "label_names", None) != class_names:
        smoother.set_label_names(class_names)
    net_score = aggregated_predictions["net_score"]
    valid = aggregated_predictions.get("has_valid_predictions")
    attribution = aggregated_predictions.get("attribution")
    batches = [
        smoother(
            net_score[start : start + batch_size],
            None if valid is None else valid[start : start + batch_size],
            **(
                {}
                if attribution is None
                else {"attribution": attribution[start : start + batch_size]}
            ),
        )
        for start in range(0, net_score.shape[0], batch_size)
    ]
    if attribution is not None:
        aggregated_predictions["attribution"] = torch.cat([a for _, a in batches])
        batches = [scores for scores, _ in batches]
    aggregated_predictions["net_score"] = torch.cat(batches)
    if valid is not None:
        aggregated_predictions["has_valid_predictions"] = valid | (
            aggregated_predictions["net_score"] > decision_threshold
        )
    return aggregated_predictions


_SMOOTHER_CACHE = {}


def get_smoother(inconsistency_resolution, chebi_graph, disjoint_files, params):
    """Build a smoother, reusing a previously built one where possible.

    Building a smoother parses the disjointness files and (once the label names are known)
    computes the transitive closure of the hierarchy, which is far more expensive than the
    resolution itself - and none of it depends on the molecules being predicted.
    """
    key = (
        inconsistency_resolution,
        id(chebi_graph),
        tuple(str(file) for file in disjoint_files),
        repr(sorted(params.items())),
    )
    if key not in _SMOOTHER_CACHE:
        # the graph is kept alive alongside the smoother so that its id stays unique
        _SMOOTHER_CACHE[key] = (
            chebi_graph,
            get_smoother_class(inconsistency_resolution)(
                chebi_graph=chebi_graph,
                label_names=None,
                disjoint_files=disjoint_files,
                **params,
            ),
        )
    return _SMOOTHER_CACHE[key][1]


def base_learner_cache_path(
    prediction_cache_dir: str, model_name: str, split: str
) -> str:
    return os.path.join(prediction_cache_dir, f"{model_name}_{split}_predictions.npz")


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


def get_base_learner_predictions(
    base_learners: dict[str, Optional[BasePredictor]],
    molecules: list[str | Chem.Mol],
    prediction_cache_dir: Optional[str] = None,
    split: str = "test",
) -> dict[str, tuple[list[str], np.ndarray]]:
    """Get dense predictions from the base learners, using the cache where available.

    A base learner may be None if its predictions are known to be cached (see
    cli.build_base_learners), which avoids loading model checkpoints that are never used.
    """
    predictions = {}
    for model_name, model in base_learners.items():
        cache_path = (
            None
            if prediction_cache_dir is None
            else base_learner_cache_path(prediction_cache_dir, model_name, split)
        )
        if cache_path is not None and os.path.exists(cache_path):
            predictions[model_name] = load_dense_predictions(cache_path)
            continue
        if model is None:
            raise ValueError(
                f"Base learner '{model_name}' was not instantiated, but its predictions are "
                f"missing from the cache ({cache_path})."
            )
        predictions[model_name] = model.predict_dense(molecules)
        if cache_path is not None:
            save_dense_predictions(cache_path, *predictions[model_name])
    return predictions


def aggregate_predictions(
    base_learners: dict[str, Optional[BasePredictor]],
    ensemble_model: BaseEnsemble,
    molecules: list[str | Chem.Mol],
    prediction_cache_dir: Optional[str] = None,
    classes: Optional[list[str]] = None,
    split: str = "test",
    attribution: bool = False,
) -> tuple[dict, list[str]]:
    """Get base learner predictions and aggregate them with the ensemble model.

    The result does not depend on the inconsistency resolution method, so it can be reused for
    several resolution variants (see resolve_and_decide).
    """
    test_predictions = get_base_learner_predictions(
        base_learners, molecules, prediction_cache_dir=prediction_cache_dir, split=split
    )
    test_predictions, predicted_classes = collect_base_learner_predictions(
        test_predictions, classes=classes
    )
    aggregated_predictions = ensemble_model.predict(
        test_predictions, molecules, **({"attribution": True} if attribution else {})
    )
    if attribution:
        aggregated_predictions["base_learner_predictions"] = test_predictions
    # net_score, has_valid_predictions, intermediate_results_dict
    return aggregated_predictions, predicted_classes


def resolve_and_decide(
    aggregated_predictions: dict,
    predicted_classes: list[str],
    inconsistency_resolution: Optional[str] = "score-based",
    inconsistency_resolution_params: Optional[dict] = None,
    decision_threshold: float = 0.5,
    chebi_graph=None,
    chebi_graph_file: Optional[str] = None,
    disjoint_files=None,
) -> dict:
    """Resolve inconsistencies in aggregated predictions and turn them into class decisions.

    Net scores are probabilities, so `decision_threshold` defaults to the neutral point. Ensembles
    that tune their own operating point report it as `BaseEnsemble.decision_threshold`.

    `aggregated_predictions` is not modified, so the same aggregation can be passed to several
    resolution variants. Pass `inconsistency_resolution=None` or "none" to skip the resolution,
    and chebi_graph / disjoint_files to avoid reloading them for every variant. `chebi_graph_file`
    loads the hierarchy from a local file instead of Hugging Face.
    """
    aggregated_predictions = dict(aggregated_predictions)
    if inconsistency_resolution not in (None, "none"):
        if chebi_graph is None:
            chebi_graph = load_chebi_graph(chebi_graph_file)
        if disjoint_files is None:
            disjoint_files = get_disjoint_files()
        params = inconsistency_resolution_params or {}
        smoother = get_smoother(
            inconsistency_resolution, chebi_graph, disjoint_files, params
        )
        if "threshold" not in params and hasattr(smoother, "threshold"):
            smoother.threshold = decision_threshold
        aggregated_predictions = apply_inconsistency_resolution(
            smoother,
            predicted_classes,
            aggregated_predictions,
            decision_threshold=decision_threshold,
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


def predict(
    base_learners: dict[str, BasePredictor],
    ensemble_model: BaseEnsemble,
    molecules: list[str | Chem.Mol],
    prediction_cache_dir: Optional[str] = None,
    resolve_inconsistencies: bool = True,
    inconsistency_resolution: str = "score-based",
    inconsistency_resolution_params: Optional[dict] = None,
    decision_threshold: Optional[float] = None,
    classes: Optional[list[str]] = None,
    split: str = "test",
    attribution: bool = False,
    chebi_graph_file: Optional[str] = None,
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
        inconsistency_resolution (str): Which resolution method to use, see SMOOTHER_NAMES.
        decision_threshold (Optional[float]): Threshold for class decisions based on net score.
            If None (the default), the threshold reported by the ensemble is used.
        classes (Optional[list[str]]): Column space to map the base learner predictions onto, see
            collect_base_learner_predictions. If None (the default), the union of all classes is used.
        split (str): Name of the dataset split, used to separate cached base learner predictions of
            different splits within the same cache directory.
        attribution (bool): Also report, per (molecule, class), the share of the decision each base
            learner is responsible for (summing to 1 over the base learners), together with the raw
            base learner predictions it was derived from (`base_learner_predictions`, one
            (num_molecules, num_classes) tensor per model). Only supported by the voting ensembles
            and the score-based inconsistency resolution.
        chebi_graph_file (Optional[str]): Local ChEBI graph (pickled networkx graph) the
            inconsistency resolution runs on. If None (the default), it is downloaded from
            Hugging Face.

    Returns:
        dict: A dictionary containing the final predictions and optionally the smoothed predictions.
    """
    aggregated_predictions, predicted_classes = aggregate_predictions(
        base_learners,
        ensemble_model,
        molecules,
        prediction_cache_dir=prediction_cache_dir,
        classes=classes,
        split=split,
        attribution=attribution,
    )
    return resolve_and_decide(
        aggregated_predictions,
        predicted_classes,
        inconsistency_resolution=(
            inconsistency_resolution if resolve_inconsistencies else "none"
        ),
        inconsistency_resolution_params=inconsistency_resolution_params,
        chebi_graph_file=chebi_graph_file,
        decision_threshold=(
            ensemble_model.decision_threshold
            if decision_threshold is None
            else decision_threshold
        ),
    )
