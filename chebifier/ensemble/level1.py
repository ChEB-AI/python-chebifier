from pathlib import Path

import numpy as np
import pandas as pd

N_FOLDS = 5
RANDOM_SEED = 42
POSITIVE_THRESHOLD = 0.5
THRESHOLD_STEPS = 80


def stack_predictions(predictions, model_names=None, dtype=np.float32):
    if model_names is None:
        model_names = list(predictions)
    missing = [name for name in model_names if name not in predictions]
    if missing:
        raise ValueError(
            "Predictions are missing for models the ensemble was calibrated on: "
            + ", ".join(missing)
        )
    stacked = np.stack(
        [np.asarray(predictions[name], dtype=dtype) for name in model_names], axis=2
    )
    return stacked, list(model_names)


def rescale_to_threshold(scores, thresholds):
    scores = np.asarray(scores, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    return np.where(
        scores < thresholds,
        POSITIVE_THRESHOLD * scores / thresholds,
        POSITIVE_THRESHOLD
        + POSITIVE_THRESHOLD * (scores - thresholds) / (1 - thresholds),
    )


def threshold_array(thresholds, model_names):
    missing = [name for name in model_names if name not in thresholds]
    if missing:
        raise ValueError("Prediction thresholds are missing for: " + ", ".join(missing))
    return np.array([thresholds[name] for name in model_names], dtype=np.float32)


def coverage_of(scores):
    return ~np.isnan(scores).all(axis=0)


def class_statistics(scores, labels, thresholds, molecule_mask=None, chunk_size=512):
    """Per-class statistics of the base learners: one F1 score per (class, model), followed by the
    prevalence and the number of positives of the class. NaN scores count as negative predictions,
    matching the class-wise F1 scores of the weighted majority vote ensemble."""
    rows = (
        np.arange(labels.shape[0])
        if molecule_mask is None
        else np.flatnonzero(molecule_mask)
    )
    counts = np.zeros((3, scores.shape[1], scores.shape[2]), dtype=np.int64)
    for start in range(0, len(rows), chunk_size):
        block = rows[start : start + chunk_size]
        predicted = scores[block] > thresholds
        truth = labels[block][:, :, None]
        counts[0] += (predicted & truth).sum(axis=0)
        counts[1] += (predicted & ~truth).sum(axis=0)
        counts[2] += (~predicted & truth).sum(axis=0)
    tp, fp, fn = counts
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    positives = labels[rows].sum(axis=0)
    return np.column_stack([f1, positives / max(len(rows), 1), positives]).astype(
        np.float32
    )


def select_candidates(scores, k):
    n_molecules, n_classes, n_models = scores.shape
    k = min(k, n_classes)
    mask = np.zeros((n_molecules, n_classes), dtype=bool)
    for model_idx in range(n_models):
        column = scores[:, :, model_idx]
        column = np.where(np.isnan(column), -np.inf, column)
        top = np.argpartition(-column, k - 1, axis=1)[:, :k]
        np.put_along_axis(mask, top, True, axis=1)
    return mask


def candidate_pairs(mask):
    molecule, class_index = np.nonzero(mask)
    return (
        molecule.astype(np.int64),
        class_index.astype(np.int64),
        mask.sum(axis=1).astype(np.int64),
    )


def cv_folds(n_molecules, n_folds=N_FOLDS, seed=0):
    permutation = np.random.default_rng(seed).permutation(n_molecules)
    return [permutation[fold::n_folds] for fold in range(n_folds)]


def holdout_split(molecule_mask, dev_fraction=0.2, seed=RANDOM_SEED):
    rows = np.flatnonzero(molecule_mask)
    n_dev = max(int(len(rows) * dev_fraction), 1)
    dev = np.zeros(len(molecule_mask), dtype=bool)
    dev[np.random.default_rng(seed).choice(rows, size=n_dev, replace=False)] = True
    return molecule_mask & ~dev, dev


class PairScorer:
    def __init__(self, labels, molecule, class_index):
        self.n_molecules, self.n_classes = labels.shape
        self.molecule = molecule
        self.class_index = class_index
        self.y = labels[molecule, class_index]
        self.class_positives = labels.sum(axis=0).astype(np.int64)

    def _class_counts(self, predicted):
        tp = np.bincount(
            self.class_index[predicted & self.y], minlength=self.n_classes
        ).astype(np.int64)
        positives = np.bincount(
            self.class_index[predicted], minlength=self.n_classes
        ).astype(np.int64)
        return tp, positives - tp

    def macro_f1(self, net, tau):
        tp, fp = self._class_counts(net > tau)
        fn = self.class_positives - tp
        denominator = 2 * tp + fp + fn
        return float(
            np.where(denominator > 0, 2 * tp / np.maximum(denominator, 1), 0.0).mean()
        )

    def micro_f1(self, net, tau):
        tp, fp = self._class_counts(net > tau)
        tp, fp = int(tp.sum()), int(fp.sum())
        fn = int(self.class_positives.sum()) - tp
        denominator = 2 * tp + fp + fn
        return 2 * tp / denominator if denominator > 0 else 0.0

    def tune(self, net, metric=None, n_steps=THRESHOLD_STEPS):
        metric = self.macro_f1 if metric is None else metric
        grid = np.quantile(net, np.linspace(0.02, 0.9995, n_steps))
        best_tau, best_score = float(grid[0]), -1.0
        for tau in grid:
            score = metric(net, tau)
            if score > best_score:
                best_tau, best_score = float(tau), score
        return best_tau, best_score


def pair_scorer(labels, molecule, class_index, molecule_mask):
    keep = molecule_mask[molecule]
    rows = np.flatnonzero(molecule_mask)
    remap = np.full(len(molecule_mask), -1, dtype=np.int64)
    remap[rows] = np.arange(len(rows))
    return PairScorer(labels[rows], remap[molecule[keep]], class_index[keep]), keep


def dense_from_pairs(molecule, class_index, net, shape, floor=None):
    if floor is None:
        floor = min(float(net.min()) - 1.0, -1.0) if net.size else -1.0
    dense = np.full(shape, floor, dtype=np.float32)
    dense[molecule, class_index] = net
    return dense


def fit_platt(scores, labels):
    """Fit `P(positive) = sigmoid(a * score + b)` so that `a * score + b` is calibrated log-odds.

    A lambdarank score is only trained to order pairs within a group, so its scale carries no
    probabilistic meaning. Inconsistency resolution needs one: it compares scores across classes and
    maps them through `sigmoid(k * score)`. Being a monotone map, this leaves the ensemble's own
    thresholded decisions untouched - what it buys is a scale on which those downstream comparisons
    are meaningful.
    """
    from sklearn.linear_model import LogisticRegression

    scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    platt = LogisticRegression(C=1e6).fit(scores, np.asarray(labels))
    slope, intercept = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    if slope <= 0:
        raise ValueError(
            f"Platt calibration found a non-positive slope ({slope:.4g}), meaning the ranker scores "
            "anti-correlate with the labels. Refusing to calibrate - check the base learner "
            "predictions and the ranker training."
        )
    return slope, intercept


def noncandidate_log_odds(n_noncandidate, n_missed_positives):
    """Log-odds that a pair outside the candidate set is nevertheless a positive.

    Candidate selection keeps only the top-k classes per model, so the pairs it drops are not
    "unknown" - they are overwhelmingly true negatives, and how overwhelmingly is measurable on the
    validation set. This turns the fill value for those pairs from an arbitrary floor into a
    calibrated score that can be compared against the candidates. The Jeffreys-style pseudo-count
    keeps the result finite when no positive is missed at all.
    """
    p = (n_missed_positives + 0.5) / (n_noncandidate + 1.0)
    return float(np.log(p / (1.0 - p)))


def save_hyperparameter_results(ensemble_dir, results, best):
    results_path = Path(ensemble_dir) / "hyperparameter_search.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    best_path = Path(ensemble_dir) / "best_hyperparameters.csv"
    pd.DataFrame([best]).to_csv(best_path, index=False)
    print(
        f"Saved hyperparameter search results to {results_path}. Recommended parameters "
        f"(saved to {best_path}): "
        + ", ".join(f"{key}={value}" for key, value in best.items())
    )
