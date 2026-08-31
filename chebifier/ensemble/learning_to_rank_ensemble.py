import json
from pathlib import Path

import numpy as np
import torch

from chebifier.ensemble.level1 import (
    N_FOLDS,
    RANDOM_SEED,
    candidate_pairs,
    class_statistics,
    cv_folds,
    dense_from_pairs,
    fit_platt,
    holdout_split,
    noncandidate_probability,
    pair_scorer,
    save_hyperparameter_results,
    select_candidates,
    stack_predictions,
    threshold_array,
)
from chebifier.ensemble.voting_ensemble import VotingEnsemble

CANDIDATE_K_GRID = (30, 50, 70)
EARLY_STOPPING_ROUNDS = 30
LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10, 30],
    "label_gain": [0, 1],
    "lambdarank_truncation_level": 60,
    "max_depth": 4,
    "num_leaves": 15,
    "learning_rate": 0.05,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": RANDOM_SEED,
}


def build_rows(scores, k, labels=None):
    mask = select_candidates(scores, k)
    molecule, class_index, group = candidate_pairs(mask)
    block = scores[molecule, class_index]
    valid = ~np.isnan(block)
    n_valid = valid.sum(axis=1).astype(np.float32)
    covered = n_valid > 0
    denominator = np.maximum(n_valid, 1)
    mean = np.where(valid, block, 0.0).sum(axis=1) / denominator
    variance = (
        np.where(valid, (block - mean[:, None]) ** 2, 0.0).sum(axis=1) / denominator
    )
    maximum = np.where(valid, block, -np.inf).max(axis=1)
    aggregates = np.column_stack(
        [
            n_valid,
            np.where(covered, maximum, np.nan),
            np.where(covered, mean, np.nan),
            np.where(covered, np.sqrt(variance), np.nan),
        ]
    )
    rows = {
        "X": np.column_stack([block, aggregates]).astype(np.float32),
        "molecule": molecule,
        "class_index": class_index,
        "group": group,
        "n_molecules": scores.shape[0],
    }
    if labels is not None:
        rows["y"] = labels[molecule, class_index].astype(np.int8)
    return rows


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-x)))


def design_matrix(rows, class_stats=None):
    if class_stats is None:
        return rows["X"]
    return np.column_stack([rows["X"], class_stats[rows["class_index"]]])


class LearningToRankEnsemble(VotingEnsemble):

    def __init__(
        self,
        ensemble_dir: str,
        candidate_k=None,
        candidate_k_grid=CANDIDATE_K_GRID,
        n_estimators: int = 500,
        class_stats: bool = True,
        **kwargs,
    ):
        super().__init__(ensemble_dir)
        self.candidate_k = candidate_k
        self.candidate_k_grid = tuple(candidate_k_grid)
        self.n_estimators = n_estimators
        self.class_stats = bool(class_stats)
        self._booster = None
        self._metadata = None
        self._class_stats = None

    @property
    def _model_path(self):
        return Path(self.ensemble_dir) / "ltr_ranker.txt"

    @property
    def _metadata_path(self):
        return Path(self.ensemble_dir) / "ltr_metadata.json"

    @property
    def _class_stats_path(self):
        return Path(self.ensemble_dir) / "ltr_class_stats.npy"

    def calibrate(self, validation_predictions, validation_data, validation_labels):
        if self.class_stats:
            # fits the per-model prediction thresholds the class-wise F1 scores are based on
            super().calibrate(
                validation_predictions, validation_data, validation_labels
            )
        else:
            print(
                f"Calibrating {self.ensemble_name} with {len(validation_predictions)} base learners..."
            )
        scores, model_names = stack_predictions(validation_predictions)
        labels = np.asarray(validation_labels, dtype=bool)
        thresholds = (
            threshold_array(self._load_prediction_thresholds(), model_names)
            if self.class_stats
            else None
        )

        candidate_k = self.candidate_k
        best = None
        if candidate_k is None:
            candidate_k, best = self._optimize_candidate_k(scores, labels, thresholds)

        rows = build_rows(scores, candidate_k, labels)
        train_mask, dev_mask = holdout_split(np.ones(scores.shape[0], dtype=bool))
        booster, tau, dev_macro_f1, _, dev_scores = self._fit(
            rows, labels, train_mask, dev_mask, scores, thresholds
        )
        platt_a, platt_b = fit_platt(dev_scores, rows["y"][dev_mask[rows["molecule"]]])
        floor = noncandidate_probability(
            scores.shape[0] * scores.shape[1] - rows["molecule"].size,
            int(labels.sum()) - int(rows["y"].sum()),
        )

        booster.save_model(str(self._model_path), num_iteration=booster.best_iteration)
        if self.class_stats:
            # the ranker is trained on class statistics of the training molecules only, but
            # predicts with the statistics of the whole validation split - the same out-of-fold
            # arrangement the weighted majority vote ensemble uses for its class-wise F1 weights
            self._class_stats = class_statistics(scores, labels, thresholds)
            np.save(self._class_stats_path, self._class_stats)
        metadata = {
            "model_names": model_names,
            "candidate_k": int(candidate_k),
            "class_stats": self.class_stats,
            "tau": float(tau),
            "platt_a": platt_a,
            "platt_b": platt_b,
            "tau_probability": _sigmoid(platt_a * float(tau) + platt_b),
            "floor": floor,
            "n_classes": int(scores.shape[1]),
            "best_iteration": int(booster.best_iteration),
            "dev_macro_f1": float(dev_macro_f1),
            "params": {key: value for key, value in LGB_PARAMS.items()},
        }
        if best is not None:
            metadata["cross_validated_macro_f1"] = best["mean_macro_f1"]
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        self._booster, self._metadata = booster, metadata
        print(
            f"Saved ranker to {self._model_path} (candidate_k={candidate_k}, tau={tau:.4f}, "
            f"held-out macro-f1: {dev_macro_f1:.4f}). Calibrated to probabilities with "
            f"a={platt_a:.4f}, b={platt_b:.4f} (decision threshold "
            f"{metadata['tau_probability']:.4f}); non-candidate pairs scored at {floor:.2e}."
        )

    def _fit(self, rows, labels, train_mask, dev_mask, scores=None, thresholds=None):
        import lightgbm as lgb

        class_stats = None
        if self.class_stats:
            # estimated on the training molecules only, otherwise the features would carry the
            # labels of the molecules the ranker is scored on
            class_stats = class_statistics(scores, labels, thresholds, train_mask)
        X = design_matrix(rows, class_stats)
        train_rows = train_mask[rows["molecule"]]
        dev_rows = dev_mask[rows["molecule"]]
        train_set = lgb.Dataset(
            X[train_rows],
            label=rows["y"][train_rows],
            group=rows["group"][train_mask],
            free_raw_data=False,
        )
        dev_set = lgb.Dataset(
            X[dev_rows],
            label=rows["y"][dev_rows],
            group=rows["group"][dev_mask],
            reference=train_set,
            free_raw_data=False,
        )
        booster = lgb.train(
            LGB_PARAMS,
            train_set,
            num_boost_round=self.n_estimators,
            valid_sets=[dev_set],
            valid_names=["dev"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        dev_scores = booster.predict(
            X[dev_rows], num_iteration=booster.best_iteration
        ).astype(np.float32)
        scorer, _ = pair_scorer(labels, rows["molecule"], rows["class_index"], dev_mask)
        tau, dev_macro_f1 = scorer.tune(dev_scores)
        return booster, tau, dev_macro_f1, class_stats, dev_scores

    def _score_candidate_k(self, rows, labels, folds, scores=None, thresholds=None):
        fold_scores = []
        for fold, test_idx in enumerate(folds):
            test_mask = np.zeros(rows["n_molecules"], dtype=bool)
            test_mask[test_idx] = True
            train_mask, inner_dev_mask = holdout_split(
                ~test_mask, seed=RANDOM_SEED + fold
            )
            booster, tau, _, class_stats, _ = self._fit(
                rows, labels, train_mask, inner_dev_mask, scores, thresholds
            )
            test_rows = test_mask[rows["molecule"]]
            net = booster.predict(
                design_matrix(rows, class_stats)[test_rows],
                num_iteration=booster.best_iteration,
            ).astype(np.float32)
            scorer, _ = pair_scorer(
                labels, rows["molecule"], rows["class_index"], test_mask
            )
            fold_scores.append(scorer.macro_f1(net, tau))
        return fold_scores

    def _optimize_candidate_k(self, scores, labels, thresholds=None):
        print(
            f"Optimizing candidate_k with {N_FOLDS}-fold cross-validation on the validation set..."
        )
        folds = cv_folds(scores.shape[0])
        results = []
        for candidate_k in self.candidate_k_grid:
            rows = build_rows(scores, candidate_k, labels)
            recall = float(
                labels[rows["molecule"], rows["class_index"]].sum()
                / max(int(labels.sum()), 1)
            )
            fold_scores = self._score_candidate_k(
                rows, labels, folds, scores, thresholds
            )
            mean_score = float(np.mean(fold_scores))
            results.append(
                {
                    "candidate_k": candidate_k,
                    "candidate_recall": recall,
                    "mean_macro_f1": mean_score,
                    "std_macro_f1": float(np.std(fold_scores)),
                    **{f"fold_{i}_macro_f1": s for i, s in enumerate(fold_scores)},
                }
            )
            print(
                f"candidate_k={candidate_k} (candidate recall {recall:.4f}): macro-f1 {mean_score:.4f}"
            )
        best = max(results, key=lambda result: result["mean_macro_f1"])
        save_hyperparameter_results(
            self.ensemble_dir,
            results,
            {
                "candidate_k": best["candidate_k"],
                "mean_macro_f1": best["mean_macro_f1"],
            },
        )
        return best["candidate_k"], best

    def _load(self):
        if self._booster is not None:
            return
        import lightgbm as lgb

        if not self._metadata_path.exists():
            raise FileNotFoundError(
                f"No calibrated ranker found in ensemble directory: {self.ensemble_dir}. "
                "Please calibrate the ensemble first."
            )
        with open(self._metadata_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)
        self._booster = lgb.Booster(model_file=str(self._model_path))
        self.class_stats = self._metadata.get("class_stats", False)
        if self.class_stats:
            self._class_stats = np.load(self._class_stats_path)

    @property
    def decision_threshold(self):
        """Probability a candidate has to exceed, the operating point macro-F1 peaks at."""
        self._load()
        return float(self._metadata["tau_probability"])

    def predict(self, test_predictions, molecules=None):
        self._load()
        scores, _ = stack_predictions(test_predictions, self._metadata["model_names"])
        rows = build_rows(scores, self._metadata["candidate_k"])
        raw = self._booster.predict(design_matrix(rows, self._class_stats)).astype(
            np.float32
        )
        # the ranker score is not a probability, so it goes through the calibration fitted during
        # `calibrate`; the decision threshold is left to the caller so that inconsistency
        # resolution sees the probabilities themselves
        net = 1.0 / (
            1.0 + np.exp(-(self._metadata["platt_a"] * raw + self._metadata["platt_b"]))
        )
        dense = dense_from_pairs(
            rows["molecule"],
            rows["class_index"],
            net.astype(np.float32),
            scores.shape[:2],
            floor=self._metadata["floor"],
        )
        return {
            "net_score": torch.from_numpy(dense),
            "has_valid_predictions": torch.from_numpy((~np.isnan(scores)).any(axis=2)),
        }
