import json
import pickle
from pathlib import Path

import numpy as np
import torch

from chebifier.ensemble.level1 import (
    N_FOLDS,
    POSITIVE_THRESHOLD,
    RANDOM_SEED,
    candidate_pairs,
    coverage_of,
    cv_folds,
    dense_from_pairs,
    holdout_split,
    pair_scorer,
    rescale_to_threshold,
    save_hyperparameter_results,
    select_candidates,
    stack_predictions,
    threshold_array,
)
from chebifier.ensemble.voting_ensemble import VotingEnsemble

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
# without chirality, ECFP4 gives stereoisomers - which are distinct ChEBI classes - the same
# fingerprint, so the region of competence can be anchored on a molecule of a different class
MORGAN_CHIRALITY = True
REGION_SIZE_GRID = (1, 7)
PROFILE_SIZE_GRID = (5, 9, 15)
VOTE_GRID = ("plain", "confidence")
CHUNK_SIZE = 512
MAX_META_SAMPLES = 2_000_000
MLP_HIDDEN_LAYERS = (64, 32)


def fingerprints(
    molecules, radius=MORGAN_RADIUS, n_bits=MORGAN_BITS, chirality=MORGAN_CHIRALITY
):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, includeChirality=chirality
    )
    out = np.zeros((len(molecules), n_bits), dtype=np.uint8)
    ok = np.zeros(len(molecules), dtype=bool)
    for i, molecule in enumerate(molecules):
        if isinstance(molecule, str):
            molecule = Chem.MolFromSmiles(molecule)
        if molecule is None:
            continue
        out[i] = generator.GetFingerprintAsNumPy(molecule)
        ok[i] = True
    return out, ok


def tanimoto_knn(
    query, reference, reference_ids, k, self_columns=None, chunk_size=1024
):
    query = query.astype(np.float32, copy=False)
    reference = reference.astype(np.float32, copy=False)
    query_bits = query.sum(axis=1)
    reference_bits = reference.sum(axis=1)
    k = min(k, reference.shape[0])
    out = np.zeros((query.shape[0], k), dtype=np.int32)
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        intersection = query[start:end] @ reference.T
        union = query_bits[start:end, None] + reference_bits[None, :] - intersection
        block = intersection / np.maximum(union, 1e-6)
        if self_columns is not None:
            rows = np.arange(end - start)
            columns = self_columns[start:end]
            present = columns >= 0
            block[rows[present], columns[present]] = -1.0
        take = min(4 * k, block.shape[1] - 1)
        near = np.argpartition(-block, take, axis=1)[:, : take + 1]
        near_similarity = np.take_along_axis(block, near, axis=1)
        near_ids = reference_ids[near]
        order = np.lexsort((near_ids, -near_similarity), axis=1)[:, :k]
        out[start:end] = np.take_along_axis(near_ids, order, axis=1)
    return out


def region_of_competence(
    query_fingerprints,
    query_ok,
    reference_fingerprints,
    reference_ids,
    k,
    self_columns=None,
):
    neighbours = tanimoto_knn(
        query_fingerprints, reference_fingerprints, reference_ids, k, self_columns
    )
    neighbours[~query_ok] = -1
    return neighbours


class Candidates:
    def __init__(self, scores, k, thresholds):
        mask = select_candidates(scores, k)
        self.molecule, self.class_index, group = candidate_pairs(mask)
        self.profiles = rescale_to_threshold(
            scores[self.molecule, self.class_index], thresholds
        )
        self.n_molecules = scores.shape[0]
        self.pointer = np.zeros(self.n_molecules + 1, dtype=np.int64)
        np.cumsum(group, out=self.pointer[1:])

    def slice_of(self, start, end):
        return int(self.pointer[start]), int(self.pointer[end])

    def iter_chunks(self, chunk_size):
        for start in range(0, self.n_molecules, chunk_size):
            end = min(start + chunk_size, self.n_molecules)
            yield start, end, self.slice_of(start, end)


class Dsel:
    def __init__(self, scores, labels, thresholds):
        self.scores = scores
        self.labels = labels
        self.thresholds = thresholds

    def gather(self, rows, class_index):
        n_pairs, n_neighbours = rows.shape
        flat_rows = np.maximum(rows, 0).ravel()
        flat_classes = np.repeat(class_index, n_neighbours)
        scores = rescale_to_threshold(
            self.scores[flat_rows, flat_classes], self.thresholds
        )
        labels = self.labels[flat_rows, flat_classes]
        return (
            scores.reshape(n_pairs, n_neighbours, -1),
            labels.reshape(n_pairs, n_neighbours),
        )


def output_profile_knn(
    query, dsel, dsel_mask, coverage, kp, same_split, row_chunk=4096
):
    n_classes = coverage.shape[0]
    out = np.full((len(query.class_index), kp), -1, dtype=np.int32)
    query_order = np.argsort(query.class_index, kind="stable")
    dsel_order = np.argsort(dsel.class_index, kind="stable")
    query_pointer = np.searchsorted(
        query.class_index[query_order], np.arange(n_classes + 1)
    )
    dsel_pointer = np.searchsorted(
        dsel.class_index[dsel_order], np.arange(n_classes + 1)
    )

    for class_idx in range(n_classes):
        query_rows = query_order[
            query_pointer[class_idx] : query_pointer[class_idx + 1]
        ]
        if len(query_rows) == 0:
            continue
        dsel_rows = dsel_order[dsel_pointer[class_idx] : dsel_pointer[class_idx + 1]]
        dsel_rows = dsel_rows[dsel_mask[dsel.molecule[dsel_rows]]]
        if len(dsel_rows) == 0:
            continue
        columns = np.flatnonzero(coverage[class_idx])
        reference = np.nan_to_num(
            dsel.profiles[dsel_rows][:, columns], nan=POSITIVE_THRESHOLD
        )
        reference_ids = dsel.molecule[dsel_rows]
        reference_square = (reference**2).sum(axis=1)
        effective_kp = min(kp, len(dsel_rows) - 1 if same_split else len(dsel_rows))
        if effective_kp < 1:
            continue

        for start in range(0, len(query_rows), row_chunk):
            block = query_rows[start : start + row_chunk]
            probe = np.nan_to_num(
                query.profiles[block][:, columns], nan=POSITIVE_THRESHOLD
            )
            distance = (
                (probe**2).sum(axis=1)[:, None]
                + reference_square[None, :]
                - 2 * probe @ reference.T
            )
            if same_split:
                distance[query.molecule[block][:, None] == reference_ids[None, :]] = (
                    np.inf
                )
            take = min(4 * effective_kp, distance.shape[1] - 1)
            near = np.argpartition(distance, take, axis=1)[:, : take + 1]
            near_distance = np.take_along_axis(distance, near, axis=1)
            near_ids = reference_ids[near]
            order = np.lexsort((near_ids, near_distance), axis=1)[:, :effective_kp]
            out[block, :effective_kp] = np.take_along_axis(near_ids, order, axis=1)
            if effective_kp < kp:
                out[block, effective_kp:] = out[block, effective_kp - 1][:, None]
    return out


class MetaChunk:
    __slots__ = ("X", "covered", "profiles", "consensus", "alpha", "molecule")

    def __init__(self, X, covered, profiles, consensus, alpha, molecule):
        self.X = X
        self.covered = covered
        self.profiles = profiles
        self.consensus = consensus
        self.alpha = alpha
        self.molecule = molecule


def build_chunk(
    candidates,
    pair_slice,
    region,
    output_profiles,
    dsel,
    coverage,
    labels=None,
    model_id=False,
):
    first, last = pair_slice
    molecule = candidates.molecule[first:last]
    class_index = candidates.class_index[first:last]
    query_scores = candidates.profiles[first:last]
    covered = coverage[class_index]

    neighbours = region[molecule]
    neighbour_scores, neighbour_labels = dsel.gather(neighbours, class_index)
    matches = (neighbour_scores > POSITIVE_THRESHOLD) == neighbour_labels[:, :, None]
    f1 = matches & covered[:, None, :]
    f2 = np.where(
        neighbour_labels[:, :, None], neighbour_scores, 1.0 - neighbour_scores
    )
    f3 = f1.sum(axis=1, dtype=np.float32) / region.shape[1]

    profile_scores, profile_labels = dsel.gather(
        output_profiles[first:last], class_index
    )
    f4 = (
        (profile_scores > POSITIVE_THRESHOLD) == profile_labels[:, :, None]
    ) & covered[:, None, :]

    filled_query = np.nan_to_num(query_scores, nan=POSITIVE_THRESHOLD)
    f5 = 2.0 * np.abs(filled_query - POSITIVE_THRESHOLD)

    blocks = [
        f1.transpose(0, 2, 1).astype(np.float32),
        np.nan_to_num(f2, nan=POSITIVE_THRESHOLD).transpose(0, 2, 1),
        f3[:, :, None],
        f4.transpose(0, 2, 1).astype(np.float32),
        f5[:, :, None],
    ]
    if model_id:
        n_models = covered.shape[1]
        blocks.append(
            np.broadcast_to(
                np.eye(n_models, dtype=np.float32), (len(molecule), n_models, n_models)
            )
        )
    X = np.concatenate(blocks, axis=2)

    alpha = None
    if labels is not None:
        truth = labels[molecule, class_index]
        alpha = ((query_scores > POSITIVE_THRESHOLD) == truth[:, None]) & covered

    positive = ((query_scores > POSITIVE_THRESHOLD) & covered).sum(axis=1)
    n_covered = covered.sum(axis=1)
    agreement = np.maximum(positive, n_covered - positive) / np.maximum(n_covered, 1)

    unusable = (neighbours < 0).any(axis=1)
    if unusable.any():
        covered = covered.copy()
        covered[unusable] = False

    return MetaChunk(X, covered, filled_query, agreement, alpha, molecule)


def aggregate(delta, chunk, competence_threshold, vote):
    selected = (delta > competence_threshold) & chunk.covered
    empty = ~selected.any(axis=1)
    selected[empty] = chunk.covered[empty]
    weight = delta * selected
    direction = np.where(chunk.profiles > POSITIVE_THRESHOLD, 1.0, -1.0)
    if vote == "confidence":
        direction = direction * 2.0 * np.abs(chunk.profiles - POSITIVE_THRESHOLD)
    total = weight.sum(axis=1)
    return ((weight * direction).sum(axis=1) / np.maximum(total, 1e-6)).astype(
        np.float32
    )


class DynamicSelectionEnsemble(VotingEnsemble):

    def __init__(
        self,
        ensemble_dir: str,
        candidate_k: int = 50,
        region_size=None,
        profile_size=None,
        vote=None,
        consensus_threshold: float = 0.7,
        competence_threshold: float = 0.5,
        region_size_grid=REGION_SIZE_GRID,
        profile_size_grid=PROFILE_SIZE_GRID,
        vote_grid=VOTE_GRID,
        chunk_size: int = CHUNK_SIZE,
        use_model_id: bool = True,
        meta_classifier: str = "nb",
        max_meta_samples: int = MAX_META_SAMPLES,
        morgan_radius: int = MORGAN_RADIUS,
        morgan_bits: int = MORGAN_BITS,
        morgan_chirality: bool = MORGAN_CHIRALITY,
        full_dsel: bool = False,
        **kwargs,
    ):
        super().__init__(ensemble_dir)
        self.morgan_radius = int(morgan_radius)
        self.morgan_bits = int(morgan_bits)
        self.morgan_chirality = bool(morgan_chirality)
        self.full_dsel = bool(full_dsel)
        if meta_classifier not in ("nb", "mlp"):
            raise ValueError(
                f"Unknown meta_classifier '{meta_classifier}', expected 'nb' or 'mlp'."
            )
        self.use_model_id = bool(use_model_id)
        self.meta_classifier = meta_classifier
        self.max_meta_samples = int(max_meta_samples)
        self.candidate_k = candidate_k
        self.region_size = region_size
        self.profile_size = profile_size
        self.vote = vote
        self.consensus_threshold = consensus_threshold
        self.competence_threshold = competence_threshold
        self.region_size_grid = tuple(region_size_grid)
        self.profile_size_grid = tuple(profile_size_grid)
        self.vote_grid = tuple(vote_grid)
        self.chunk_size = chunk_size
        self._classifier = None
        self._metadata = None
        self._dsel = None
        self._dsel_fingerprints = None
        self._coverage = None
        self._thresholds = None

    @property
    def _classifier_path(self):
        return Path(self.ensemble_dir) / "des_meta_classifier.pkl"

    @property
    def _dsel_path(self):
        return Path(self.ensemble_dir) / "des_dsel.npz"

    @property
    def _metadata_path(self):
        return Path(self.ensemble_dir) / "des_metadata.json"

    def calibrate(self, validation_predictions, validation_data, validation_labels):
        super().calibrate(validation_predictions, validation_data, validation_labels)
        scores, model_names = stack_predictions(
            validation_predictions, dtype=np.float16
        )
        thresholds = threshold_array(self._load_prediction_thresholds(), model_names)
        labels = np.asarray(validation_labels, dtype=bool)
        coverage = coverage_of(scores)
        molecule_fingerprints, parsed = fingerprints(
            validation_data, self.morgan_radius, self.morgan_bits, self.morgan_chirality
        )
        if not parsed.all():
            print(
                f"{int((~parsed).sum())} of {len(parsed)} validation molecules could not be "
                "fingerprinted and are excluded from the dynamic selection reference set."
            )
        candidates = Candidates(scores, self.candidate_k, thresholds)
        dsel = Dsel(scores, labels, thresholds)

        best = None
        if None in (self.region_size, self.profile_size, self.vote):
            best = self._optimize_hyperparameters(
                candidates, dsel, coverage, molecule_fingerprints, parsed, labels
            )
        region_size = self.region_size or best["region_size"]
        profile_size = self.profile_size or best["profile_size"]
        vote = self.vote or best["vote"]

        dsel_mask, dev_mask = holdout_split(parsed)
        region, profiles = self._neighbourhoods(
            candidates,
            molecule_fingerprints,
            parsed,
            dsel_mask,
            coverage,
            region_size,
            profile_size,
        )
        classifier = self._fit_meta_classifier(
            candidates, region, profiles, dsel, coverage, dsel_mask, labels
        )
        net = self._score(
            candidates, region, profiles, dsel, coverage, classifier, dev_mask, [vote]
        )[vote]
        scorer, keep = pair_scorer(
            labels, candidates.molecule, candidates.class_index, dev_mask
        )
        tau, dev_macro_f1 = scorer.tune(net[keep])

        # the meta-classifier and tau are fitted with the dev molecules held out of the reference
        # set, but nothing stops prediction from looking neighbours up in all of them
        reference_mask = parsed if self.full_dsel else dsel_mask
        self._save(
            classifier,
            scores,
            labels,
            molecule_fingerprints,
            reference_mask,
            coverage,
            {
                "model_names": model_names,
                "candidate_k": int(self.candidate_k),
                "region_size": int(region_size),
                "profile_size": int(profile_size),
                "vote": vote,
                "use_model_id": self.use_model_id,
                "meta_classifier": self.meta_classifier,
                "morgan_radius": self.morgan_radius,
                "morgan_bits": self.morgan_bits,
                "morgan_chirality": self.morgan_chirality,
                "full_dsel": self.full_dsel,
                "consensus_threshold": float(self.consensus_threshold),
                "competence_threshold": float(self.competence_threshold),
                "tau": float(tau),
                "n_classes": int(scores.shape[1]),
                "n_dsel": int(reference_mask.sum()),
                "dev_macro_f1": float(dev_macro_f1),
                "normalized": True,
            },
        )
        print(
            f"Saved meta-classifier to {self._classifier_path} (region_size={region_size}, "
            f"profile_size={profile_size}, vote={vote}, meta_classifier={self.meta_classifier}, "
            f"use_model_id={self.use_model_id}, tau={tau:.4f}, "
            f"held-out macro-f1: {dev_macro_f1:.4f})."
        )

    def _neighbourhoods(
        self,
        candidates,
        molecule_fingerprints,
        parsed,
        dsel_mask,
        coverage,
        region_size,
        profile_size,
    ):
        reference_ids = np.flatnonzero(dsel_mask)
        self_columns = np.full(len(parsed), -1, dtype=np.int64)
        self_columns[reference_ids] = np.arange(len(reference_ids))
        region = region_of_competence(
            molecule_fingerprints,
            parsed,
            molecule_fingerprints[reference_ids],
            reference_ids,
            region_size,
            self_columns=self_columns,
        )
        profiles = output_profile_knn(
            candidates, candidates, dsel_mask, coverage, profile_size, same_split=True
        )
        return region, profiles

    def _meta_samples(
        self, candidates, region, profiles, dsel, coverage, molecule_mask, labels
    ):
        """The meta-training samples, one block of (features, is the base learner correct?) rows
        per chunk of molecules. Only pairs the base learners disagree on are kept."""
        for start, end, pair_slice in candidates.iter_chunks(self.chunk_size):
            block = molecule_mask[start:end]
            if not block.any():
                continue
            chunk = build_chunk(
                candidates,
                pair_slice,
                region,
                profiles,
                dsel,
                coverage,
                labels=labels,
                model_id=self.use_model_id,
            )
            keep = block[chunk.molecule - start] & (
                chunk.consensus < self.consensus_threshold
            )
            if not keep.any():
                continue
            mask = chunk.covered & keep[:, None]
            if not mask.any():
                continue
            yield chunk.X[mask], chunk.alpha[mask]

    def _fit_meta_classifier(
        self, candidates, region, profiles, dsel, coverage, molecule_mask, labels
    ):
        blocks = self._meta_samples(
            candidates, region, profiles, dsel, coverage, molecule_mask, labels
        )
        if self.meta_classifier == "nb":
            from sklearn.naive_bayes import GaussianNB

            classifier = GaussianNB()
            meta_classes = np.array([0, 1])
            seen = np.zeros(2, dtype=np.int64)
            for X, alpha in blocks:
                classifier.partial_fit(X, alpha, classes=meta_classes)
                seen += np.bincount(alpha.astype(np.int64), minlength=2)
        else:
            X, alpha = [], []
            for block_X, block_alpha in blocks:
                X.append(block_X)
                alpha.append(block_alpha)
            if not X:
                X, alpha = [np.zeros((0, 0), dtype=np.float32)], [
                    np.zeros(0, dtype=bool)
                ]
            X, alpha = np.concatenate(X), np.concatenate(alpha)
            seen = np.bincount(alpha.astype(np.int64), minlength=2)
            classifier = None
            if seen.all():
                if len(X) > self.max_meta_samples:
                    print(
                        f"Subsampling {len(X)} meta-training samples to {self.max_meta_samples}."
                    )
                    take = np.random.default_rng(RANDOM_SEED).choice(
                        len(X), size=self.max_meta_samples, replace=False
                    )
                    X, alpha = X[take], alpha[take]
                classifier = self._fit_mlp(X, alpha)
        if not seen.all():
            raise RuntimeError(
                f"Meta-training set is unusable ({seen[1]} correct / {seen[0]} incorrect base "
                "learner predictions survived the consensus filter). Increase "
                "consensus_threshold or check the base learner predictions."
            )
        return classifier

    def _fit_mlp(self, X, alpha):
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        print(f"Fitting MLP meta-classifier on {X.shape[0]} x {X.shape[1]} samples...")
        classifier = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=MLP_HIDDEN_LAYERS,
                early_stopping=True,
                n_iter_no_change=5,
                max_iter=200,
                random_state=RANDOM_SEED,
            ),
        )
        classifier.fit(X, alpha)
        return classifier

    def _score(
        self,
        candidates,
        region,
        profiles,
        dsel,
        coverage,
        classifier,
        molecule_mask,
        votes,
    ):
        nets = {
            vote: np.zeros(len(candidates.class_index), dtype=np.float32)
            for vote in votes
        }
        for start, end, pair_slice in candidates.iter_chunks(self.chunk_size):
            if molecule_mask is not None and not molecule_mask[start:end].any():
                continue
            chunk = build_chunk(
                candidates,
                pair_slice,
                region,
                profiles,
                dsel,
                coverage,
                model_id=self.use_model_id,
            )
            delta = np.zeros(chunk.covered.shape, dtype=np.float32)
            if chunk.covered.any():
                delta[chunk.covered] = classifier.predict_proba(chunk.X[chunk.covered])[
                    :, 1
                ].astype(np.float32)
            first, last = pair_slice
            for vote in votes:
                nets[vote][first:last] = aggregate(
                    delta, chunk, self.competence_threshold, vote
                )
        return nets

    def _optimize_hyperparameters(
        self, candidates, dsel, coverage, molecule_fingerprints, parsed, labels
    ):
        print(
            f"Optimizing region_size / profile_size / vote with {N_FOLDS}-fold "
            "cross-validation on the validation set..."
        )
        widest_region = max(self.region_size_grid)
        widest_profile = max(self.profile_size_grid)
        fold_scores = {
            (region_size, profile_size, vote): []
            for region_size in self.region_size_grid
            for profile_size in self.profile_size_grid
            for vote in self.vote_grid
        }

        for fold, test_idx in enumerate(cv_folds(candidates.n_molecules)):
            print(f"Calibrating fold {fold + 1}/{N_FOLDS}...")
            test_mask = np.zeros(candidates.n_molecules, dtype=bool)
            test_mask[test_idx] = True
            dsel_mask, dev_mask = holdout_split(
                ~test_mask & parsed, seed=RANDOM_SEED + fold
            )
            region, profiles = self._neighbourhoods(
                candidates,
                molecule_fingerprints,
                parsed,
                dsel_mask,
                coverage,
                widest_region,
                widest_profile,
            )
            dev_scorer, dev_keep = pair_scorer(
                labels, candidates.molecule, candidates.class_index, dev_mask
            )
            test_scorer, test_keep = pair_scorer(
                labels, candidates.molecule, candidates.class_index, test_mask
            )
            for region_size in self.region_size_grid:
                for profile_size in self.profile_size_grid:
                    classifier = self._fit_meta_classifier(
                        candidates,
                        region[:, :region_size],
                        profiles[:, :profile_size],
                        dsel,
                        coverage,
                        dsel_mask,
                        labels,
                    )
                    nets = self._score(
                        candidates,
                        region[:, :region_size],
                        profiles[:, :profile_size],
                        dsel,
                        coverage,
                        classifier,
                        dev_mask | test_mask,
                        self.vote_grid,
                    )
                    for vote, net in nets.items():
                        tau, _ = dev_scorer.tune(net[dev_keep])
                        fold_scores[(region_size, profile_size, vote)].append(
                            test_scorer.macro_f1(net[test_keep], tau)
                        )

        results = []
        for (region_size, profile_size, vote), scores in fold_scores.items():
            mean_score = float(np.mean(scores))
            results.append(
                {
                    "region_size": region_size,
                    "profile_size": profile_size,
                    "vote": vote,
                    "mean_macro_f1": mean_score,
                    "std_macro_f1": float(np.std(scores)),
                    **{f"fold_{i}_macro_f1": s for i, s in enumerate(scores)},
                }
            )
            print(
                f"region_size={region_size}, profile_size={profile_size}, vote={vote}: "
                f"macro-f1 {mean_score:.4f}"
            )
        best = max(results, key=lambda result: result["mean_macro_f1"])
        save_hyperparameter_results(
            self.ensemble_dir,
            results,
            {
                "region_size": best["region_size"],
                "profile_size": best["profile_size"],
                "vote": best["vote"],
                "mean_macro_f1": best["mean_macro_f1"],
            },
        )
        return best

    def _save(
        self,
        classifier,
        scores,
        labels,
        molecule_fingerprints,
        dsel_mask,
        coverage,
        metadata,
    ):
        with open(self._classifier_path, "wb") as f:
            pickle.dump(classifier, f)
        rows = np.flatnonzero(dsel_mask)
        np.savez_compressed(
            self._dsel_path,
            scores=scores[rows],
            labels=labels[rows],
            fingerprints=np.packbits(molecule_fingerprints[rows], axis=1),
            coverage=coverage,
        )
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _load(self):
        if self._classifier is not None:
            return
        if not self._metadata_path.exists():
            raise FileNotFoundError(
                f"No calibrated meta-classifier found in ensemble directory: {self.ensemble_dir}. "
                "Please calibrate the ensemble first."
            )
        with open(self._metadata_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)
        if not self._metadata.get("normalized"):
            raise ValueError(
                f"The ensemble in {self.ensemble_dir} was calibrated before net scores were "
                "normalised by the selection weight. Its stored tau is on the old (unnormalised) "
                "scale and would reject every prediction. Please re-run `chebifier build` for this "
                "ensemble."
            )
        with open(self._classifier_path, "rb") as f:
            self._classifier = pickle.load(f)
        self._thresholds = threshold_array(
            self._load_prediction_thresholds(), self._metadata["model_names"]
        )
        # the stored reference fingerprints are of whatever kind calibration used, so the query
        # fingerprints have to be built the same way rather than from today's defaults
        self.morgan_radius = self._metadata["morgan_radius"]
        self.morgan_bits = self._metadata["morgan_bits"]
        self.morgan_chirality = self._metadata["morgan_chirality"]
        with np.load(self._dsel_path) as data:
            self._dsel = Dsel(data["scores"], data["labels"], self._thresholds)
            self._dsel_fingerprints = np.unpackbits(
                data["fingerprints"], axis=1, count=self.morgan_bits
            )
            self._coverage = data["coverage"]
        self.consensus_threshold = self._metadata["consensus_threshold"]
        self.competence_threshold = self._metadata["competence_threshold"]
        # the stored classifier was fit on a feature layout these two decide - restoring them is
        # what lets `evaluate` load a variant without being told which one it is
        self.use_model_id = self._metadata["use_model_id"]
        self.meta_classifier = self._metadata["meta_classifier"]

    def predict(self, test_predictions, molecules=None):
        if molecules is None:
            raise ValueError(
                f"{self.ensemble_name} needs the molecules it predicts for, to look up their "
                "region of competence. Pass them to chebifier.predict.predict."
            )
        self._load()
        scores, _ = stack_predictions(
            test_predictions, self._metadata["model_names"], dtype=np.float16
        )
        query_fingerprints, parsed = fingerprints(
            molecules, self.morgan_radius, self.morgan_bits, self.morgan_chirality
        )
        candidates = Candidates(scores, self._metadata["candidate_k"], self._thresholds)
        dsel_candidates = Candidates(
            self._dsel.scores, self._metadata["candidate_k"], self._thresholds
        )
        reference_ids = np.arange(self._dsel.scores.shape[0])
        region = region_of_competence(
            query_fingerprints,
            parsed,
            self._dsel_fingerprints,
            reference_ids,
            self._metadata["region_size"],
        )
        profiles = output_profile_knn(
            candidates,
            dsel_candidates,
            np.ones(dsel_candidates.n_molecules, dtype=bool),
            self._coverage,
            self._metadata["profile_size"],
            same_split=False,
        )
        vote = self._metadata["vote"]
        net = self._score(
            candidates,
            region,
            profiles,
            self._dsel,
            self._coverage,
            self._classifier,
            None,
            [vote],
        )[vote]
        dense = dense_from_pairs(
            candidates.molecule,
            candidates.class_index,
            net - self._metadata["tau"],
            scores.shape[:2],
        )
        return {
            "net_score": torch.from_numpy(dense),
            "has_valid_predictions": torch.from_numpy((~np.isnan(scores)).any(axis=2)),
        }
