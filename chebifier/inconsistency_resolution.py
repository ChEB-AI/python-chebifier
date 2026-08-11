import csv
import os
from pathlib import Path

import networkx as nx
import torch
from chebi_utils.obo_extractor import get_hierarchy_subgraph


def get_disjoint_groups(disjoint_files):
    if disjoint_files is None:
        disjoint_files = os.path.join("data", "chebi-disjoints.owl")
    disjoint_pairs, disjoint_groups = [], []
    for file in disjoint_files:
        if isinstance(file, Path):
            file = str(file)
        if file.endswith(".csv"):
            with open(file, "r") as f:
                reader = csv.reader(f)
                disjoint_pairs += [line for line in reader]
        elif file.endswith(".owl"):
            with open(file, "r") as f:
                plaintext = f.read()
                segments = plaintext.split("<")
                disjoint_pairs = []
                left = None
                for seg in segments:
                    if seg.startswith("rdf:Description ") or seg.startswith(
                        "owl:Class"
                    ):
                        left = seg.split('rdf:about="&obo;CHEBI_')[1].split('"')[0]
                    elif seg.startswith("owl:disjointWith"):
                        right = seg.split('rdf:resource="&obo;CHEBI_')[1].split('"')[0]
                        disjoint_pairs.append([left, right])

                disjoint_groups = []
                for seg in plaintext.split("<rdf:Description>"):
                    if "owl;AllDisjointClasses" in seg:
                        classes = seg.split('rdf:about="&obo;CHEBI_')[1:]
                        classes = [c.split('"')[0] for c in classes]
                        disjoint_groups.append(classes)
        else:
            raise NotImplementedError(
                "Unsupported disjoint file format: " + file.split(".")[-1]
            )

    disjoint_all = disjoint_pairs + disjoint_groups
    # one disjointness is commented out in the owl-file
    # (the correct way would be to parse the owl file and notice the comment symbols, but for this case, it should work)
    if ["22729", "51880"] in disjoint_all:
        disjoint_all.remove(["22729", "51880"])
    # print(f"Found {len(disjoint_all)} disjoint groups")
    return disjoint_all


NEUTRAL = 0.5


def to_logit(p):
    """Log-odds of a probability, for the one place that needs them: the HEX log-linear model.

    Ensembles report probabilities, which is what the resolution methods work in. Only the HEX
    softmax over legal states needs logits, and its unanimous predictions sit exactly at 0 and 1,
    so the clamp is what keeps them finite.
    """
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p) - torch.log1p(-p)


def densified_exclusion_matrix(label_names, label_successors, disjoint_groups):
    label_index = {label: i for i, label in enumerate(label_names)}
    succ = label_successors[0] if label_successors.dim() == 3 else label_successors
    n = succ.shape[0]
    excl = torch.zeros((n, n), dtype=torch.bool)
    for group in disjoint_groups:
        members = [label_index[g] for g in group if g in label_index]
        for gi in range(len(members)):
            for gj in range(gi + 1, len(members)):
                subs_a = succ[:, members[gi]]
                subs_b = succ[:, members[gj]]
                block = subs_a.unsqueeze(1) & subs_b.unsqueeze(0)
                excl |= block | block.T
    excl.fill_diagonal_(False)
    return excl


def densified_exclusion_pairs(label_names, label_successors, disjoint_groups):
    excl = densified_exclusion_matrix(label_names, label_successors, disjoint_groups)
    return torch.nonzero(torch.triu(excl), as_tuple=False)


def get_smoother_class(name):
    from chebifier.hex_graph import HexSmoother
    from chebifier.ilr import GodelILRSmoother, LukasiewiczILRSmoother

    smoothers = {
        "score-based": ScoreBasedPredictionSmoother,
        "ilr-godel": GodelILRSmoother,
        "ilr-lukasiewicz": LukasiewiczILRSmoother,
        "hex": HexSmoother,
    }
    if name not in smoothers:
        raise ValueError(
            f"Unknown inconsistency resolution method '{name}'. "
            f"Available: {', '.join(smoothers)}"
        )
    return smoothers[name]


SMOOTHER_NAMES = ["score-based", "ilr-godel", "ilr-lukasiewicz", "hex"]


class PredictionSmoother:
    """Removes implication and disjointness violations from predictions.

    Predictions are probabilities in [0, 1]: NEUTRAL (0.5) means the ensemble is undecided, and a
    class is predicted when its probability exceeds the ensemble's decision threshold.
    """

    def __init__(
        self, chebi_graph, label_names=None, disjoint_files=None, verbose=False
    ):
        self.chebi_graph = chebi_graph
        self.set_label_names(label_names)
        self.disjoint_groups = get_disjoint_groups(disjoint_files)
        self.verbose = verbose

    def set_label_names(self, label_names):
        if label_names is not None:
            self.label_names = label_names
            # the ChEBI graph also contains non-subsumption relations (has role, conjugate
            # acid/base, has functional parent, ...) which are not implications
            isa_graph = get_hierarchy_subgraph(self.chebi_graph)
            label_index = {label: i for i, label in enumerate(self.label_names)}
            self.label_successors = torch.zeros(
                (len(self.label_names), len(self.label_names)), dtype=torch.bool
            )
            for i, label in enumerate(self.label_names):
                self.label_successors[i, i] = 1
                if label not in isa_graph:
                    continue
                # transitive closure: superclasses can be connected via intermediate
                # classes that are not themselves labels
                for p in nx.descendants(isa_graph, label):
                    if p in label_index:
                        self.label_successors[i, label_index[p]] = 1
            self.label_successors = self.label_successors.unsqueeze(0)

    def resolve_subsumption_violations(self, preds):
        preds = preds.unsqueeze(1)
        preds_masked_succ = torch.where(self.label_successors, preds, 0)
        # preds_masked_succ shape: (n_samples, n_labels, n_labels)
        return preds_masked_succ.max(dim=2).values

    def resolve_disjointness_violations(self, preds):
        preds_sum_orig = torch.sum(preds)

        for disj_group in self.disjoint_groups:
            disj_group = [
                self.label_names.index(g) for g in disj_group if g in self.label_names
            ]
            if len(disj_group) > 1:
                group_preds = preds[:, disj_group]
                keep = torch.zeros_like(group_preds, dtype=torch.bool)
                keep[torch.arange(group_preds.shape[0]), group_preds.argmax(dim=1)] = (
                    True
                )
                preds[:, disj_group] = torch.where(
                    keep, group_preds, group_preds.clamp(max=NEUTRAL)
                )
        if self.verbose and torch.sum(preds) != preds_sum_orig:
            print(f"Preds change (step 2): {torch.sum(preds) - preds_sum_orig}")
        preds_sum_orig = torch.sum(preds)
        # step 3: disjointness violation removal may have caused new implication inconsistencies -> set each prediction to min of superclasses
        preds = preds.unsqueeze(1)
        preds_masked_succ = torch.where(self.label_successors, preds, torch.inf)
        preds = preds_masked_succ.min(dim=2).values
        if self.verbose and torch.sum(preds) != preds_sum_orig:
            print(f"Preds change (step 3): {torch.sum(preds) - preds_sum_orig}")
        return preds

    def __call__(self, preds, valid_mask=None):
        if preds.shape[1] == 0:
            # no labels predicted
            return preds
        # preds shape: (n_samples, n_labels)
        preds_sum_orig = torch.sum(preds)
        # step 1: apply implications: for each class, set prediction to max of itself and all successors
        preds = self.resolve_subsumption_violations(preds)

        if self.verbose and torch.sum(preds) != preds_sum_orig:
            print(f"Preds change (step 1): {torch.sum(preds) - preds_sum_orig}")
        # step 2: eliminate disjointness violations: for group of disjoint classes, set all except max to 0 (if it is not already lower)
        preds = self.resolve_disjointness_violations(preds)
        return preds


class PessimisticPredictionSmoother(PredictionSmoother):
    """Always assumes the positive prediction is wrong (in case of implication violations)"""

    def resolve_subsumption_violations(self, preds):
        preds = preds.unsqueeze(1)
        preds_masked_predec = torch.where(
            torch.transpose(self.label_successors, 1, 2), preds, 1
        )
        preds = preds_masked_predec.min(dim=2).values
        return preds


class ScoreBasedPredictionSmoother(PredictionSmoother):
    """Removes implication violations from predictions based on the predicted probabilities: for A
    subclassOf B where score(A) > score(B), either set score(B) = max(score(B), score(A)) if A is
    further from NEUTRAL than B, or set score(A) = min(score(A), score(B)) otherwise.
    """

    def resolve_subsumption_violations(self, preds):
        preds = preds.unsqueeze(1)
        # label_successors[i, j] means j is a superclass of i, so raising a class to the score of its
        # subclasses means taking the max over its predecessors (and vice versa for lowering it).
        preds_masked_predec = torch.where(
            torch.transpose(self.label_successors, 1, 2), preds, -torch.inf
        )
        preds_optimistic = preds_masked_predec.max(dim=2).values
        preds_masked_succ = torch.where(self.label_successors, preds, torch.inf)
        preds_pessimistic = preds_masked_succ.min(dim=2).values
        # take whichever the ensemble is more confident about, i.e. further from NEUTRAL
        preds_direction = (preds_optimistic - NEUTRAL).abs() > (
            preds_pessimistic - NEUTRAL
        ).abs()
        return torch.where(preds_direction, preds_optimistic, preds_pessimistic)
