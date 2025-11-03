import csv
import os
from pathlib import Path

import torch


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
                        left = int(seg.split('rdf:about="&obo;CHEBI_')[1].split('"')[0])
                    elif seg.startswith("owl:disjointWith"):
                        right = int(
                            seg.split('rdf:resource="&obo;CHEBI_')[1].split('"')[0]
                        )
                        disjoint_pairs.append([left, right])

                disjoint_groups = []
                for seg in plaintext.split("<rdf:Description>"):
                    if "owl;AllDisjointClasses" in seg:
                        classes = seg.split('rdf:about="&obo;CHEBI_')[1:]
                        classes = [int(c.split('"')[0]) for c in classes]
                        disjoint_groups.append(classes)
        else:
            raise NotImplementedError(
                "Unsupported disjoint file format: " + file.split(".")[-1]
            )

    disjoint_all = disjoint_pairs + disjoint_groups
    # one disjointness is commented out in the owl-file
    # (the correct way would be to parse the owl file and notice the comment symbols, but for this case, it should work)
    if [22729, 51880] in disjoint_all:
        disjoint_all.remove([22729, 51880])
    # print(f"Found {len(disjoint_all)} disjoint groups")
    return disjoint_all


class PredictionSmoother:
    """Removes implication and disjointness violations from predictions"""

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
            chebi_subgraph = self.chebi_graph.subgraph(self.label_names)
            self.label_successors = torch.zeros(
                (len(self.label_names), len(self.label_names)), dtype=torch.bool
            )
            for i, label in enumerate(self.label_names):
                self.label_successors[i, i] = 1
                for p in chebi_subgraph.successors(label):
                    if p in self.label_names:
                        self.label_successors[i, self.label_names.index(p)] = 1
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
                disj_max = torch.max(preds[:, disj_group], dim=1)
                for i, row in enumerate(preds):
                    for l_ in range(len(preds[i])):
                        if l_ in disj_group and l_ != disj_group[disj_max.indices[i]]:
                            preds[i, l_] = 0
        if self.verbose and torch.sum(preds) != preds_sum_orig:
            print(f"Preds change (step 2): {torch.sum(preds) - preds_sum_orig}")
        preds_sum_orig = torch.sum(preds)
        # step 3: disjointness violation removal may have caused new implication inconsistencies -> set each prediction to min of predecessors
        preds = preds.unsqueeze(1)
        preds_masked_predec = torch.where(
            torch.transpose(self.label_successors, 1, 2), preds, 1
        )
        preds = preds_masked_predec.min(dim=2).values
        if self.verbose and torch.sum(preds) != preds_sum_orig:
            print(f"Preds change (step 3): {torch.sum(preds) - preds_sum_orig}")
        return preds

    def __call__(self, preds):
        if preds.shape[1] == 0:
            # no labels predicted
            return preds
        # preds shape: (n_samples, n_labels)
        preds_sum_orig = torch.sum(preds)
        # step 1: apply implications: for each class, set prediction to max of itself and all successors
        preds = self.resolve_subsumption_violations(preds)

        if self.verbose and torch.sum(preds) != preds_sum_orig:
            print(f"Preds change (step 1): {torch.sum(preds) - preds_sum_orig}")
        # step 2: eliminate disjointness violations: for group of disjoint classes, set all except max to 0.49 (if it is not already lower)
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
    """Removes implication violations from predictions based on net scores: for A subclassOf B where score(A) > score(B), either set score(B) = max(score(B), score(A))
    if abs(score(A)) > abs(score(B)) or set score(A) = min(score(A), score(B)) otherwise.
    """

    def resolve_subsumption_violations(self, preds):
        preds = preds.unsqueeze(1)
        preds_masked_succ = torch.where(self.label_successors, preds, 0)
        preds_optimistic = preds_masked_succ.max(dim=2).values
        preds_masked_predec = torch.where(
            torch.transpose(self.label_successors, 1, 2), preds, 1
        )
        preds_pessimistic = preds_masked_predec.min(dim=2).values
        # take the one with the higher absolute value
        preds_direction = preds_optimistic - preds_pessimistic > 0
        return torch.where(preds_direction, preds_optimistic, preds_pessimistic)
