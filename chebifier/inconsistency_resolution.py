import csv
import os
import torch
from pathlib import Path


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

    def __init__(self, chebi_graph, label_names=None, disjoint_files=None):
        self.chebi_graph = chebi_graph
        self.set_label_names(label_names)
        self.disjoint_groups = get_disjoint_groups(disjoint_files)

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

    def __call__(self, preds):
        if preds.shape[1] == 0:
            # no labels predicted
            return preds
        # preds shape: (n_samples, n_labels)
        preds_sum_orig = torch.sum(preds)
        # step 1: apply implications: for each class, set prediction to max of itself and all successors
        preds = preds.unsqueeze(1)
        preds_masked_succ = torch.where(self.label_successors, preds, 0)
        # preds_masked_succ shape: (n_samples, n_labels, n_labels)

        preds = preds_masked_succ.max(dim=2).values
        if torch.sum(preds) != preds_sum_orig:
            print(f"Preds change (step 1): {torch.sum(preds) - preds_sum_orig}")
        preds_sum_orig = torch.sum(preds)
        # step 2: eliminate disjointness violations: for group of disjoint classes, set all except max to 0.49 (if it is not already lower)
        preds_bounded = torch.min(preds, torch.ones_like(preds) * 0.49)
        for disj_group in self.disjoint_groups:
            disj_group = [
                self.label_names.index(g) for g in disj_group if g in self.label_names
            ]
            if len(disj_group) > 1:
                old_preds = preds[:, disj_group]
                disj_max = torch.max(preds[:, disj_group], dim=1)
                for i, row in enumerate(preds):
                    for l_ in range(len(preds[i])):
                        if l_ in disj_group and l_ != disj_group[disj_max.indices[i]]:
                            preds[i, l_] = preds_bounded[i, l_]
                samples_changed = 0
                for i, row in enumerate(preds[:, disj_group]):
                    if any(r != o for r, o in zip(row, old_preds[i])):
                        samples_changed += 1
                if samples_changed != 0:
                    print(
                        f"disjointness group {[self.label_names[d] for d in disj_group]} changed {samples_changed} samples"
                    )
        if torch.sum(preds) != preds_sum_orig:
            print(f"Preds change (step 2): {torch.sum(preds) - preds_sum_orig}")
        preds_sum_orig = torch.sum(preds)
        # step 3: disjointness violation removal may have caused new implication inconsistencies -> set each prediction to min of predecessors
        preds = preds.unsqueeze(1)
        preds_masked_predec = torch.where(
            torch.transpose(self.label_successors, 1, 2), preds, 1
        )
        preds = preds_masked_predec.min(dim=2).values
        if torch.sum(preds) != preds_sum_orig:
            print(f"Preds change (step 3): {torch.sum(preds) - preds_sum_orig}")
        return preds
