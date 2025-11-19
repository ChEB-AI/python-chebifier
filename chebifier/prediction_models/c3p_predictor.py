import os
from pathlib import Path
from typing import List, Optional

import tqdm

from chebifier import modelwise_smiles_lru_cache
from chebifier.prediction_models import BasePredictor


class C3PPredictor(BasePredictor):
    """
    Wrapper for C3P (url).
    """

    def __init__(
        self,
        model_name: str,
        program_directory: Optional[Path] = None,
        chemical_classes: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.program_directory = program_directory
        self.chemical_classes = chemical_classes
        self.chebi_graph = kwargs.get("chebi_graph", None)

    @modelwise_smiles_lru_cache.batch_decorator
    def predict_smiles_list(self, smiles_list: list[str]) -> list:
        from c3p import classifier as c3p_classifier

        result_list = []
        for batch_start in tqdm.tqdm(
            range(0, len(smiles_list), 32), desc="Classifying with C3P"
        ):
            batch_end = min(batch_start + 32, len(smiles_list))
            result_list.extend(
                c3p_classifier.classify(
                    smiles_list[batch_start:batch_end],
                    self.program_directory,
                    self.chemical_classes,
                    strict=False,
                )
            )

        result_reformatted = [dict() for _ in range(len(smiles_list))]
        for result in tqdm.tqdm(result_list, desc="Reformatting C3P results"):
            chebi_id = result.class_id.split(":")[1]
            result_reformatted[smiles_list.index(result.input_smiles)][
                chebi_id
            ] = result.is_match
            if result.is_match and self.chebi_graph is not None:
                for parent in list(self.chebi_graph.predecessors(chebi_id)):
                    result_reformatted[smiles_list.index(result.input_smiles)][
                        str(parent)
                    ] = 1
        return result_reformatted

    def explain_smiles(self, smiles):
        """
        C3P provides natural language explanations for each prediction (positive or negative). Since there are more
        than 300 classes, only take the positive ones.
        """
        from c3p import classifier as c3p_classifier

        highlights = []
        result_list = c3p_classifier.classify(
            [smiles], self.program_directory, self.chemical_classes, strict=False
        )
        for result in result_list:
            if result.is_match:
                highlights.append(
                    (
                        "text",
                        f"For {result.class_name} ({result.class_id}), C3P gave the following explanation: {result.reason}",
                    )
                )
        highlights = [
            (
                "text",
                f"C3P made positive predictions for {len(highlights)} classes. {'The explanations are as follows:' if len(highlights) > 0 else ''}",
            )
        ] + highlights

        return {"highlights": highlights}

    def calculate_trust(self, c3p_classes_path, output_path="c3p_trust.json"):
        """Use reported confidence of C3P to calculate the trust. Use either the directly reported values or infer based on subclasses"""
        from c3p.classifier import PROGRAM_DIR

        program_dir = self.program_directory or PROGRAM_DIR
        confusion_matrix = dict()
        for f in os.listdir(program_dir):
            if f.startswith("__"):
                continue
            with open(os.path.join(program_dir, f), encoding="utf-8") as file:
                txt = file.read()

                if "__metadata__" in txt:
                    txt = txt[txt.rindex("__metadata__") + 15 :]
                    chebi_id = txt[
                        txt.index("id")
                        + 12 : txt.index("id")
                        + txt[txt.index("id") :].index(",")
                        - 1
                    ]
                    conf = []
                    if (
                        chebi_id == ""
                        or chebi_id.startswith("R")
                        or chebi_id.startswith("oxy")
                    ):
                        print(f, chebi_id)
                    for name in [
                        "num_true_positives",
                        "num_false_positives",
                        "num_true_negatives",
                        "num_false_negatives",
                    ]:
                        start_index = txt.index(name) + len(name) + 2
                        end_index = start_index + txt[start_index:].index(",")
                        try:
                            number = int(txt[start_index:end_index])
                        except ValueError:
                            print(
                                "Failed to read value near ",
                                txt[start_index - 17 : end_index + 5],
                            )
                            number = 0
                        conf.append(number)
                    confusion_matrix[chebi_id] = {
                        "TP": conf[0],
                        "FP": conf[1],
                        "TN": conf[2],
                        "FN": conf[3],
                    }
                else:
                    print(f"Couldnt find metadata in {f}")

        # for classes where c3p doesn't have a number, take the sum of the subclasses
        new_confusion = dict()
        for cls in confusion_matrix:
            for parent in self.chebi_graph.predecessors(cls):
                if parent not in confusion_matrix:
                    if parent not in new_confusion:
                        new_confusion[parent] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                    new_confusion[parent]["TP"] += confusion_matrix[cls]["TP"]
                    new_confusion[parent]["FP"] += confusion_matrix[cls]["FP"]
                    new_confusion[parent]["TN"] += confusion_matrix[cls]["TN"]
                    new_confusion[parent]["FN"] += confusion_matrix[cls]["FN"]

        import json

        confusion_matrix = {**confusion_matrix, **new_confusion}
        print(
            f"After adding parent classes, confusion matrix contains {len(confusion_matrix)} classes ({len(new_confusion)} indirect)"
        )
        json.dump(confusion_matrix, open(output_path, "w+"))


if __name__ == "__main__":
    import os

    from chebifier.utils import load_chebi_graph

    chebi_graph = load_chebi_graph()
    predictor = C3PPredictor(
        "demo",
        program_directory=os.path.join("..", "c3p", "c3p", "programs"),
        chebi_graph=chebi_graph,
    )
    print(predictor.predict_smiles_list(["CO", "CO"]))
    # predictor.calculate_trust(os.path.join("..", "ensemble-eval", "ensemble_eval_model_preds", "c3p_classes.txt"), "c3p_trust_new.json")
