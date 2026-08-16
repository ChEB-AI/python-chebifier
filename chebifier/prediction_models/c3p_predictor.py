import functools
from pathlib import Path
from typing import List, Optional

import tqdm

from chebifier import modelwise_smiles_lru_cache
from chebifier.prediction_models import BasePredictor
from chebifier.utils import _isa_graph, get_superclasses, to_smiles


def _patch_c3p(c3p_classifier):
    """Two things C3P 0.5.0 assumes that do not hold on Windows, both of which make it return no
    classification at all rather than a wrong one:

    - it reads its generated programs with `open(program, "r")`, i.e. in the platform default
      encoding, while the programs are UTF-8. `open` is looked up in the module globals before the
      builtins, so binding a UTF-8 `open` there fixes the reads without affecting any other module.
    - it guards every program with timeout_decorator, which needs SIGALRM. Running the programs
      without their 2s timeout is the only way to get predictions on a platform that has no
      SIGALRM - timeout_decorator's signal-free mode forks a process per call, which is not
      affordable for 300 programs per molecule.
    """
    import signal

    if not hasattr(c3p_classifier, "open"):
        c3p_classifier.open = functools.partial(open, encoding="utf-8")
    if hasattr(signal, "SIGALRM"):
        return
    from c3p import learn

    if hasattr(learn.eval_with_timeout, "__wrapped__"):
        print("No SIGALRM on this platform, running C3P programs without a timeout.")
        learn.eval_with_timeout = learn.eval_with_timeout.__wrapped__


class C3PPredictor(BasePredictor):
    """
    Wrapper for C3P (url).
    """

    def __init__(
        self,
        model_name: str,
        program_directory: Optional[Path] = None,
        chemical_classes: Optional[List[str]] = None,
        keep_classes_outside_graph: bool = False,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.program_directory = program_directory
        self.chemical_classes = chemical_classes
        self.chebi_graph = kwargs.get("chebi_graph", None)
        # C3P ships programs for classes outside the graph's scope - minerals, atoms, mixtures,
        # polymers - which are not molecular entities. Their gold labels would be read off a
        # hierarchy that does not contain them, i.e. no positives at all, so they enter the
        # evaluation as columns nothing can score on. Set this to keep them anyway.
        self.keep_classes_outside_graph = keep_classes_outside_graph

    @modelwise_smiles_lru_cache.batch_decorator
    def predict_list(self, smiles_list: list[str]) -> list:
        from c3p import classifier as c3p_classifier

        _patch_c3p(c3p_classifier)
        # C3P only takes SMILES, while the evaluation datasets hand out RDKit molecules
        smiles_list = [to_smiles(molecule) for molecule in smiles_list]
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

        # Look up the position of each SMILES via a dict instead of scanning smiles_list
        # for every result (C3P returns one result per class and molecule, so the scan
        # made reformatting quadratic in the number of molecules). Repeated SMILES map to
        # all of their positions, which list.index could not do (it always returned the
        # first one, leaving the later rows without any predictions).
        indices_by_smiles: dict[str, list[int]] = {}
        for idx, smiles in enumerate(smiles_list):
            indices_by_smiles.setdefault(smiles, []).append(idx)

        in_scope = None
        if self.chebi_graph is not None and not self.keep_classes_outside_graph:
            in_scope = _isa_graph(self.chebi_graph)

        result_reformatted = [dict() for _ in range(len(smiles_list))]
        dropped = set()
        for result in tqdm.tqdm(result_list, desc="Reformatting C3P results"):
            chebi_id = result.class_id.split(":")[1]
            if in_scope is not None and chebi_id not in in_scope:
                dropped.add(chebi_id)
                continue
            if result.is_match and self.chebi_graph is not None:
                parents = get_superclasses(self.chebi_graph, chebi_id)
            else:
                parents = []
            for idx in indices_by_smiles[result.input_smiles]:
                preds_i = result_reformatted[idx]
                preds_i[chebi_id] = result.is_match
                for parent in parents:
                    preds_i[parent] = 1
        if dropped:
            print(
                f"C3P: dropped {len(dropped)} classes outside the ChEBI graph "
                f"({', '.join(sorted(dropped)[:8])}"
                f"{', ...' if len(dropped) > 8 else ''})"
            )
        return result_reformatted

    def explain_smiles(self, smiles):
        """
        C3P provides natural language explanations for each prediction (positive or negative). Since there are more
        than 300 classes, only take the positive ones.
        """
        from c3p import classifier as c3p_classifier

        _patch_c3p(c3p_classifier)
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
