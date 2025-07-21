from functools import lru_cache
from typing import Optional

from chebifier.prediction_models import BasePredictor
import os
import networkx as nx
from rdkit import Chem
import json


class ChEBILookupPredictor(BasePredictor):

    def __init__(
        self,
        model_name: str,
        description: str = None,
        chebi_version: int = 241,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self._description = (
            description
            or "ChEBI Lookup: If the SMILES is equivalent to a ChEBI entry, retrieve the classification of that entry."
        )
        self.chebi_version = chebi_version
        self.chebi_graph = kwargs.get("chebi_graph", None)
        if self.chebi_graph is None:
            from chebai.preprocessing.datasets.chebi import ChEBIOver50

            self.chebi_dataset = ChEBIOver50(chebi_version=self.chebi_version)
            self.chebi_dataset._download_required_data()
            self.chebi_graph = self.chebi_dataset._extract_class_hierarchy(
                os.path.join(self.chebi_dataset.raw_dir, "chebi.obo")
            )
        self.lookup_table = self.get_smiles_lookup()

    def get_smiles_lookup(self):
        path = os.path.join(
            "data", f"chebi_v{self.chebi_version}", "smiles_lookup.json"
        )
        if not os.path.exists(path):
            smiles_lookup = self.build_smiles_lookup()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(smiles_lookup, f, indent=4)
        else:
            print("Loading existing SMILES lookup...")
            with open(path, "r", encoding="utf-8") as f:
                smiles_lookup = json.load(f)
        return smiles_lookup

    def build_smiles_lookup(self):
        smiles_lookup = dict()
        for chebi_id, smiles in nx.get_node_attributes(
            self.chebi_graph, "smiles"
        ).items():
            if smiles is not None:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        print(
                            f"Failed to parse SMILES {smiles} for ChEBI ID {chebi_id}"
                        )
                        continue
                    canonical_smiles = Chem.MolToSmiles(mol)
                    if canonical_smiles not in smiles_lookup:
                        smiles_lookup[canonical_smiles] = []
                    # if the canonical SMILES is already in the lookup, append "different interpretation of the SMILES"
                    smiles_lookup[canonical_smiles].append(
                        (chebi_id, list(self.chebi_graph.predecessors(chebi_id)))
                    )
                except Exception as e:
                    print(
                        f"Failed to parse SMILES {smiles} for ChEBI ID {chebi_id}: {e}"
                    )
        return smiles_lookup

    @lru_cache(maxsize=100)
    def predict_smiles(self, smiles: str) -> Optional[dict]:
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical_smiles = Chem.MolToSmiles(mol)
        if canonical_smiles in self.lookup_table:
            parent_candidates = self.lookup_table[canonical_smiles]
            preds_i = dict()
            if len(parent_candidates) > 1:
                print(
                    f"Multiple matches found in ChEBI for SMILES {smiles}: {', '.join(str(chebi_id) for chebi_id, _ in parent_candidates)}"
                )
                for k in list(set(pp for _, p in parent_candidates for pp in p)):
                    preds_i[str(k)] = 1
            elif len(parent_candidates) == 1:
                chebi_id, parents = parent_candidates[0]
                for k in parents:
                    preds_i[str(k)] = 1
            else:
                preds_i = None
            return preds_i
        else:
            return None

    def predict_smiles_tuple(self, smiles_list: list[str]) -> list:
        predictions = []
        for smiles in smiles_list:
            predictions.append(self.predict_smiles(smiles))

        return predictions

    @property
    def info_text(self):
        if self._description is None:
            return "No description is available for this model."
        return self._description

    def explain_smiles(self, smiles: str) -> dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "highlights": [
                    (
                        "text",
                        "The input SMILES could not be parsed into a valid molecule.",
                    )
                ]
            }
        canonical_smiles = Chem.MolToSmiles(mol)
        if canonical_smiles not in self.lookup_table:
            return {
                "highlights": [
                    ("text", "The input SMILES does not match any ChEBI entry.")
                ]
            }
        parent_candidates = self.lookup_table[canonical_smiles]
        return {
            "highlights": [
                (
                    "text",
                    f"The ChEBI Lookup matches the canonical version of the input SMILES against ChEBI (v{self.chebi_version})."
                    f" It found {'1 match' if len(parent_candidates) == 1 else f'{len(parent_candidates)} matches'}:"
                    f" {', '.join(f'CHEBI:{chebi_id}' for chebi_id, _ in parent_candidates)}. The predicted classes are the"
                    f" parent classes of the matched ChEBI entries.",
                )
            ]
        }


if __name__ == "__main__":
    predictor = ChEBILookupPredictor("ChEBI Lookup")
    print(predictor.info_text)
    # Example usage
    smiles_list = [
        "CCO",
        "C1=CC=CC=C1" "*C(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(*)=O",
    ]  # SMILES with 251 matches in ChEBI
    predictions = predictor.predict_smiles_list(smiles_list)
    print(predictions)
