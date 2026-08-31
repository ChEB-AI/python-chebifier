import json
import os
from typing import Optional

from chebi_utils.read_molecule import smiles_or_inchi_to_mol
from rdkit import Chem

from chebifier import modelwise_smiles_lru_cache
from chebifier.prediction_models import BasePredictor
from chebifier.utils import get_superclasses, load_chebi_graph


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
        self.chebi_graph = kwargs.get("chebi_graph", load_chebi_graph())
        self.lookup_table = self.get_inchikey_lookup()

    def get_inchikey_lookup(self):
        path = os.path.join(
            "data", f"chebi_v{self.chebi_version}", "inchikey_lookup.json"
        )
        if not os.path.exists(path):
            inchikey_lookup = self.build_inchikey_lookup()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(inchikey_lookup, f, indent=4)
        else:
            print("Loading existing InChIKey lookup...")
            with open(path, "r", encoding="utf-8") as f:
                inchikey_lookup = json.load(f)
        return inchikey_lookup

    def build_inchikey_lookup(self):
        import networkx as nx

        inchikey_lookup = dict()
        for chebi_id, smiles in nx.get_node_attributes(
            self.chebi_graph, "smiles"
        ).items():
            if smiles is not None:
                try:
                    mol = smiles_or_inchi_to_mol(smiles)
                    if mol is None:
                        print(
                            f"Failed to parse SMILES {smiles} for ChEBI ID {chebi_id}"
                        )
                        continue
                    inchikey = Chem.MolToInchiKey(mol)
                    if inchikey not in inchikey_lookup:
                        inchikey_lookup[inchikey] = []
                    inchikey_lookup[inchikey].append(
                        (chebi_id, list(get_superclasses(self.chebi_graph, chebi_id)))
                    )
                except Exception as e:
                    print(
                        f"Failed to parse SMILES {smiles} for ChEBI ID {chebi_id}: {e}"
                    )
        return inchikey_lookup

    def predict(self, smiles: str | Chem.Mol) -> Optional[dict]:
        if not smiles:
            return None
        mol = smiles if isinstance(smiles, Chem.Mol) else smiles_or_inchi_to_mol(smiles)
        if mol is None:
            return None
        inchikey = Chem.MolToInchiKey(mol)
        if inchikey in self.lookup_table:
            parent_candidates = self.lookup_table[inchikey]
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

    @modelwise_smiles_lru_cache.batch_decorator
    def predict_list(self, smiles_list: list[str]) -> list:
        predictions = []
        for smiles in smiles_list:
            predictions.append(self.predict(smiles))

        return predictions

    @property
    def info_text(self):
        if self._description is None:
            return "No description is available for this model."
        return self._description

    def class_name(self, chebi_id) -> Optional[str]:
        chebi_id = str(chebi_id)
        if chebi_id not in self.chebi_graph:
            return None
        return self.chebi_graph.nodes[chebi_id].get("name")

    def explain_smiles(self, smiles: str) -> dict:
        mol = smiles_or_inchi_to_mol(smiles)
        if mol is None:
            return {
                "chebi_ids": [],
                "chebi_names": [],
                "highlights": [
                    (
                        "text",
                        "The input SMILES could not be parsed into a valid molecule.",
                    )
                ],
            }
        inchikey = Chem.MolToInchiKey(mol)
        if inchikey not in self.lookup_table:
            return {
                "chebi_ids": [],
                "chebi_names": [],
                "highlights": [
                    ("text", "The input SMILES does not match any ChEBI entry.")
                ],
            }
        parent_candidates = self.lookup_table[inchikey]
        matches = [
            (str(chebi_id), self.class_name(chebi_id))
            for chebi_id, _ in parent_candidates
        ]
        return {
            "chebi_ids": [chebi_id for chebi_id, _ in matches],
            "chebi_names": [name for _, name in matches],
            "highlights": [
                (
                    "text",
                    f"The ChEBI Lookup matches the InChIKey of the input structure against ChEBI (v{self.chebi_version})."
                    f" It found {'1 match' if len(matches) == 1 else f'{len(matches)} matches'}:"
                    f" {', '.join(f'CHEBI:{cid} ({name})' if name else f'CHEBI:{cid}' for cid, name in matches)}."
                    f" The predicted classes are the parent classes of the matched ChEBI entries.",
                )
            ],
        }


if __name__ == "__main__":
    predictor = ChEBILookupPredictor("ChEBI Lookup")
    print(predictor.info_text)
    # Example usage
    smiles_list = [
        "CCO",
        "C1=CC=CC=C1",
        "*C(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(*)=O",
    ]  # SMILES with 251 matches in ChEBI
    predictions = predictor.predict_list(smiles_list)
    print(predictions)
