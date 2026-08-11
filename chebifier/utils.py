import functools
import importlib.resources
import os
import pickle

import networkx as nx
import yaml
from chebi_utils.obo_extractor import get_hierarchy_subgraph
from rdkit import Chem

from chebifier.hugging_face import download_model_files

CHEBI_VERSION = 252


def load_chebi_graph(filename=None):
    """Load ChEBI graph from Hugging Face (if filename is None) or local file"""
    if filename is None:
        print("Loading ChEBI graph from Hugging Face...")
        file = download_model_files(
            {
                "repo_id": "chebai/chebifier",
                "repo_type": "dataset",
                "files": {"f": f"chebi_graph_v{CHEBI_VERSION}.pkl"},
            }
        )["f"]
    else:
        print(f"Loading ChEBI graph from local {filename}...")
        file = filename
    return pickle.load(open(file, "rb"))


def get_disjoint_files():
    """Gets local disjointness files if they are present in the right location, otherwise downloads them from Hugging Face."""
    local_disjoint_files = [
        os.path.join("data", "disjoint_chebi.csv"),
        os.path.join("data", "disjoint_additional.csv"),
    ]
    disjoint_files = []
    for file in local_disjoint_files:
        if os.path.isfile(file):
            disjoint_files.append(file)
        else:
            print(
                f"Disjoint axiom file {file} not found. Loading from huggingface instead..."
            )

            disjoint_files.append(
                download_model_files(
                    {
                        "repo_id": "chebai/chebifier",
                        "repo_type": "dataset",
                        "files": {"disjoint_file": os.path.basename(file)},
                    }
                )["disjoint_file"]
            )
    return disjoint_files


def get_default_configs():
    default_config_name = "ensemble.yml"
    print(f"Using default ensemble configuration from {default_config_name}")
    with (
        importlib.resources.files("chebifier")
        .joinpath(default_config_name)
        .open("r") as f
    ):
        return yaml.safe_load(f)


def process_config(config, model_registry):
    new_config = {}
    for model_name, entry in config.items():
        if "load_model" in entry:
            if entry["load_model"] not in model_registry:
                raise ValueError(
                    f"Model {entry['load_model']} not found in model registry. "
                    f"Available models are: {','.join(model_registry.keys())}."
                )
            new_config[model_name] = {**model_registry[entry["load_model"]], **entry}
        else:
            new_config[model_name] = entry
    return new_config


@functools.lru_cache(maxsize=128)
def _smiles_to_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is not None:
        # turn aromatic bond types into single/double
        try:
            Chem.Kekulize(mol)
        except Chem.KekulizeException as e:
            print(f"Failed to Kekulize {smiles}: {e}")
    return mol


def to_mol(molecule: str | Chem.Mol):
    """Molecules reach a predictor either as SMILES or as RDKit molecules (the evaluation datasets
    store the latter). Rule-based classifiers expect kekulised molecules, and Kekulize works in
    place, so a molecule that is not ours to modify is copied first."""
    if not isinstance(molecule, Chem.Mol):
        return _smiles_to_mol(molecule)
    molecule = Chem.Mol(molecule)
    try:
        Chem.Kekulize(molecule)
    except Chem.KekulizeException as e:
        print(f"Failed to Kekulize {Chem.MolToSmiles(molecule)}: {e}")
    return molecule


def to_smiles(molecule: str | Chem.Mol) -> str:
    return Chem.MolToSmiles(molecule) if isinstance(molecule, Chem.Mol) else molecule


@functools.lru_cache(maxsize=2)
def _isa_graph(chebi_graph):
    return get_hierarchy_subgraph(chebi_graph)


@functools.lru_cache(maxsize=None)
def get_superclasses(chebi_graph, chebi_id: str) -> tuple[str, ...]:
    """All transitive superclasses of a ChEBI class.

    is-a edges point from child to parent, and the graph also carries non-subsumption relations
    (has role, conjugate acid/base, ...), so the superclasses of a node are neither its
    predecessors nor all of its successors.
    """
    isa_graph = _isa_graph(chebi_graph)
    if chebi_id not in isa_graph:
        return ()
    return tuple(str(cls) for cls in nx.descendants(isa_graph, chebi_id))
