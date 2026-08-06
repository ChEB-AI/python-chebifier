import functools
import importlib.resources
import os
import pickle

import yaml
from rdkit import Chem

from chebifier.hugging_face import download_model_files


def load_chebi_graph(filename=None):
    """Load ChEBI graph from Hugging Face (if filename is None) or local file"""
    if filename is None:
        print("Loading ChEBI graph from Hugging Face...")
        file = download_model_files(
            {
                "repo_id": "chebai/chebifier",
                "repo_type": "dataset",
                "files": {"f": "chebi_graph_v252.pkl"},
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
