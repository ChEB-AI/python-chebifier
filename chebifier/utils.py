import functools
import os
import pickle

import networkx as nx
import numpy as np
import yaml
from chebi_utils.obo_extractor import get_hierarchy_subgraph
from chebi_utils.read_molecule import smiles_or_inchi_to_mol
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


DEFAULT_CONFIGS = {
    "eval": "config_26-09_eval.yml",
    "web": "config_26-09_web.yml",
}


def load_ensemble_config(ensemble_config=None):
    """Resolve an ensemble configuration to a config dict.

    'web' and 'eval' are downloaded from the chebifier Hugging Face dataset, anything else is
    treated as a path to a config file. None defaults to 'web'.
    """
    if ensemble_config is None:
        ensemble_config = "web"
    if ensemble_config in DEFAULT_CONFIGS:
        filename = DEFAULT_CONFIGS[ensemble_config]
        print(
            f"Loading '{ensemble_config}' ensemble configuration ({filename}) from Hugging Face..."
        )
        path = download_model_files(
            {
                "repo_id": "chebai/chebifier",
                "repo_type": "dataset",
                "files": {"config": filename},
            }
        )["config"]
    else:
        print(f"Loading ensemble configuration from {ensemble_config}")
        path = ensemble_config
    with open(path, "r") as f:
        return yaml.safe_load(f)


DEFAULT_ENSEMBLE_CALIBRATION = "wmv-f1-3star-symbolic"


def download_ensemble_calibration(subfolder=DEFAULT_ENSEMBLE_CALIBRATION):
    """Download the calibration of the standard ensemble from Hugging Face, returning its local path."""
    from huggingface_hub import snapshot_download

    print(f"Downloading ensemble calibration '{subfolder}' from Hugging Face...")
    root = snapshot_download(
        "chebai/chebifier", repo_type="dataset", allow_patterns=f"{subfolder}/*"
    )
    return os.path.join(root, subfolder)


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


def to_mol(molecule: str | Chem.Mol):
    """Molecules reach a predictor either as SMILES/InChI or as RDKit molecules (the evaluation
    datasets store the latter). Rule-based classifiers expect kekulised molecules, and Kekulize
    works in place, so a molecule that is not ours to modify is copied first."""
    if not isinstance(molecule, Chem.Mol):
        molecule = smiles_or_inchi_to_mol(molecule)
        if molecule is None:
            return None
    else:
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


def labels_from_graph(chebi_ids, classes, chebi_graph) -> np.ndarray:
    cls_to_idx = {str(cls): idx for idx, cls in enumerate(classes)}
    labels = np.zeros((len(chebi_ids), len(classes)), dtype=bool)
    for row, chebi_id in enumerate(chebi_ids):
        chebi_id = str(chebi_id)
        # a molecule is a member of its own class, which is not among its superclasses
        columns = [
            cls_to_idx[cls]
            for cls in (chebi_id,) + get_superclasses(chebi_graph, chebi_id)
            if cls in cls_to_idx
        ]
        labels[row, columns] = True
    return labels
