import os
import tempfile

import networkx as nx
import numpy as np
import pytest
import yaml
from rdkit import Chem

from chebifier import modelwise_smiles_lru_cache
from chebifier.cli import build_base_learners
from chebifier.predict import get_base_learner_predictions
from chebifier.utils import get_superclasses, load_chebi_graph

pytestmark = pytest.mark.skipif(
    os.environ.get("CHEBIFIER_INTEGRATION") != "1",
    reason="loads real models, set CHEBIFIER_INTEGRATION=1 to run",
)

CHEBI_GRAPH_FILE = "data/chebi_graph_v252.pkl"
L_ALANINE = "16977"
ALA_ALA = "72816"
HEXADECANOIC_ACID = "15756"
CHLOROFORM = "35255"

MODELS = {
    "gat": ({"load_model": "gat_chebi25_v252"}, L_ALANINE),
    "gat-aug": ({"load_model": "gat-aug_chebi25_v252"}, L_ALANINE),
    "resgated": ({"load_model": "resgated_chebi25_v252"}, L_ALANINE),
    "resgated-aug": ({"load_model": "resgated-aug_chebi25_v252"}, L_ALANINE),
    "c3p": ({"type": "c3p"}, L_ALANINE),
    "chebi_lookup": ({"type": "chebi_lookup"}, L_ALANINE),
    "chemlog": ({"type": "chemlog"}, ALA_ALA),
    "chemlog_peptides": ({"type": "chemlog_peptides"}, ALA_ALA),
    "chemlog_element": ({"type": "chemlog_element"}, CHLOROFORM),
    "chemlog_organox": ({"type": "chemlog_organox"}, CHLOROFORM),
    "lopster": ({"type": "lopster_clingo"}, HEXADECANOIC_ACID),
}

INVALID_INPUTS = ["C1CC(", "not-a-smiles", None]


@pytest.fixture(scope="module")
def chebi_graph():
    return load_chebi_graph(CHEBI_GRAPH_FILE)


@pytest.fixture(scope="module")
def models():
    loaded = {}

    def get(name):
        if name not in loaded:
            path = os.path.join(tempfile.mkdtemp(), "config.yml")
            with open(path, "w") as f:
                yaml.safe_dump({name: MODELS[name][0]}, f)
            loaded[name] = build_base_learners(path, chebi_graph_file=CHEBI_GRAPH_FILE)[
                name
            ]
        return loaded[name]

    return get


@pytest.fixture(autouse=True)
def clear_cache():
    modelwise_smiles_lru_cache.clear()
    yield
    modelwise_smiles_lru_cache.clear()


def molecule(chebi_graph, name):
    chebi_id = MODELS[name][1]
    smiles = nx.get_node_attributes(chebi_graph, "smiles")[chebi_id]
    expected = {chebi_id, *get_superclasses(chebi_graph, chebi_id)}
    return smiles, expected


def positives(classes, row):
    return {str(c) for c, s in zip(classes, row) if s > 0.5}


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("as_mol", [False, True], ids=["smiles", "mol"])
def test_valid_molecule_gets_realistic_prediction(name, as_mol, models, chebi_graph):
    smiles, expected = molecule(chebi_graph, name)
    inp = Chem.MolFromSmiles(smiles) if as_mol else smiles
    classes, scores = models(name).predict_dense([inp])
    predicted = positives(classes, scores[0])
    assert not np.isnan(scores[0]).all()
    assert predicted & expected, f"{name} predicted {sorted(predicted)[:20]}"


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("invalid", INVALID_INPUTS, ids=["unclosed", "garbage", "none"])
def test_invalid_molecule_is_reported_as_missing(name, invalid, models, chebi_graph):
    smiles, expected = molecule(chebi_graph, name)
    classes, scores = get_base_learner_predictions(
        {name: models(name)}, [smiles, invalid]
    )[name]
    assert positives(classes, scores[0]) & expected
    assert np.isnan(scores[1]).all()


@pytest.mark.parametrize("name", MODELS)
def test_prediction_leaves_molecule_unchanged(name, models, chebi_graph):
    smiles, _ = molecule(chebi_graph, name)
    mol = Chem.MolFromSmiles(smiles)
    before = Chem.MolToSmiles(mol)
    models(name).predict_dense([mol])
    assert Chem.MolToSmiles(mol) == before
    assert all(atom.GetAtomMapNum() == 0 for atom in mol.GetAtoms())
