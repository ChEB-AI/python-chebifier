import networkx as nx
from rdkit import Chem

from chebifier.utils import get_superclasses, to_mol, to_smiles


def make_graph():
    graph = nx.DiGraph()
    # is-a edges point from child to parent
    graph.add_edge("dipeptide", "peptide", relation="is_a")
    graph.add_edge("peptide", "molecular entity", relation="is_a")
    graph.add_edge("dipeptide", "drug", relation="has_role")
    return graph


def test_superclasses_follow_isa_edges_upwards_and_transitively():
    assert set(get_superclasses(make_graph(), "dipeptide")) == {
        "peptide",
        "molecular entity",
    }


def test_superclasses_ignore_non_subsumption_relations():
    assert "drug" not in get_superclasses(make_graph(), "dipeptide")


def test_a_class_without_superclasses_has_none():
    assert get_superclasses(make_graph(), "molecular entity") == ()
    assert get_superclasses(make_graph(), "not in chebi") == ()


def test_to_mol_kekulises_without_touching_the_input():
    original = Chem.MolFromSmiles("c1ccccc1")

    converted = to_mol(original)

    assert not any(
        bond.GetBondType() == Chem.BondType.AROMATIC for bond in converted.GetBonds()
    )
    assert all(
        bond.GetBondType() == Chem.BondType.AROMATIC for bond in original.GetBonds()
    )


def test_to_mol_and_to_smiles_pass_the_other_representation_through():
    assert to_smiles("CCO") == "CCO"
    assert to_smiles(Chem.MolFromSmiles("OCC")) == "CCO"
    assert Chem.MolToSmiles(to_mol("CCO")) == "CCO"
