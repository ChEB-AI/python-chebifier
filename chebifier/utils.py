import os

import networkx as nx
import requests
import fastobo
from chebifier.hugging_face import download_model_files
import pickle


def load_chebi_graph(filename=None):
    """Load ChEBI graph from Hugging Face (if filename is None) or local file"""
    if filename is None:
        print("Loading ChEBI graph from Hugging Face...")
        file = download_model_files(
            {
                "repo_id": "chebai/chebifier",
                "repo_type": "dataset",
                "files": {"f": "chebi_graph.pkl"},
            }
        )["f"]
    else:
        print(f"Loading ChEBI graph from local {filename}...")
        file = filename
    return pickle.load(open(file, "rb"))


def term_callback(doc):
    """Similar to the chebai function, but reduced to the necessary fields. Also, ChEBI IDs are strings"""
    parents = []
    name = None
    smiles = None
    for clause in doc:
        if isinstance(clause, fastobo.term.PropertyValueClause):
            t = clause.property_value
            if str(t.relation) == "http://purl.obolibrary.org/obo/chebi/smiles":
                assert smiles is None
                smiles = t.value
        # in older chebi versions, smiles strings are synonyms
        # e.g. synonym: "[F-].[Na+]" RELATED SMILES [ChEBI]
        elif isinstance(clause, fastobo.term.SynonymClause):
            if "SMILES" in clause.raw_value():
                assert smiles is None
                smiles = clause.raw_value().split('"')[1]
        elif isinstance(clause, fastobo.term.IsAClause):
            chebi_id = str(clause.term)
            chebi_id = chebi_id[chebi_id.index(":") + 1 :]
            parents.append(chebi_id)
        elif isinstance(clause, fastobo.term.NameClause):
            name = str(clause.name)

        if isinstance(clause, fastobo.term.IsObsoleteClause):
            if clause.obsolete:
                # if the term document contains clause as obsolete as true, skips this document.
                return False
    chebi_id = str(doc.id)
    chebi_id = chebi_id[chebi_id.index(":") + 1 :]
    return {
        "id": chebi_id,
        "parents": parents,
        "name": name,
        "smiles": smiles,
    }


def build_chebi_graph(chebi_version=241):
    """Creates a networkx graph for the ChEBI hierarchy. Usually, you don't want to call this function directly, but rather use the `load_chebi_graph` function."""
    chebi_path = os.path.join("data", f"chebi_v{chebi_version}", "chebi.obo")
    os.makedirs(os.path.join("data", f"chebi_v{chebi_version}"), exist_ok=True)
    if not os.path.exists(chebi_path):
        url = f"http://purl.obolibrary.org/obo/chebi/{chebi_version}/chebi.obo"
        r = requests.get(url, allow_redirects=True)
        open(chebi_path, "wb").write(r.content)
    with open(chebi_path, encoding="utf-8") as chebi:
        chebi = "\n".join(line for line in chebi if not line.startswith("xref:"))

    elements = []
    for term_doc in fastobo.loads(chebi):
        if (
            term_doc
            and isinstance(term_doc.id, fastobo.id.PrefixedIdent)
            and term_doc.id.prefix == "CHEBI"
        ):
            term_dict = term_callback(term_doc)
            if term_dict:
                elements.append(term_dict)

    g = nx.DiGraph()
    for n in elements:
        g.add_node(n["id"], **n)

    # Only take the edges which connect the existing nodes, to avoid internal creation of obsolete nodes
    # https://github.com/ChEB-AI/python-chebai/pull/55#issuecomment-2386654142
    g.add_edges_from(
        [(p, q["id"]) for q in elements for p in q["parents"] if g.has_node(p)]
    )
    return nx.transitive_closure_dag(g)


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


if __name__ == "__main__":
    # chebi_graph = build_chebi_graph(chebi_version=241)
    # save the graph to a file
    # pickle.dump(chebi_graph, open("chebi_graph.pkl", "wb"))
    chebi_graph = load_chebi_graph()
    print(chebi_graph)
