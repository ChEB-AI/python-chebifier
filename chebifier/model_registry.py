from chebifier.ensemble.base_ensemble import BaseEnsemble
from chebifier.ensemble.weighted_majority_ensemble import (
    WMVwithF1Ensemble,
    WMVwithPPVNPVEnsemble,
)
from chebifier.prediction_models import (
    ChEBILookupPredictor,
    ChemlogPeptidesPredictor,
    ElectraPredictor,
    GNNPredictor,
)
from chebifier.prediction_models.c3p_predictor import C3PPredictor
from chebifier.prediction_models.chemlog_predictor import (
    ChemlogAllPredictor,
    ChemlogOrganoXCompoundPredictor,
    ChemlogXMolecularEntityPredictor,
)
from chebifier.prediction_models.gnn_predictor import GATPredictor

ENSEMBLES = {
    "mv": BaseEnsemble,
    "wmv-ppvnpv": WMVwithPPVNPVEnsemble,
    "wmv-f1": WMVwithF1Ensemble,
}


MODEL_TYPES = {
    "electra": ElectraPredictor,
    "resgated": GNNPredictor,
    "gat": GATPredictor,
    "chemlog": ChemlogAllPredictor,
    "chemlog_peptides": ChemlogPeptidesPredictor,
    "chebi_lookup": ChEBILookupPredictor,
    "chemlog_element": ChemlogXMolecularEntityPredictor,
    "chemlog_organox": ChemlogOrganoXCompoundPredictor,
    "c3p": C3PPredictor,
}


common_keys = MODEL_TYPES.keys() & ENSEMBLES.keys()
assert (
    not common_keys
), f"Overlapping keys between MODEL_TYPES and ENSEMBLES: {common_keys}"
