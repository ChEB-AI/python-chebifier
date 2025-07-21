from chebifier.ensemble.base_ensemble import BaseEnsemble
from chebifier.ensemble.weighted_majority_ensemble import (
    WMVwithF1Ensemble,
    WMVwithPPVNPVEnsemble,
)
from chebifier.prediction_models import (
    ChemlogPeptidesPredictor,
    ElectraPredictor,
    ResGatedPredictor,
    ChEBILookupPredictor,
)
from chebifier.prediction_models.c3p_predictor import C3PPredictor
from chebifier.prediction_models.chemlog_predictor import (
    ChemlogXMolecularEntityPredictor,
    ChemlogOrganoXCompoundPredictor,
)

ENSEMBLES = {
    "mv": BaseEnsemble,
    "wmv-ppvnpv": WMVwithPPVNPVEnsemble,
    "wmv-f1": WMVwithF1Ensemble,
}


MODEL_TYPES = {
    "electra": ElectraPredictor,
    "resgated": ResGatedPredictor,
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
