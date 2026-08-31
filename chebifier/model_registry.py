from chebifier.ensemble.dynamic_selection_ensemble import DynamicSelectionEnsemble
from chebifier.ensemble.learning_to_rank_ensemble import LearningToRankEnsemble
from chebifier.ensemble.voting_ensemble import (
    MajorityVotingEnsemble,
    WMVwithConfidenceEnsemble,
)
from chebifier.ensemble.weighted_majority_ensemble import WMVwithF1Ensemble
from chebifier.prediction_models import (
    ChEBILookupPredictor,
    ChemlogPeptidesPredictor,
    ElectraPredictor,
    GNNPredictor,
)
from chebifier.prediction_models.c3p_predictor import C3PPredictor
from chebifier.prediction_models.chemlog_predictor import (
    ChemlogAllPredictor,
    ChemLogLopsterClingoPredictor,
    ChemlogLopsterPredictor,
    ChemlogOrganoXCompoundPredictor,
    ChemlogXMolecularEntityPredictor,
)

ENSEMBLES = {
    "mv": MajorityVotingEnsemble,
    "wmv-conf": WMVwithConfidenceEnsemble,
    "wmv-f1": WMVwithF1Ensemble,
    "ltr": LearningToRankEnsemble,
    "des": DynamicSelectionEnsemble,
}


MODEL_TYPES = {
    "electra": ElectraPredictor,
    "resgated": GNNPredictor,
    "gat": GNNPredictor,
    "chemlog": ChemlogAllPredictor,  # combines all Chemlog predictors (chemlog_peptides, chemlog_element, chemlog_organox)
    "chemlog_peptides": ChemlogPeptidesPredictor,
    "chebi_lookup": ChEBILookupPredictor,
    "chemlog_element": ChemlogXMolecularEntityPredictor,
    "chemlog_organox": ChemlogOrganoXCompoundPredictor,
    "lopster": ChemlogLopsterPredictor,  # uses a Lopster->Python translation of the original rules, less efficient tnat lopster_clingo
    "lopster_clingo": ChemLogLopsterClingoPredictor,  # uses the Clingo solver to evaluate the rules, recommended
    "c3p": C3PPredictor,
}


common_keys = MODEL_TYPES.keys() & ENSEMBLES.keys()
assert (
    not common_keys
), f"Overlapping keys between MODEL_TYPES and ENSEMBLES: {common_keys}"
