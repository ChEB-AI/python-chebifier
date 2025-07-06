from chebifier.ensemble.base_ensemble import BaseEnsemble
from chebifier.ensemble.weighted_majority_ensemble import (
    WMVwithF1Ensemble,
    WMVwithPPVNPVEnsemble,
)
from chebifier.prediction_models import (
    ChemLogPredictor,
    ElectraPredictor,
    ResGatedPredictor,
)

ENSEMBLES = {
    "en_mv": BaseEnsemble,
    "en_wmv-ppvnpv": WMVwithPPVNPVEnsemble,
    "en_wmv-f1": WMVwithF1Ensemble,
}


MODEL_TYPES = {
    "electra": ElectraPredictor,
    "resgated": ResGatedPredictor,
    "chemlog": ChemLogPredictor,
}


common_keys = MODEL_TYPES.keys() & ENSEMBLES.keys()
assert (
    not common_keys
), f"Overlapping keys between MODEL_TYPES and ENSEMBLES: {common_keys}"
