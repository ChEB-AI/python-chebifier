from chebifier.prediction_models.nn_predictor import NNPredictor
from chebai.models.electra import Electra
from chebai.preprocessing.reader import ChemDataReader


class ElectraPredictor(NNPredictor):

    def __init__(self, model_name: str, ckpt_path: str, **kwargs):
        super().__init__(model_name, ckpt_path, reader_cls=ChemDataReader, **kwargs)
        print(f"Initialised Electra model {self.model_name} (device: {self.device})")

    def init_model(self, ckpt_path: str, **kwargs) -> Electra:
        model = Electra.load_from_checkpoint(
            ckpt_path,
            map_location=self.device,
            criterion=None, strict=False,
            metrics=dict(train=dict(), test=dict(), validation=dict()), pretrained_checkpoint=None
        )
        model.eval()
        return model


