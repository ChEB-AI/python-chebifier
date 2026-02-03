from .nn_predictor import NNPredictor


class GNNPredictor(NNPredictor):
    def __init__(
        self,
        model_name: str,
        ckpt_path: str,
        **kwargs,
    ):
        super().__init__(model_name, ckpt_path, **kwargs)
        print(
            f"Initialised GNN model {self.model_name} (device: {self.predictor.device})"
        )
