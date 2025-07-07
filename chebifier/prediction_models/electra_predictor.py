import numpy as np

from chebifier.prediction_models.nn_predictor import NNPredictor
from chebai.models.electra import Electra
from chebai.preprocessing.reader import ChemDataReader, EMBEDDING_OFFSET

def build_graph_from_attention(att, node_labels, token_labels, threshold=0.0):
    n_nodes = len(node_labels)
    return dict(
        nodes=[
            dict(
                label=token_labels[n],
                id=f"{group}_{i}",
                fixed=dict(x=True, y=True),
                y=100 * int(group == "r"),
                x=30 * i,
                group=group,
            )
            for i, n in enumerate([0] + node_labels)
            for group in ("l", "r")
        ],
        edges=[
            {
                "from": f"l_{i}",
                "to": f"r_{j}",
                "color": {"opacity": att[i, j].item()},
                "smooth": False,
                "physics": False,
            }
            for i in range(n_nodes)
            for j in range(n_nodes)
            if att[i, j] > threshold
        ],
    )


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

    def explain_smiles(self, smiles) -> dict:
        reader = self.reader_cls()
        token_dict = reader.to_data(dict(features=smiles, labels=None))
        tokens = np.array(token_dict["features"]).astype(int).tolist()
        result = self.calculate_results([token_dict])

        token_labels = (
                ["[CLR]"] + [None for _ in range(EMBEDDING_OFFSET - 1)] + list(reader.cache.keys())
        )

        graphs = [
            [
                build_graph_from_attention(
                    a[0, i], tokens, token_labels, threshold=0.1
                )
                for i in range(a.shape[1])
            ]
            for a in result["attentions"]
        ]
        return {"graphs": graphs}

