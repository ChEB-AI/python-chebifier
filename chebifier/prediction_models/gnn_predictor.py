from chebifier.prediction_models.nn_predictor import NNPredictor
import chebai_graph.preprocessing.properties as p
import torch
from chebai_graph.models.graph import ResGatedGraphConvNetGraphPred
from chebai_graph.preprocessing.reader import GraphPropertyReader
from chebai_graph.preprocessing.property_encoder import IndexEncoder, OneHotEncoder
from torch_geometric.data.data import Data as GeomData


class ResGatedPredictor(NNPredictor):

    def __init__(self, model_name: str, ckpt_path: str, molecular_properties, **kwargs):
        super().__init__(model_name, ckpt_path, reader_cls=GraphPropertyReader, **kwargs)
        # molecular_properties is a list of class paths
        if molecular_properties is not None:
            properties = [self.load_class(prop)() for prop in molecular_properties]
            properties = sorted(
                properties, key=lambda prop: f"{prop.name}_{prop.encoder.name}"
            )
        else:
            properties = []
        self.molecular_properties = properties
        assert isinstance(self.molecular_properties, list) and all(
            isinstance(prop, p.MolecularProperty) for prop in self.molecular_properties
        )
        print(f"Initialised GNN model {self.model_name} (device: {self.device})")

    def load_class(self, class_path: str):
        module_path, class_name = class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    def init_model(self, ckpt_path: str, **kwargs) -> ResGatedGraphConvNetGraphPred:
        model = ResGatedGraphConvNetGraphPred.load_from_checkpoint(
            ckpt_path, map_location=torch.device(self.device), criterion=None, strict=False, input_dim=128,
            metrics=dict(train=dict(), test=dict(), validation=dict()), pretrained_checkpoint=None,
            config={"in_length": 256, "hidden_length": 512, "dropout_rate": 0.1, "n_conv_layers": 3,
                    "n_linear_layers": 3, "n_atom_properties": 158, "n_bond_properties": 7,
                    "n_molecule_properties": 200})
        model.eval()
        return model

    def read_smiles(self, smiles):
        reader = self.reader_cls()
        d = reader.to_data(dict(features=smiles, labels=None))
        geom_data = d["features"]
        edge_attr = geom_data.edge_attr
        x = geom_data.x
        molecule_attr = torch.empty((1, 0))
        for prop in self.molecular_properties:
            property_values = reader.read_property(smiles, prop)
            encoded_values = []
            for value in property_values:
                # cant use standard encode for index encoder because model has been trained on a certain range of values
                # use default value if we meet an unseen value
                if isinstance(prop.encoder, IndexEncoder):
                    if str(value) in prop.encoder.cache:
                        index = prop.encoder.cache.index(str(value)) + prop.encoder.offset
                    else:
                        index = 0
                        print(f"Unknown property value {value} for property {prop} at smiles {smiles}")
                    if isinstance(prop.encoder, OneHotEncoder):
                        encoded_values.append(torch.nn.functional.one_hot(
                            torch.tensor(index), num_classes=prop.encoder.get_encoding_length()
                        ))
                    else:
                        encoded_values.append(torch.tensor([index]))

                else:
                    encoded_values.append(prop.encoder.encode(value))
            if len(encoded_values) > 0:
                encoded_values = torch.stack(encoded_values)

            if isinstance(encoded_values, torch.Tensor):
                if len(encoded_values.size()) == 0:
                    encoded_values = encoded_values.unsqueeze(0)
                if len(encoded_values.size()) == 1:
                    encoded_values = encoded_values.unsqueeze(1)
            else:
                encoded_values = torch.zeros(
                    (0, prop.encoder.get_encoding_length())
                )
            if isinstance(prop, p.AtomProperty):
                x = torch.cat([x, encoded_values], dim=1)
            elif isinstance(prop, p.BondProperty):
                edge_attr = torch.cat([edge_attr, encoded_values], dim=1)
            else:
                molecule_attr = torch.cat([molecule_attr, encoded_values[0]], dim=1)

        d["features"] = GeomData(
            x=x,
            edge_index=geom_data.edge_index,
            edge_attr=edge_attr,
            molecule_attr=molecule_attr,
        )
        return d