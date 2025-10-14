from typing import TYPE_CHECKING, Optional

import torch

from .nn_predictor import NNPredictor

if TYPE_CHECKING:
    from chebai_graph.models.gat import GATGraphPred
    from chebai_graph.models.resgated import ResGatedGraphPred


class ResGatedPredictor(NNPredictor):
    def __init__(
        self,
        model_name: str,
        ckpt_path: str,
        molecular_properties,
        dataset_cls: Optional[str] = None,
        **kwargs,
    ):
        from chebai_graph.preprocessing.datasets.chebi import (
            ChEBI50GraphProperties,
            GraphPropertiesMixIn,
        )
        from chebai_graph.preprocessing.properties import MolecularProperty

        # molecular_properties is a list of class paths
        if molecular_properties is not None:
            properties = [self.load_class(prop)() for prop in molecular_properties]
            properties = sorted(
                properties, key=lambda prop: f"{prop.name}_{prop.encoder.name}"
            )
        else:
            properties = []
        for property in properties:
            property.encoder.eval = True
        self.molecular_properties = properties
        assert isinstance(self.molecular_properties, list) and all(
            isinstance(prop, MolecularProperty) for prop in self.molecular_properties
        )
        # TODO it should not be necessary to refer to the whole dataset class, disentangle dataset and molecule reading
        self.dataset_cls = (
            self.load_class(dataset_cls)
            if dataset_cls is not None
            else ChEBI50GraphProperties
        )
        self.dataset: Optional[GraphPropertiesMixIn] = self.dataset_cls(
            properties=molecular_properties
        )

        super().__init__(
            model_name, ckpt_path, reader_cls=self.dataset.READER, **kwargs
        )

        print(f"Initialised GNN model {self.model_name} (device: {self.device})")

    def load_class(self, class_path: str):
        module_path, class_name = class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    def init_model(self, ckpt_path: str, **kwargs) -> "ResGatedGraphPred":
        import torch
        from chebai_graph.models.resgated import ResGatedGraphPred

        model = ResGatedGraphPred.load_from_checkpoint(
            ckpt_path,
            map_location=torch.device(self.device),
            criterion=None,
            strict=False,
            metrics=dict(train=dict(), test=dict(), validation=dict()),
            pretrained_checkpoint=None,
        )
        model.eval()
        return model

    def read_smiles(self, smiles):
        from chebai_graph.preprocessing.datasets.chebi import GraphPropAsPerNodeType

        d = self.dataset.READER().to_data(dict(features=smiles, labels=None))
        property_data = d
        # TODO merge props into base should not be a method of a dataset (or at least static)
        for property in self.dataset.properties:
            property.encoder.eval = True
            property_value = self.reader.read_property(smiles, property)
            if property_value is None or len(property_value) == 0:
                encoded_value = None
            else:
                encoded_value = torch.stack(
                    [property.encoder.encode(v) for v in property_value]
                )
                if len(encoded_value.shape) == 3:
                    encoded_value = encoded_value.squeeze(0)
            property_data[property.name] = encoded_value
        # Augmented graphs need an additional argument
        if isinstance(self.dataset, GraphPropAsPerNodeType):
            d["features"] = self.dataset._merge_props_into_base(
                property_data, max_len_node_properties=self.model.gnn.in_channels
            )
        else:
            d["features"] = self.dataset._merge_props_into_base(property_data)
        return d


class GATPredictor(ResGatedPredictor):

    def init_model(self, ckpt_path: str, **kwargs) -> "GATGraphPred":
        import torch
        from chebai_graph.models.gat import GATGraphPred

        model = GATGraphPred.load_from_checkpoint(
            ckpt_path,
            map_location=torch.device(self.device),
            criterion=None,
            strict=False,
            metrics=dict(train=dict(), test=dict(), validation=dict()),
            pretrained_checkpoint=None,
        )
        model.eval()
        return model
