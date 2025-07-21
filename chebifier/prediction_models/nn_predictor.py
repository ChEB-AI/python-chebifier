from functools import lru_cache

import numpy as np
import torch
import tqdm
from rdkit import Chem

from .base_predictor import BasePredictor


class NNPredictor(BasePredictor):
    def __init__(
        self,
        model_name: str,
        ckpt_path: str,
        reader_cls,
        target_labels_path: str,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.reader_cls = reader_cls

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.init_model(ckpt_path=ckpt_path)
        self.target_labels = [
            line.strip() for line in open(target_labels_path, encoding="utf-8")
        ]
        self.batch_size = kwargs.get("batch_size", 1)

    def init_model(self, ckpt_path: str, **kwargs):
        raise NotImplementedError(
            "Model initialization must be implemented in subclasses."
        )

    def calculate_results(self, batch):
        collator = self.reader_cls.COLLATOR()
        dat = self.model._process_batch(collator(batch).to(self.device), 0)
        return self.model(dat, **dat["model_kwargs"])

    def batchify(self, batch):
        cache = []
        for r in batch:
            cache.append(r)
            if len(cache) >= self.batch_size:
                yield cache
                cache = []
        if cache:
            yield cache

    def read_smiles(self, smiles):
        reader = self.reader_cls()
        d = reader.to_data(dict(features=smiles, labels=None))
        return d

    @lru_cache(maxsize=100)
    def predict_smiles_tuple(self, smiles_list: tuple[str]) -> list:
        """Returns a list with the length of smiles_list, each element is either None (=failure) or a dictionary
        Of classes and predicted values."""
        token_dicts = []
        could_not_parse = []
        index_map = dict()
        for i, smiles in enumerate(smiles_list):
            if not smiles:
                print(
                    f"Model {self.model_name} received a missing SMILES string at position {i}."
                )
                could_not_parse.append(i)
                continue
            try:
                d = self.read_smiles(smiles)
                # This is just for sanity checks
                rdmol = Chem.MolFromSmiles(smiles, sanitize=False)
                if rdmol is None:
                    print(
                        f"Model {self.model_name} received a SMILES string RDKit can't read at position {i}: {smiles}"
                    )
                    could_not_parse.append(i)
                    continue
            except Exception:
                could_not_parse.append(i)
                print(
                    f"Model {self.model_name} failed to parse a SMILES string at position {i}: {smiles}"
                )
                continue
            index_map[i] = len(token_dicts)
            token_dicts.append(d)
        results = []
        if len(token_dicts) > 0:
            for batch in tqdm.tqdm(
                self.batchify(token_dicts),
                desc=f"{self.model_name}",
                total=len(token_dicts) // self.batch_size,
            ):
                result = self.calculate_results(batch)
                if isinstance(result, dict) and "logits" in result:
                    result = result["logits"]
                results += torch.sigmoid(result).cpu().detach().tolist()
            results = np.stack(results, axis=0)
            preds = [
                (
                    {
                        self.target_labels[j]: p
                        for j, p in enumerate(results[index_map[i]])
                    }
                    if i not in could_not_parse
                    else None
                )
                for i in range(len(smiles_list))
            ]
            return preds
        else:
            return [None for _ in smiles_list]
