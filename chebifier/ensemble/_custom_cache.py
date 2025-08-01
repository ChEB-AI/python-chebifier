import os
import pickle
import threading
from collections import OrderedDict
from collections.abc import Iterable
from functools import wraps
from typing import Any, Callable


class PerSmilesPerModelLRUCache:
    def __init__(self, max_size: int = 100, persist_path: str | None = None):
        self._cache = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._persist_path = persist_path

        self.hits = 0
        self.misses = 0

        if self._persist_path:
            self._load_cache()

    def get(self, smiles: str, model_name: str) -> Any | None:
        key = (smiles, model_name)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            else:
                self.misses += 1
                return None

    def set(self, smiles: str, model_name: str, value: Any) -> None:
        assert value is not None, "Value must not be None"
        key = (smiles, model_name)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        self._save_cache()
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            if self._persist_path and os.path.exists(self._persist_path):
                os.remove(self._persist_path)

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses}

    def batch_decorator(self, func: Callable) -> Callable:
        """Decorator for class methods that accept a batch of SMILES as a tuple,
        and want caching per (smiles, model_name) combination.
        """

        @wraps(func)
        def wrapper(instance, smiles_list: list[str]):
            assert isinstance(smiles_list, list), "smiles_list must be a list."
            model_name = getattr(instance, "model_name", None)
            assert model_name is not None, "Instance must have a model_name attribute."

            results = []
            missing_smiles = []
            missing_indices = []

            # First: try to fetch all from cache
            for i, smiles in enumerate(smiles_list):
                result = self.get(smiles=smiles, model_name=model_name)
                if result is not None:
                    results.append((i, result))  # save index for reordering
                else:
                    missing_smiles.append(smiles)
                    missing_indices.append(i)

            # If some are missing, call original function
            if missing_smiles:
                new_results = func(instance, tuple(missing_smiles))
                assert isinstance(
                    new_results, Iterable
                ), "Function must return an  Iterable."
                # Save to cache and append
                for smiles, prediction in zip(missing_smiles, new_results):
                    if prediction is not None:
                        self.set(smiles, model_name, prediction)
                    results.append((missing_indices.pop(0), prediction))

            # Reorder results to match original indices
            results.sort(key=lambda x: x[0])  # sort by index
            ordered = [result for _, result in results]
            return ordered

        return wrapper

    def __len__(self):
        with self._lock:
            return len(self._cache)

    def __repr__(self):
        return self._cache.__repr__()

    def save(self):
        self._save_cache()

    def load(self):
        self._load_cache()

    def _save_cache(self) -> None:
        """Serialize the cache to disk."""
        if self._persist_path:
            try:
                with open(self._persist_path, "wb") as f:
                    pickle.dump(self._cache, f)
            except Exception as e:
                print(f"[Cache Save Error] {e}")

    def _load_cache(self) -> None:
        """Load the cache from disk."""
        if os.path.exists(self._persist_path):
            try:
                with open(self._persist_path, "rb") as f:
                    loaded = pickle.load(f)
                    if isinstance(loaded, OrderedDict):
                        self._cache = loaded
            except Exception as e:
                print(f"[Cache Load Error] {e}")


if __name__ == "__main__":
    # cache will persist across runs in "cache.pkl"
    cache = PerSmilesPerModelLRUCache(max_size=50)

    class ExamplePredictor:
        model_name = "example_model"

        @cache.batch_decorator
        def predict(self, smiles_list: tuple[str]) -> list[dict]:
            # Simulate a prediction function
            return [{"prediction": hash(smiles) % 100} for smiles in smiles_list]

    # Create an instance of the predictor
    predictor = ExamplePredictor()

    # Prediction set 1 — new model, all should be cache misses
    predictor.model_name = "example_model"
    predictor.predict(["CCC", "C", "CCO", "CCN"])  # MISS × 4
    print("Cache Stats:", cache.stats())

    # Prediction set 2 — same model, partial hit/miss
    predictor.model_name = "example_model"
    predictor.predict(["CCC", "CO", "CCO", "CN"])  # HIT: CCC, CCO — MISS: CO, CN
    print("Cache Stats:", cache.stats())

    # Prediction set 3 — new model, same SMILES — should all be misses (per-model caching)
    predictor.model_name = "example_model_2"
    predictor.predict(["CCC", "C", "CO", "CN"])  # MISS × 4 (new model)
    print("Cache Stats:", cache.stats())

    # Prediction set 4 — another model
    predictor.model_name = "example_model_3"
    predictor.predict(["CCCC", "CCCl", "CCBr", "C(C)C"])  # MISS × 4
    print("Cache Stats:", cache.stats())

    from pprint import pprint

    pprint(cache)
