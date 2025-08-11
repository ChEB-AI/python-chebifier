import os
import pickle
import threading
from collections import OrderedDict
from collections.abc import Iterable
from functools import wraps
from typing import Any, Callable


class PerSmilesPerModelLRUCache:
    """
    A thread-safe, optionally persistent LRU cache for storing
    (SMILES, model_name) → result mappings.
    """

    def __init__(self, max_size: int = 100, persist_path: str | None = None):
        """
        Initialize the cache.

        Args:
            max_size (int): Maximum number of items to keep in the cache.
            persist_path (str | None): Optional path to persist cache using pickle.
        """
        self._cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._persist_path = persist_path

        self.hits = 0
        self.misses = 0

        if self._persist_path:
            self._load_cache()

    def get(self, smiles: str, model_name: str) -> Any | None:
        """
        Retrieve value from cache if present, otherwise return None.

        Args:
            smiles (str): SMILES string key.
            model_name (str): Model identifier.

        Returns:
            Any | None: Cached value or None.
        """
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
        """
        Store value in cache under (smiles, model_name) key.

        Args:
            smiles (str): SMILES string key.
            model_name (str): Model identifier.
            value (Any): Value to cache.
        """
        assert value is not None, "Value must not be None"
        key = (smiles, model_name)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """
        Clear the cache and remove the persistence file if present.
        """
        self._save_cache()
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            if self._persist_path and os.path.exists(self._persist_path):
                os.remove(self._persist_path)

    def stats(self) -> dict[str, int]:
        """
        Return cache hit/miss statistics.

        Returns:
            dict[str, int]: Dictionary with 'hits' and 'misses' keys.
        """
        return {"hits": self.hits, "misses": self.misses}

    def batch_decorator(self, func: Callable) -> Callable:
        """
        Decorator for class methods that accept a batch of SMILES as a list,
        and cache predictions per (smiles, model_name) key.

        The instance is expected to have a `model_name` attribute.

        Args:
            func (Callable): The method to decorate.

        Returns:
            Callable: The wrapped method.
        """

        @wraps(func)
        def wrapper(instance, smiles_list: list[str]) -> list[Any]:
            assert isinstance(smiles_list, list), "smiles_list must be a list."
            model_name = getattr(instance, "model_name", None)
            assert model_name is not None, "Instance must have a model_name attribute."

            missing_smiles: list[str] = []
            missing_indices: list[int] = []
            ordered_results: list[Any] = [None] * len(smiles_list)

            # First: try to fetch all from cache
            for idx, smiles in enumerate(smiles_list):
                prediction = self.get(smiles=smiles, model_name=model_name)
                if prediction is not None:
                    # For debugging purposes, you can uncomment the print statement below
                    # print(
                    #     f"[Cache Hit] Prediction for smiles: {smiles} and model: {model_name} are retrieved from cache."
                    # )
                    ordered_results[idx] = prediction
                else:
                    missing_smiles.append(smiles)
                    missing_indices.append(idx)

            # If some are missing, call original function
            if missing_smiles:
                new_results = func(instance, tuple(missing_smiles))
                assert isinstance(
                    new_results, Iterable
                ), "Function must return an Iterable."

                # Save to cache and append
                for smiles, prediction, missing_idx in zip(
                    missing_smiles, new_results, missing_indices
                ):
                    if prediction is not None:
                        self.set(smiles, model_name, prediction)
                    ordered_results[missing_idx] = prediction

            return ordered_results

        return wrapper

    def __len__(self) -> int:
        """
        Return number of items in the cache.

        Returns:
            int: Number of entries in the cache.
        """
        with self._lock:
            return len(self._cache)

    def __repr__(self) -> str:
        """
        String representation of the underlying cache.

        Returns:
            str: String version of the OrderedDict.
        """
        return self._cache.__repr__()

    def save(self) -> None:
        """
        Save the cache to disk, if persistence is enabled.
        """
        self._save_cache()

    def load(self) -> None:
        """
        Load the cache from disk, if persistence is enabled.
        """
        self._load_cache()

    def _save_cache(self) -> None:
        """
        Serialize the cache to disk using pickle.
        """
        if self._persist_path:
            try:
                with open(self._persist_path, "wb") as f:
                    pickle.dump(self._cache, f)
            except Exception as e:
                print(f"[Cache Save Error] {e}")

    def _load_cache(self) -> None:
        """
        Load the cache from disk, if the file exists and is non-empty.
        """
        if (
            self._persist_path
            and os.path.exists(self._persist_path)
            and os.path.getsize(self._persist_path) > 0
        ):
            try:
                with open(self._persist_path, "rb") as f:
                    loaded = pickle.load(f)
                    if isinstance(loaded, OrderedDict):
                        self._cache = loaded
            except Exception as e:
                print(f"[Cache Load Error] {e}")
