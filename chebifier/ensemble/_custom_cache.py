import os
import pickle
import threading
from collections import OrderedDict
from typing import Any


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

    def _save_cache(self) -> None:
        """Serialize the cache to disk."""
        if not self._persist_path:
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
    # Example usage
    cache = PerSmilesPerModelLRUCache(max_size=100, persist_path="cache.pkl")
