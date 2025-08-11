# Note: The top-level package __init__.py runs only once,
# even if multiple subpackages are imported later.

from ._custom_cache import PerSmilesPerModelLRUCache

modelwise_smiles_lru_cache = PerSmilesPerModelLRUCache(max_size=100)
