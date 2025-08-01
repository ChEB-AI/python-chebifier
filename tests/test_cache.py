import os
import tempfile
import unittest

from chebifier import PerSmilesPerModelLRUCache

g_cache = PerSmilesPerModelLRUCache(max_size=3)


class DummyPredictor:
    def __init__(self, model_name):
        self.model_name = model_name

    @g_cache.batch_decorator
    def predict(self, smiles_list: tuple[str]):
        # Simple predictable dummy function for tests
        return [f"{self.model_name}{i}" for i in range(len(smiles_list))]


class TestPerSmilesPerModelLRUCache(unittest.TestCase):
    def setUp(self):
        # Create temp file for persistence tests
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
        self.cache = PerSmilesPerModelLRUCache(
            max_size=3, persist_path=self.temp_file.name
        )

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_cache_miss_and_set_get(self):
        # Initially empty
        self.assertEqual(len(self.cache), 0)
        self.assertIsNone(self.cache.get("CCC", "model1"))

        # Set and get
        self.cache.set("CCC", "model1", "result1")
        self.assertEqual(self.cache.get("CCC", "model1"), "result1")
        self.assertEqual(self.cache.hits, 1)
        self.assertEqual(self.cache.misses, 1)  # One miss from first get

    def test_cache_eviction(self):
        self.cache.set("a", "m", "v1")
        self.cache.set("b", "m", "v2")
        self.cache.set("c", "m", "v3")
        self.assertEqual(len(self.cache), 3)
        # Adding one more triggers eviction of oldest
        self.cache.set("d", "m", "v4")
        self.assertEqual(len(self.cache), 3)
        self.assertIsNone(self.cache.get("a", "m"))  # 'a' evicted
        self.assertIsNotNone(self.cache.get("d", "m"))  # 'd' present

    def test_batch_decorator_hits_and_misses(self):
        predictor = DummyPredictor("modelA")
        predictor2 = DummyPredictor("modelB")

        # Clear cache before starting the test
        g_cache.clear()

        smiles = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        # First call all misses
        results1 = predictor.predict(smiles)
        results1_model2 = predictor2.predict(smiles)

        # all prediction as retrived from actual prediction function and not from cache
        self.assertListEqual(
            results1, ["modelA_P0", "modelA_P1", "modelA_P2", "modelA_P3", "modelA_P4"]
        )
        self.assertListEqual(
            results1_model2,
            ["modelB_P0", "modelB_P1", "modelB_P2", "modelB_P3", "modelB_P4"],
        )
        stats_after_first = g_cache.stats()
        self.assertEqual(stats_after_first["misses"], 3)

        # cache = {("AAA", "modelA"): "modelA_P0", ("BBB", "modelA"): "modelA_P1", ("CCC", "modelA"): "modelA_P2"}
        # Second call with some hits and some misses
        results2 = predictor.predict(["FFF", "DDD"])
        # AAA from cache
        # FFF is not in cache, so it predicted, hence it has P0 as its the only one passed to prediction function
        # and dummy predictor returns iterates over the smiles list and return P{idx} corresponding to the index
        self.assertListEqual(results2, ["P3", "P0"])
        stats_after_second = g_cache.stats()
        self.assertEqual(stats_after_second["hits"], 1)
        self.assertEqual(stats_after_second["misses"], 4)

        # cache = {("AAA", "modelA"): "P0", ("BBB", "modelA"): "P1", ("CCC", "modelA"): "P2",
        # ("DDD", "modelA"): "P3", ("EEE", "modelA"): "P4", ("FFF", "modelA"): "P0"}

        # Third call with some hits and some misses
        results3 = predictor.predict(["EEE", "GGG", "DDD", "HHH", "BBB", "ZZZ"])
        # Here, predictions for [EEE, DDD, BBB] are retrived from cache,
        # while [GGG, HHH, ZZZ] are not in cache and hence passe to the prediction function
        self.assertListEqual(results3, ["P4", "P0", "P3", "P0", "P1", "P0"])
        stats_after_third = g_cache.stats()
        self.assertEqual(stats_after_third["hits"], 1)
        self.assertEqual(stats_after_third["misses"], 4)

    def test_persistence_save_and_load(self):
        # Set some values
        self.cache.set("sm1", "modelX", "val1")
        self.cache.set("sm2", "modelX", "val2")

        # Save cache to file
        self.cache.save()

        # Create new cache instance loading from file
        new_cache = PerSmilesPerModelLRUCache(
            max_size=3, persist_path=self.temp_file.name
        )
        new_cache.load()

        self.assertEqual(new_cache.get("sm1", "modelX"), "val1")
        self.assertEqual(new_cache.get("sm2", "modelX"), "val2")

    def test_clear_cache(self):
        self.cache.set("x", "m", "v")
        self.cache.save()
        self.assertTrue(os.path.exists(self.temp_file.name))
        self.cache.clear()
        self.assertEqual(len(self.cache), 0)
        self.assertFalse(os.path.exists(self.temp_file.name))


if __name__ == "__main__":
    unittest.main()
