import unittest
from pathlib import Path

from expes.dataloader import load_etr_fr, load_orangesum, load_wikilarge_fr


class TestLoadDatasets(unittest.TestCase):
    def run_test(self, dataset, splits, columns):
        assert all(key in dataset for key in splits)
        assert all(
            all(col in dataset[key].column_names for col in columns)
            for key in splits
        )

    def test_load_orangesum(self):
        splits = ["train", "test", "validation"]
        columns = ["text", "summary"]
        dataset = load_orangesum()
        self.run_test(dataset, splits, columns)

    def test_load_wikilarge_fr(self):
        splits = ["train", "test", "validation"]
        columns = ["src", "dst"]
        location = Path(__file__).parents[3] / "data" / "wikilarge-fr"
        assert location.exists()
        dataset = load_wikilarge_fr(
            location, use_cache=False, save_to_disk=False
        )
        self.run_test(dataset, splits, columns)

    def test_load_etr_fr(self):
        splits = ["train", "test", "validation"]
        columns = ["src", "dst"]
        location = Path(__file__).parents[3] / "data" / "etr-fr"
        assert location.exists()
        dataset = load_etr_fr(location, use_cache=False, save_to_disk=False)
        self.run_test(dataset, splits, columns)
