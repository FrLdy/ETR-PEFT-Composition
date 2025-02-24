import unittest
from pathlib import Path

from datasets import DatasetDict

from expes.dataset import (
    DST_KEY,
    SRC_KEY,
    build_mtl_dataset,
    load_etr_fr,
    load_orangesum,
    load_wikilarge_fr,
)

from .utils import get_mtl_dataset, lorem_ipsum_dataset


class TestLoadDatasets(unittest.TestCase):
    data_dir = Path(__file__).parents[1] / "data"

    columns = [SRC_KEY, DST_KEY]
    splits = ["train", "test", "validation"]

    def run_test(self, dataset):
        assert all(key in dataset for key in self.splits)
        assert all(len(dataset[key]) > 0 for key in self.splits)
        assert all(
            all(col in dataset[key].column_names for col in self.columns)
            for key in self.splits
        )

    def test_load_orangesum(self):
        dataset = load_orangesum()
        self.run_test(dataset)

    def test_load_wikilarge_fr(self):
        location = self.data_dir / "wikilarge-fr"
        assert location.exists()
        dataset = load_wikilarge_fr(location, use_cache=False)
        self.run_test(dataset)
        dataset = load_wikilarge_fr(location, use_cache=True)
        self.run_test(dataset)

    def test_load_etr_fr(self):
        location = self.data_dir / "etr-fr"
        assert location.exists()
        dataset = load_etr_fr(location, use_cache=False)
        self.run_test(dataset)
        dataset = load_etr_fr(location, use_cache=True)
        self.run_test(dataset)


class TestDataset(unittest.TestCase):
    splits = ["train", "validation", "test"]
    tasks = ["ds1", "ds2", "ds3"]
    dataset_sizes = [(10, 11, 12), (13, 14, 15), (16, 17, 18)]

    def get_datasets(self):
        return get_mtl_dataset(self.tasks, self.dataset_sizes)

    def run_base_tests(self, mtl_dataset, splits=None):
        self.assertEqual(len(mtl_dataset), 3)
        self.assertEqual(set(mtl_dataset.keys()), set(self.splits))
        splits = splits or self.splits
        for split in splits:
            for task in self.tasks:
                self.assertIn(task, mtl_dataset[split])

    def test_build_mtl_dataset(self):
        datasets = self.get_datasets()
        mtl_dataset = build_mtl_dataset(datasets, stopping_strategy=None)
        self.run_base_tests(mtl_dataset)

    def test_build_mtl_dataset_interleave_f_exhausted(self):
        datasets = self.get_datasets()

        mtl_dataset = build_mtl_dataset(
            datasets, stopping_strategy="first_exhausted"
        )
        self.run_base_tests(mtl_dataset, splits=["validation", "test"])

    def test_build_mtl_dataset_interleave_all_exhausted(self):
        datasets = self.get_datasets()
        mtl_dataset = build_mtl_dataset(
            datasets, stopping_strategy="all_exhausted"
        )
        self.run_base_tests(mtl_dataset, splits=["validation", "test"])
