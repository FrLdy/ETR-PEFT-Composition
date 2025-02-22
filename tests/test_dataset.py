import unittest
from pathlib import Path

from expes.dataset import (
    DST_KEY,
    SRC_KEY,
    build_mtl_dataset,
    load_etr_fr,
    load_orangesum,
    load_wikilarge_fr,
)

from .utils import lorem_ipsum_dataset


def get_datasets():
    dataset_names = ["ds1", "ds2", "ds3"]
    dataset_sizes = [10, 34, 50]
    datasets = {
        name: lorem_ipsum_dataset(size)
        for (name, size) in zip(dataset_names, dataset_sizes)
    }
    return datasets, dataset_names


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
    def test_build_mtl_dataset(self):
        datasets, task_ids_map = get_datasets()
        mtl_dataset = build_mtl_dataset(
            datasets, task_ids_map, stopping_strategy=None
        )
        assert len(mtl_dataset) == sum(len(ds) for ds in datasets.values())

    def test_build_mtl_dataset_interleave_f_exhausted(self):
        datasets, task_ids_map = get_datasets()

        mtl_dataset = build_mtl_dataset(
            datasets, task_ids_map, stopping_strategy="first_exhausted"
        )
        assert len(mtl_dataset) == min(
            len(ds) for ds in datasets.values()
        ) * len(datasets)

    def test_build_mtl_dataset_interleave_all_exhausted(self):
        datasets, task_ids_map = get_datasets()
        mtl_dataset = build_mtl_dataset(
            datasets, task_ids_map, stopping_strategy="all_exhausted"
        )
        assert len(mtl_dataset) == max(
            len(ds) for ds in datasets.values()
        ) * len(datasets)
