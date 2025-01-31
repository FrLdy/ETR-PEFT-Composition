import unittest

from expes.dataset import build_mtl_dataset
from expes.tests.utils import lorem_ipsum_dataset


def build_datasets():
    dataset_names = ["ds1", "ds2", "ds3"]
    dataset_sizes = [10, 34, 50]
    task_ids_map = {name: idx for idx, name in enumerate(dataset_names)}
    datasets = {
        name: lorem_ipsum_dataset(size)
        for (name, size) in zip(dataset_names, dataset_sizes)
    }
    return datasets, task_ids_map


class TestDataset(unittest.TestCase):
    def test_build_mtl_dataset(self):
        datasets, task_ids_map = build_datasets()
        mtl_dataset = build_mtl_dataset(
            datasets, task_ids_map, stopping_strategy=None
        )
        assert len(mtl_dataset) == sum(len(ds) for ds in datasets.values())

    def test_build_mtl_dataset_interleave_f_exhausted(self):
        datasets, task_ids_map = build_datasets()

        mtl_dataset = build_mtl_dataset(
            datasets, task_ids_map, stopping_strategy="first_exhausted"
        )
        assert len(mtl_dataset) == min(
            len(ds) for ds in datasets.values()
        ) * len(datasets)

    def test_build_mtl_dataset_interleave_all_exhausted(self):
        datasets, task_ids_map = build_datasets()
        mtl_dataset = build_mtl_dataset(
            datasets, task_ids_map, stopping_strategy="all_exhausted"
        )
        assert len(mtl_dataset) == max(
            len(ds) for ds in datasets.values()
        ) * len(datasets)
