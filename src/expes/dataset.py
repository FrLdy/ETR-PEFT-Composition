import os
import re
from functools import partial
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
    load_from_disk,
)
from datasets.config import HF_DATASETS_CACHE
from datasets.packaged_modules.parquet.parquet import ds

SRC_KEY = "src"
DST_KEY = "dst"


def load_orangesum():
    dataset = load_dataset(
        "EdinburghNLP/orange_sum", "abstract", trust_remote_code=True
    )
    dataset = dataset.rename_columns({"text": SRC_KEY, "summary": DST_KEY})
    return dataset


def load_wikilarge_fr(
    location: Optional[Union[str, Path]] = None,
    use_cache=True,
):

    cache_location = HF_DATASETS_CACHE / "wikilarge_fr"

    if use_cache and cache_location.exists():
        return load_from_disk(cache_location)

    def clean(text):
        pattern = r"-?.?LRB.?-?|-?.?RRB.?-?"
        result = re.sub(pattern, "", text)
        return result

    location = location or os.environ.get("WIKILARGE_FR_DIR")
    location = Path(location).resolve()
    assert location is not None
    source_location = location / "sources"
    file_format = "wikilarge-fr.{}.csv"
    file_paths = {
        "train": str(source_location / file_format.format("train")),
        "validation": str(source_location / file_format.format("val")),
        "test": str(source_location / file_format.format("test")),
    }

    dataset_dict = DatasetDict()

    for split, file_path in file_paths.items():
        dataset = load_dataset("csv", data_files=file_path)["train"]
        if "simple1" in dataset.column_names:
            dataset = dataset.remove_columns("simple1")
        dataset = dataset.rename_columns(
            {"original": SRC_KEY, "simple": DST_KEY}
        )
        dataset = dataset.map(
            lambda example: {
                SRC_KEY: clean(example[SRC_KEY]),
                DST_KEY: clean(example[DST_KEY]),
            },
            batched=False,
        )
        dataset_dict[split] = dataset

    if use_cache:
        dataset_dict.save_to_disk(cache_location)

    return dataset_dict


def load_etr_fr(
    location: Optional[Union[str, Path]] = None,
    use_cache=True,
):

    cache_location = HF_DATASETS_CACHE / "etr_fr"

    if use_cache and cache_location.exists():
        return load_from_disk(cache_location)

    location = location or os.environ.get("ETR_FR_DIR")
    assert location is not None
    location = Path(location).resolve()
    source_location = location / "sources"
    file_paths = {
        "train": str(source_location / "train.json"),
        "validation": str(source_location / "val.json"),
        "test": str(source_location / "test.json"),
    }

    dataset_dict = load_dataset("json", data_files=file_paths).rename_columns(
        {"original": SRC_KEY, "falc": DST_KEY}
    )

    if use_cache:
        dataset_dict.save_to_disk(cache_location)

    return dataset_dict


def build_mtl_dataset(
    datasets: Dict[str, DatasetDict],
    tasks: List[str] = None,
    stopping_strategy: Optional[
        Literal["concatenate", "first_exhausted", "all_exhausted"]
    ] = None,
):
    tasks = tasks or list(datasets.keys())
    datasets = {
        ds_name: datasets[ds_name].map(lambda _: {"task_ids": i})
        for i, ds_name in enumerate(tasks)
    }

    concat_fn = {
        "concatenate": concatenate_datasets,
        "first_exhausted": partial(
            interleave_datasets, stopping_strategy="first_exhausted"
        ),
        "all_exhausted": partial(
            interleave_datasets, stopping_strategy="all_exhausted"
        ),
    }

    def group_by_split(split, interleave=None):
        ds_split = DatasetDict(
            {name: dset[split] for name, dset in datasets.items()}
        )
        if interleave is not None:
            ds_split = concat_fn[interleave](list(ds_split.values()))
        return ds_split

    return DatasetDict(
        {
            "train": group_by_split("train", stopping_strategy),
            "test": group_by_split("test"),
            "validation": group_by_split("validation"),
        }
    )


DS_KEY_ETR_FR = "etr_fr"
DS_KEY_WIKILARGE_FR = "wikilarge_fr"
DS_KEY_ORANGESUM = "orangesum"

DATASETS = {
    DS_KEY_ETR_FR: load_etr_fr,
    DS_KEY_WIKILARGE_FR: load_wikilarge_fr,
    DS_KEY_ORANGESUM: load_orangesum,
}


def base_mtl_dataset(
    tasks,
    interleave: Optional[
        Literal["concatenate", "first_exhausted", "all_exhausted"]
    ] = None,
):
    datasets = {task: DATASETS[task]() for task in tasks}
    return build_mtl_dataset(datasets, tasks, interleave)
