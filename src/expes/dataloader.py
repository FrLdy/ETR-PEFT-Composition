import os
import re
from pathlib import Path
from typing import Optional, Union

from datasets import DatasetDict, load_dataset, load_from_disk

SRC_KEY = "src"
DST_KEY = "dst"


def load_orangesum():
    dataset = load_dataset("EdinburghNLP/orange_sum", "abstract")
    return dataset


def load_wikilarge_fr(
    location: Optional[Union[str, Path]] = None,
    use_cache=True,
    save_to_disk=False,
):

    location = location or os.environ.get("WIKILARGE_FR_DIR")
    assert location is not None
    location = Path(location).resolve()
    cache_location = location / "hf_dataset"

    if use_cache and cache_location.exists():
        return load_from_disk(cache_location)

    def clean(text):
        pattern = r"-?.?LRB.?-?|-?.?RRB.?-?"
        result = re.sub(pattern, "", text)
        return result

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

    if save_to_disk:
        dataset_dict.save_to_disk(cache_location)

    return dataset_dict


def load_etr_fr(
    location: Optional[Union[str, Path]] = None,
    use_cache=True,
    save_to_disk=True,
):
    location = location or os.environ.get("ETR_FR_DIR")

    assert location is not None
    location = Path(location).resolve()
    cache_location = location / "hf_dataset"

    if use_cache and cache_location.exists():
        return load_from_disk(cache_location)

    source_location = location / "sources"
    file_paths = {
        "train": str(source_location / "train.json"),
        "validation": str(source_location / "val.json"),
        "test": str(source_location / "test.json"),
    }

    dataset_dict = load_dataset("json", data_files=file_paths).rename_columns(
        {"original": SRC_KEY, "falc": DST_KEY}
    )

    if save_to_disk:
        dataset_dict.save_to_disk(cache_location)

    return dataset_dict
