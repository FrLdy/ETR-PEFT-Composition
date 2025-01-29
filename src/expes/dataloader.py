import os
import re
from pathlib import Path
from typing import Optional, Union

from datasets import DatasetDict, load_dataset, load_from_disk


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
    file_paths = {
        "train": str(source_location / "wikilarge-fr.train.csv"),
        "validation": str(source_location / "wikilarge-fr.val.csv"),
        "test": str(source_location / "wikilarge-fr.test.csv"),
    }

    dataset_dict = DatasetDict()

    for split, file_path in file_paths.items():
        dataset = load_dataset("csv", data_files=file_path)["train"]
        if "simple1" in dataset.column_names:
            dataset = dataset.remove_columns("simple1")
        dataset = dataset.rename_columns({"original": "src", "simple": "dst"})
        dataset = dataset.map(
            lambda example: {
                "src": clean(example["src"]),
                "dst": clean(example["dst"]),
            },
            batched=False,
        )
        dataset_dict[split] = dataset

    if save_to_disk:
        dataset_dict.save_to_disk(cache_location)

    return dataset_dict
