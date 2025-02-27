import os
import re
from pathlib import Path
from typing import Optional, Union

from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.config import HF_DATASETS_CACHE

from expes.config import TrainingConfig
from expes.dataset import build_mtl_dataset

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


DS_KEY_ETR_FR = "etr_fr"
DS_KEY_WIKILARGE_FR = "wikilarge_fr"
DS_KEY_ORANGESUM = "orangesum"

DATASETS = {
    DS_KEY_ETR_FR: load_etr_fr,
    DS_KEY_WIKILARGE_FR: load_wikilarge_fr,
    DS_KEY_ORANGESUM: load_orangesum,
}


def get_tasks_datasets(
    tasks=None,
):
    tasks = tasks or list(DATASETS.keys())
    datasets = DatasetDict({task: DATASETS[task]() for task in tasks})
    return datasets


def prepare_datasets(config: TrainingConfig, tokenizer):
    datasets = get_tasks_datasets(tasks=config.tasks)
    data_config = config.data_config
    if data_config.n_samples is not None:
        for split in datasets:
            for task in datasets[split]:
                datasets[split][task] = datasets[split][task].select(
                    range(data_config.n_samples)
                )

    if data_config.tokenize_dataset:

        def tokenize_function(batch):
            inputs = tokenizer(
                batch[SRC_KEY],
                truncation=True,
                max_length=128,
            )
            targets = tokenizer(
                text_target=batch[DST_KEY],
                truncation=True,
                max_length=128,
            )
            inputs["labels"] = targets["input_ids"]

            return inputs

        for task in datasets:
            datasets[task] = datasets[task].map(
                tokenize_function,
                batched=True,
                remove_columns=[SRC_KEY, DST_KEY],
            )

    return build_mtl_dataset(
        datasets, stopping_strategy=data_config.stopping_strategy
    )
