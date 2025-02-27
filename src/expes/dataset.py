from functools import partial
from typing import Dict, List, Literal, Optional

from datasets import DatasetDict, concatenate_datasets, interleave_datasets

from expes.config import TrainingConfig

SRC_KEY = "src"
DST_KEY = "dst"

StoppingStrategy = Literal["concatenate", "first_exhausted", "all_exhausted"]


def build_mtl_dataset(
    datasets: Dict[str, DatasetDict],
    stopping_strategy: Optional[StoppingStrategy] = None,
):
    datasets = {
        ds_name: datasets[ds_name].map(lambda _: {"task_ids": i})
        for i, ds_name in enumerate(datasets.keys())
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


class MTLDatasetFactory:
    def __init__(self, available_dataset_loaders, singleton=True) -> None:
        self.available_dataset_loaders = available_dataset_loaders
        self.singleton = singleton
        self._prepared_datasets = {}

    def __call__(self, tasks: List[str], stopping_strategy: StoppingStrategy):
        hash = "{tasks}/{stopping_strategy}".format(
            tasks=";".join(tasks), stopping_strategy=stopping_strategy
        )
        if self.singleton and hash in self._prepared_datasets:
            prepared_dataset = self._prepared_datasets[hash]
        else:
            datasets = self.get_tasks_datasets(tasks)
            prepared_dataset = build_mtl_dataset(datasets, stopping_strategy)

        if self.singleton and hash not in self._prepared_datasets:
            self._prepared_datasets[hash] = prepared_dataset

        return prepared_dataset

    def get_tasks_datasets(
        self,
        tasks,
    ):
        datasets = DatasetDict(
            {task: self.available_dataset_loaders[task]() for task in tasks}
        )
        return datasets


def tokenize_dataset(dataset, tokenizer, **kwargs):
    def tokenize_batch(batch):
        inputs = tokenizer(
            batch[SRC_KEY],
            truncation=True,
            max_length=kwargs.get("input_max_length", 128),
        )
        targets = tokenizer(
            text_target=batch[DST_KEY],
            truncation=True,
            max_length=kwargs.get("label_max_length", 128),
        )
        inputs["labels"] = targets["input_ids"]

        return inputs

    for task in dataset:
        dataset[task] = dataset[task].map(
            tokenize_batch,
            batched=True,
            remove_columns=[SRC_KEY, DST_KEY],
        )

    return dataset


def get_dataset_factory_fn(available_dataset_loaders, singleton):
    dataset_factory = MTLDatasetFactory(
        available_dataset_loaders=available_dataset_loaders, singleton=singleton
    )

    def factory_fn(config: TrainingConfig, tokenizer):
        data_config = config.data_config
        datasets = dataset_factory(
            tasks=config.tasks, stopping_strategy=data_config.stopping_strategy
        )

        if data_config.tokenize_dataset:
            datasets = tokenize_dataset(
                datasets,
                tokenizer,
                input_max_length=data_config.input_max_length,
                output_max_length=data_config.output_max_length,
            )

    return factory_fn
