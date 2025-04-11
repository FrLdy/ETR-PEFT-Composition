from functools import partial
from optparse import Option
from typing import Dict, List, Optional

from datasets import DatasetDict, concatenate_datasets, interleave_datasets

from expes.config import TrainingConfig
from expes.types import SamplingStrategy

SRC_KEY = "src"
DST_KEY = "dst"


def balanced(datasets):
    max_n_samples = min(len(d) for d in datasets)
    return concatenate_datasets(
        [d.select(range(max_n_samples)) for d in datasets]
    )


def build_mtl_dataset(
    datasets: Dict[str, DatasetDict],
    train_tasks: List[str],
    validation_tasks: List[str],
    test_tasks: List[str],
    task_to_task_ids: Optional[Dict] = None,
    sampling_strategy: Optional[SamplingStrategy] = None,
):
    datasets = {
        ds_name: datasets[ds_name].map(
            lambda _: {
                "task_ids": (
                    i if task_to_task_ids is None else task_to_task_ids[ds_name] if ds_name in task_to_task_ids else None
                )
            }
        )
        for i, ds_name in enumerate(datasets.keys())
    }

    concat_fn = {
        "balanced": balanced,  # TODO: add strategy using n_samples
        "concatenate": concatenate_datasets,
        "first_exhausted": partial(
            interleave_datasets, stopping_strategy="first_exhausted"
        ),
        "all_exhausted": partial(
            interleave_datasets, stopping_strategy="all_exhausted"
        ),
    }

    def group_by_split(split, tasks, interleave=None):
        ds_split = DatasetDict(
            {
                name: dset[split]
                for name, dset in datasets.items()
                if name in tasks and split in dset
            }
        )
        if interleave is not None:
            ds_split = concat_fn[interleave](list(ds_split.values()))
        return ds_split

    return DatasetDict(
        {
            "train": group_by_split("train", train_tasks, sampling_strategy),
            "validation": group_by_split("validation", validation_tasks),
            "test": group_by_split("test", test_tasks),
        }
    )


class MTLDatasetFactory:
    def __init__(self, available_dataset_loaders, singleton=True) -> None:
        self.available_dataset_loaders = available_dataset_loaders
        self.singleton = singleton
        self._prepared_datasets = {}

    def __call__(self, config: TrainingConfig):
        train_tasks = config.train_tasks
        eval_tasks = config.validation_tasks
        test_tasks = config.test_tasks
        sampling_strategy = config.data_config.sampling_strategy
        hash = "{train_tasks}/{eval_tasks}/{test_tasks}/{stopping_strategy}".format(
            train_tasks=";".join(train_tasks),
            eval_tasks=";".join(eval_tasks),
            test_tasks=";".join(test_tasks),
            stopping_strategy=sampling_strategy,
        )
        if self.singleton and hash in self._prepared_datasets:
            prepared_dataset = self._prepared_datasets[hash]
        else:
            datasets = self.get_tasks_datasets(
                set([*train_tasks, *eval_tasks, *test_tasks])
            )
            prepared_dataset = build_mtl_dataset(
                datasets,
                train_tasks,
                eval_tasks,
                test_tasks,
                task_to_task_ids=config.task_to_task_ids,
                sampling_strategy=sampling_strategy,
            )

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
        datasets = dataset_factory(config)

        if data_config.tokenize_dataset:
            datasets = tokenize_dataset(
                datasets,
                tokenizer,
                input_max_length=data_config.input_max_length,
                output_max_length=data_config.output_max_length,
            )
        return datasets

    return factory_fn
