import random

import lorem
import torch
from datasets import Dataset, DatasetDict
from transformers.testing_utils import torch_device

global_rng = random.Random()


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return (
        torch.tensor(data=values, dtype=torch.long, device=torch_device)
        .view(shape)
        .contiguous()
    )


def get_trainable_param_names(model):
    trainable_parameters = (
        n for n, p in model.named_parameters() if p.requires_grad
    )

    return trainable_parameters


def lorem_ipsum_dataset(n):
    def gen():
        for _ in range(n):
            yield {"src": lorem.sentence(), "dst": lorem.sentence()}

    return Dataset.from_generator(gen)


def dummy_dataset(n_sample, input_len=100, label_len=100):
    def gen():
        for _ in range(n_sample):
            yield {
                "input_ids": ids_tensor(input_len, 1000).to(torch_device),
                "labels": ids_tensor(label_len, 1000).to(torch_device),
            }

    return Dataset.from_generator(gen())


def dummy_compute_metrics(**kwargs):
    return {f"metric{i}": random.uniform(0, 1) for i in range(5)}


def get_dataset(n_tasks=None, task_id=None):
    dataset = lorem_ipsum_dataset(25)
    dataset_size = len(dataset)
    if n_tasks:
        dataset = dataset.add_column(
            "task_ids",
            torch.randint(0, n_tasks, (dataset_size,)).tolist(),
        )
    if task_id is not None:
        dataset = dataset.add_column(
            "task_ids", torch.tensor([task_id] * dataset_size).tolist()
        )
    return dataset


def get_mtl_dataset(tasks, dataset_sizes):
    datasets = {}
    for task, (train, dev, test) in zip(tasks, dataset_sizes):
        datasets[task] = DatasetDict(
            {
                "train": lorem_ipsum_dataset(train),
                "validation": lorem_ipsum_dataset(dev),
                "test": lorem_ipsum_dataset(test),
            }
        )
    return DatasetDict(datasets)
