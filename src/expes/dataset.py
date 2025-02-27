from functools import partial
from typing import Dict, Literal, Optional

from datasets import DatasetDict, concatenate_datasets, interleave_datasets

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
