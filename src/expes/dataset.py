from functools import partial
from typing import Dict, Literal, Optional

from datasets import Dataset, concatenate_datasets, interleave_datasets


def build_mtl_dataset(
    datasets: Dict[str, Dataset],
    task_ids_map: Dict[str, int],
    stopping_strategy: Optional[Literal["first_exhausted", "all_exhausted"]] = None,
):

    datasets = {
        ds_name: ds.map(lambda _: {"task_ids": task_ids_map[ds_name]})
        for ds_name, ds in datasets.items()
    }

    concat_fn = {
        None: concatenate_datasets,
        "first_exhausted": partial(
            interleave_datasets, stopping_strategy="first_exhausted"
        ),
        "all_exhausted": partial(
            interleave_datasets, stopping_strategy="all_exhausted"
        ),
    }

    mtl_dataset = concat_fn[stopping_strategy](list(datasets.values()))
    return mtl_dataset
