from dataclasses import dataclass, field
from functools import partial
from os import wait
from typing import Callable, List, Optional

from ray import train

from etr_fr_expes.dataset import (
    AVAILABLE_DATASETS,
    DS_KEY_ETR_FR,
    DS_KEY_ETR_FR_POLITIC,
    DS_KEY_ORANGESUM,
    DS_KEY_WIKILARGE_FR,
)
from etr_fr_expes.metric import METRIC_KEY_SRB, ETRMetrics
from expes.config import DataConfig, InferenceConfig, TrainingConfig, TunerConfig
from expes.dataset import get_dataset_factory_fn


@dataclass
class ETRTunerConfig(TunerConfig):
    metric: str = f"eval_{DS_KEY_ETR_FR}_{METRIC_KEY_SRB}"
    mode: str = "max"
    num_samples: int = 1
    robustness_num_samples: int = 5
    num_to_keep: Optional[int] = 1


@dataclass
class ETRTrainingConfig(TrainingConfig):
    train_tasks: List[str] = field(
        default_factory=lambda: [
            DS_KEY_ETR_FR,
            DS_KEY_WIKILARGE_FR,
            DS_KEY_ORANGESUM,
        ]
    )
    get_metrics_fn: Callable = partial(ETRMetrics, lang="fr")


@dataclass
class ETRDataConfig(DataConfig):
    get_datasets: Callable = get_dataset_factory_fn(
        AVAILABLE_DATASETS, singleton=True
    )


def get_inference_config(train_tasks):
    inference_config = InferenceConfig(
        validation_tasks=train_tasks,
        test_tasks=train_tasks,
        task_to_task_ids={
            i:task for task,i in enumerate(train_tasks)
        }
    )
    
    if DS_KEY_ETR_FR in train_tasks:
        inference_config.test_tasks.append(DS_KEY_ETR_FR_POLITIC)
        inference_config.task_to_task_ids.update({DS_KEY_ETR_FR_POLITIC: train_tasks.index(DS_KEY_ETR_FR)})

    return inference_config

