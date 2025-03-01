from dataclasses import dataclass, field
from functools import partial
from typing import Callable, List, Optional

from etr_fr_expes.dataset import (
    AVAILABLE_DATASETS,
    DS_KEY_ETR_FR,
    DS_KEY_ORANGESUM,
    DS_KEY_WIKILARGE_FR,
)
from etr_fr_expes.metric import METRIC_KEY_SRB, ETRMetrics
from expes.config import DataConfig, TrainingConfig, TunerConfig
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
