from dataclasses import dataclass, field
from typing import Callable, List, Optional

from etr_fr_expes.dataset import (
    AVAILABLE_DATASETS,
    DS_KEY_ETR_FR,
    DS_KEY_ORANGESUM,
    DS_KEY_WIKILARGE_FR,
)
from etr_fr_expes.metric import METRIC_KEY_SRB, etr_compute_metrics
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
class ETRDataConfig(DataConfig):
    get_dataset: Callable = get_dataset_factory_fn(
        AVAILABLE_DATASETS, singleton=True
    )


@dataclass
class ETRTrainingConfig(TrainingConfig):
    compute_metric: Callable = etr_compute_metrics
    tasks: List[str] = field(
        default_factory=lambda: [
            DS_KEY_ETR_FR,
            DS_KEY_WIKILARGE_FR,
            DS_KEY_ORANGESUM,
        ]
    )
