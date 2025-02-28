from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Type, Union

import torch
from adapters.composition import AdapterCompositionBlock
from adapters.trainer import Seq2SeqAdapterTrainer
from ray.air.config import SampleRange
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_seq2seq import Seq2SeqTrainer

from expes.adapter_trainer import AdapterTrainer
from expes.chat_template import ChatTemplate
from expes.eval_pred_manager import (
    ChatEvalPredictionManager,
    EvalPredManager,
    Seq2SeqEvalPredManager,
)
from expes.trainer import Trainer
from expes.training_args import Seq2SeqTrainingArguments, TrainingArguments
from expes.types import SamplingStrategy


@dataclass()
class TunerConfig:
    metric: str
    mode: str
    num_samples: int = 1
    robustness_num_samples: int = 5
    grace_period: Optional[int] = None
    num_to_keep: Optional[int] = None


@dataclass()
class RessourcesConfig:
    num_workers: Optional[Union[int, SampleRange]] = None
    use_gpu: Optional[bool] = None
    max_concurrent_trials: Optional[int] = 1
    cpus_per_worker: Optional[int] = 1
    gpus_per_worker: Optional[int] = 1

    def __post_init__(self):
        self.use_gpu = self.use_gpu or torch.cuda.is_available()
        self.num_workers = self.num_workers or (
            torch.cuda.device_count if torch.cuda.is_available() else 1
        )


@dataclass
class DataConfig:
    get_datasets: Callable
    sampling_strategy: Optional[SamplingStrategy] = "balanced"
    tokenize_dataset: bool = False
    input_max_length: int = 128
    output_max_length: int = 128


@dataclass
class TrainingConfig:
    train_tasks: List[str] = field(default_factory=list)
    validation_tasks: List[str] = field(default_factory=list)
    test_tasks: List[str] = field(default_factory=list)
    data_config: Optional[DataConfig] = None
    model_class: Optional[Union[type, Type[PreTrainedModel]]] = None
    compute_metrics: Optional[Callable] = None
    pad_token: Optional[str] = None
    eval_pred_manager_cls: Optional[Type[EvalPredManager]] = None
    chat_template: Optional[ChatTemplate] = None
    is_causal_lm: bool = True
    model_config: Optional[PretrainedConfig] = None
    model_checkpoint: Optional[str] = None
    tokenizer_checkpoint: Optional[str] = None
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    adapter_configs: Dict = field(default_factory=dict)
    adapter_activation: Optional[Union[str, AdapterCompositionBlock]] = None
    generation_config: Dict = field(default_factory=dict)
    trainer_cls: Optional[Type[Trainer]] = None
    training_args_cls: Optional[Type[TrainingArguments]] = None
    training_kwargs: Dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.model_config or self.model_checkpoint
        if self.model_checkpoint is None:
            assert self.tokenizer_checkpoint
        if (
            self.tokenizer_checkpoint is None
            and self.model_checkpoint is not None
        ):
            self.tokenizer_checkpoint = self.model_checkpoint

        if self.trainer_cls is None:
            if self.adapter_configs is not None:
                self.trainer_cls = (
                    AdapterTrainer
                    if self.is_causal_lm
                    else Seq2SeqAdapterTrainer
                )
            else:
                self.trainer_cls = (
                    Trainer if self.is_causal_lm else Seq2SeqTrainer
                )

        if self.training_args_cls is None:
            if issubclass(self.trainer_cls, Trainer):
                self.training_args_cls = TrainingArguments
            elif issubclass(self.trainer_cls, Seq2SeqTrainer):
                self.training_args_cls = Seq2SeqTrainingArguments

        # set eval pred manager
        if self.eval_pred_manager_cls is None:
            if not self.is_causal_lm:
                self.eval_pred_manager_cls = Seq2SeqEvalPredManager
            elif self.chat_template:
                self.eval_pred_manager_cls = partial(
                    ChatEvalPredictionManager,
                    chat_template=self.chat_template,
                )
