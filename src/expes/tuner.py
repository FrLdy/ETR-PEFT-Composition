import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from ray import tune
from ray.air.config import SampleRange
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.huggingface.transformers import prepare_trainer
from ray.train.torch import TorchTrainer
from ray.tune.result_grid import ResultGrid
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from transformers.trainer import Trainer

from expes.callbacks import RayTrainReportCallback
from expes.tuner_factory import TrainFuncFactories


@dataclass()
class TunerResults:
    hp_search_results: Optional[ResultGrid] = None
    best_config_robustness_results: Optional[ResultGrid] = None


@dataclass()
class RayTunerConfig:
    metric: str
    mode: str
    num_samples: int = 1
    robustness_num_samples: int = 5
    grace_period: Optional[int] = None
    num_to_keep: Optional[int] = None
    overwrite: Optional[bool] = False
    num_workers: Optional[Union[int, SampleRange]] = None
    use_gpu: Optional[bool] = None
    max_concurrent_trials: Optional[int] = 1
    cpus_per_worker: Optional[int] = 1
    gpus_per_worker: Optional[int] = 1

    def __post_init__(self):
        self.num_workers = (
            self.num_workers
            or os.environ.get("SLURM_GPUS_ON_NODE")
            or (torch.cuda.device_count if torch.cuda.is_available() else 1)
        )

        self.cpus_per_worker = self.cpus_per_worker or os.environ.get(
            "SLURM_CPUS_PER_GPU"
        )


class RayTuner:

    def __init__(
        self,
        storage_path: Path,
        expe_name: str,
        tuner_config: RayTunerConfig,
        factories: TrainFuncFactories,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.expe_name = expe_name
        self.tuner_config = tuner_config
        self.factories = factories

    def __call__(self) -> TunerResults:
        hp_search_results = self.hp_search()
        robustness_results = None
        if self.tuner_config.robustness_num_samples:
            robustness_results = self.retrain_with_best_config(
                hp_search_results
            )

        return TunerResults(
            hp_search_results=hp_search_results,
            best_config_robustness_results=robustness_results,
        )

    def train_func(self, config):
        config = self.factories.tuning_config.__class__(**config)
        tokenizer = self.factories.get_tokenizer(config)
        model = self.factories.get_model(config, tokenizer)
        data_collators = self.factories.get_datacollators(
            config, tokenizer, model
        )

        compute_metrics = self.factories.get_compute_metrics(tokenizer)

        datasets = self.factories.get_datasets(config, tokenizer)
        train_dataset = datasets.get("train")
        eval_dataset = datasets.get("validation")
        test_dataset = datasets.get("test")

        training_args = self.factories.get_training_args(config)

        trainer = self.factories.get_trainer(
            config=config,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            **data_collators,
            test_dataset=test_dataset,
        )

        trainer.add_callback(RayTrainReportCallback())
        trainer = prepare_trainer(trainer)
        trainer.train()

    def hp_search(self):
        tuner_config = self.tuner_config
        param_space = self.factories.get_hp_space()
        return self._fit(
            tuner_config=tuner_config,
            expe_name=f"{self.expe_name}_hp_search",
            param_space=param_space,
        )

    def retrain_with_best_config(self, result_grid: tune.ResultGrid):
        tuner_config = deepcopy(self.tuner_config)
        tuner_config.num_samples = tuner_config.robustness_num_samples
        param_space = result_grid.get_best_result().config["train_loop_config"]
        return self._fit(
            tuner_config=tuner_config,
            expe_name=f"{self.expe_name}_robustness",
            param_space=param_space,
        )

    def _fit(
        self,
        tuner_config: RayTunerConfig,
        expe_name: str,
        param_space: Optional[Dict] = None,
    ):
        path = (self.storage_path / expe_name).resolve().as_posix()
        use_gpu = tuner_config.use_gpu or torch.cuda.is_available()

        trainer = TorchTrainer(
            self.train_func,
            scaling_config=ScalingConfig(
                num_workers=tuner_config.num_workers,
                use_gpu=use_gpu,
                resources_per_worker={
                    "CPU": tuner_config.cpus_per_worker,
                    "GPU": tuner_config.gpus_per_worker if use_gpu else 0,
                },
            ),
        )

        if tune.Tuner.can_restore(path) and not tuner_config.overwrite:
            tuner = tune.Tuner.restore(
                path, trainable=trainer, resume_errored=True
            )
        else:
            tuner = self.get_tuner(
                trainer,
                param_space,
                expe_name,
                tuner_config,
            )

        return tuner.fit()

    def get_tuner(
        self,
        trainer: Trainer,
        param_space: Dict,
        expe_name: Optional[str] = None,
        tuner_config: Optional[RayTunerConfig] = None,
    ):
        tuner_config = tuner_config or self.tuner_config
        tuner = tune.Tuner(
            trainer,
            param_space={"train_loop_config": param_space},
            tune_config=tune.TuneConfig(
                metric=tuner_config.metric,
                mode=tuner_config.mode,
                max_concurrent_trials=tuner_config.max_concurrent_trials,
                num_samples=tuner_config.num_samples,
                scheduler=(
                    ASHAScheduler(
                        grace_period=tuner_config.grace_period,
                    )
                    if tuner_config.grace_period
                    else None
                ),
            ),
            run_config=RunConfig(
                storage_path=self.storage_path.resolve().as_posix(),
                name=expe_name or self.expe_name,
                checkpoint_config=CheckpointConfig(
                    num_to_keep=tuner_config.num_to_keep,
                    checkpoint_score_attribute=tuner_config.metric,
                    checkpoint_score_order=tuner_config.mode,
                ),
            ),
        )
        return tuner
