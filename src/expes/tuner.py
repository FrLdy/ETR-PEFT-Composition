import os
from pathlib import Path

import torch
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.huggingface.transformers import (
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer
from ray.tune.schedulers.async_hyperband import ASHAScheduler

from expes.callbacks import (
    LogParametersTrainedCallback,
    TestModelEachEpochCallback,
)
from expes.tuner_factory import TunerFactories


class RayTuner:

    def __init__(self, factories: TunerFactories) -> None:
        self.factories = factories

    def train_func(self, config):
        tokenizer = self.factories.get_tokenizer(config)
        model = self.factories.get_model(config, tokenizer)

        data_collators = self.factories.get_datacollators(tokenizer, config)

        compute_metrics = self.factories.get_compute_metrics(tokenizer)

        datasets = self.factories.get_datasets(config, tokenizer)
        train_dataset = datasets.get("train")
        eval_dataset = datasets.get("validation")
        test_dataset = datasets.get("test")

        training_args = self.factories.get_training_args(config)

        trainer = self.factories.get_trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            **data_collators,
        )

        trainer.add_callback(LogParametersTrainedCallback(trainer))
        trainer.add_callback(
            TestModelEachEpochCallback(trainer, test_dataset=test_dataset)
        )
        trainer.add_callback(RayTrainReportCallback())
        trainer = prepare_trainer(trainer)
        trainer.train()

    def hp_search(
        self,
        storage_path,
        expe_name,
        metric,
        mode,
        num_samples=1,
        grace_period=None,
        num_to_keep=None,
        overwrite=False,
        num_workers=None,
        use_gpu=None,
        cpus_per_worker=1,
        gpus_per_worker=1,
    ):
        storage_path = Path(storage_path)
        path = (storage_path / expe_name).resolve().as_posix()

        num_workers = (
            num_workers
            or os.environ.get("SLURM_GPUS_ON_NODE")
            or (torch.cuda.device_count if torch.cuda.is_available() else 1)
        )
        use_gpu = use_gpu or torch.cuda.is_available()

        trainer = TorchTrainer(
            self.train_func,
            scaling_config=ScalingConfig(
                num_workers=num_workers,
                use_gpu=use_gpu,
                resources_per_worker={
                    "CPU": cpus_per_worker,
                    "GPU": gpus_per_worker if use_gpu else 0,
                },
            ),
        )

        if tune.Tuner.can_restore(path) and not overwrite:
            tuner = tune.Tuner.restore(
                path, trainable=trainer, resume_errored=True
            )
        else:
            param_space = self.factories.get_hp_space()
            tuner = tune.Tuner(
                trainer,
                param_space={"train_loop_config": param_space},
                tune_config=tune.TuneConfig(
                    metric=metric,
                    mode=mode,
                    num_samples=num_samples,
                    scheduler=(
                        ASHAScheduler(
                            grace_period=grace_period,
                        )
                        if grace_period
                        else None
                    ),
                ),
                run_config=RunConfig(
                    storage_path=storage_path.resolve().as_posix(),
                    name=expe_name,
                    checkpoint_config=CheckpointConfig(
                        num_to_keep=num_to_keep,
                        checkpoint_score_attribute=metric,
                        checkpoint_score_order=mode,
                    ),
                ),
            )

        return tuner.fit()
