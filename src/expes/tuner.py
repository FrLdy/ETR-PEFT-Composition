from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Optional

from adapters import AdapterConfig, init
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.huggingface.transformers import prepare_trainer
from ray.train.torch import TorchTrainer
from ray.tune.result_grid import ResultGrid
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer

from expes.callbacks import (
    LogParametersTrainedCallback,
    RayTrainReportCallback,
    TestModelEachEpochCallback,
)
from expes.config import (
    DataConfig,
    RessourcesConfig,
    TrainingConfig,
    TunerConfig,
)
from expes.datacollator import DataCollatorForSeq2SeqCausalLM


@dataclass()
class TunerResults:
    hp_search_results: Optional[ResultGrid] = None
    best_config_robustness_results: Optional[ResultGrid] = None


@dataclass
class TrainFuncFactories:

    training_config: TrainingConfig

    def get_hp_space(self, **kwargs):
        return asdict(self.training_config)

    def add_pad_token(self, config: TrainingConfig, tokenizer):
        return (
            config.pad_token is not None
            and tokenizer.pad_token is None
            and config.pad_token not in tokenizer.get_vocab()
        )

    def get_tokenizer(self, config: TrainingConfig):
        checkpoint = config.tokenizer_checkpoint or config.model_checkpoint
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, **config.tokenizer_kwargs
        )
        chat_template = config.chat_template
        eval_pred_manager_cls = config.eval_pred_manager_cls

        if self.add_pad_token(config, tokenizer):
            tokenizer.add_special_tokens({"pad_token": config.pad_token})

        if chat_template is not None:
            tokenizer = chat_template.apply_to_tokenizer(tokenizer)

        if eval_pred_manager_cls is not None:
            tokenizer.eval_pred_manager = eval_pred_manager_cls(tokenizer)

        return tokenizer

    def get_model(self, config: TrainingConfig, tokenizer):
        if config.model_config:
            model = config.model_class(config.model_config)
        else:
            model = config.model_class.from_pretrained(config.model_checkpoint)

        if config.pad_token:
            model.resize_token_embeddings(len(tokenizer))
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        if config.adapter_configs and config.adapter_activation:
            model = self.setup_adapters(model, config)

        if config.generation_config:
            unused_params = model.generation_config.update(
                **config.generation_config
            )
            assert unused_params == {}

        return model

    def setup_adapters(self, model, config: TrainingConfig):
        init(model)
        adapter_configs = config.adapter_configs
        for adapter_name, adapter_conf in adapter_configs.items():
            adapter_conf = AdapterConfig.load(adapter_conf)
            model.add_adapter(adapter_name, adapter_conf)
        model.active_adapters = config.adapter_activation or list(
            adapter_configs.keys()
        )
        model.train_adapter(model.active_adapters)

        return model

    def get_datasets(self, config: TrainingConfig, tokenizer):
        return config.data_config.get_dataset(config, tokenizer)

    def get_datacollators(self, config: TrainingConfig, tokenizer, model):
        if config.is_causal_lm:
            return {
                "data_collator": DataCollatorForSeq2SeqCausalLM(
                    tokenizer=tokenizer,
                    loss_completion_only=True,
                ),
                "eval_data_collator": DataCollatorForSeq2SeqCausalLM(
                    tokenizer=tokenizer,
                    loss_completion_only=True,
                    eval_mode=True,
                ),
            }
        return {
            "data_collator": DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                return_tensors="pt",
            )
        }

    def get_compute_metrics_fn(self, tokenizer):
        return partial(
            self.training_config.compute_metrics, tokenizer=tokenizer
        )

    def get_training_args(self, config: TrainingConfig):
        return config.training_args_cls(
            output_dir=".",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            remove_unused_columns=False,
            predict_with_generate=True,
            include_for_metrics=["inputs"],
            push_to_hub=False,
            disable_tqdm=True,
            report_to="none",
            **config.training_kwargs,
        )

    def get_trainer(self, config: TrainingConfig, **kwargs):
        test_dataset = kwargs.pop("test_dataset", None)
        trainer = config.trainer_cls(**kwargs)
        trainer.add_callback(LogParametersTrainedCallback(trainer))
        if test_dataset is not None:
            trainer.add_callback(
                TestModelEachEpochCallback(trainer, test_dataset=test_dataset)
            )
        return trainer


class RayTuner:

    def __init__(
        self,
        storage_path: Path,
        expe_name: str,
        tuner_config: TunerConfig,
        factories: TrainFuncFactories,
        ressources_config: RessourcesConfig = RessourcesConfig(),
        overwrite: bool = False,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.expe_name = expe_name
        self.overwrite = overwrite
        self.tuner_config = tuner_config
        self.ressources_config = ressources_config
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
        # TODO : use automatic class finder to recreate dataclass object
        data_config = DataConfig(**config.pop("data_config"))
        config = self.factories.training_config.__class__(
            **config, data_config=data_config
        )
        tokenizer = self.factories.get_tokenizer(config)
        model = self.factories.get_model(config, tokenizer)
        data_collators = self.factories.get_datacollators(
            config, tokenizer, model
        )

        compute_metrics = self.factories.get_compute_metrics_fn(tokenizer)
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
        param_space = self.factories.get_hp_space()
        return self._fit(
            expe_name=f"{self.expe_name}_hp_search",
            param_space=param_space,
            num_samples=self.tuner_config.num_samples,
        )

    def retrain_with_best_config(self, result_grid: tune.ResultGrid):
        param_space = result_grid.get_best_result().config["train_loop_config"]
        return self._fit(
            expe_name=f"{self.expe_name}_robustness",
            param_space=param_space,
            num_samples=self.tuner_config.robustness_num_samples,
        )

    def _fit(
        self,
        expe_name: str,
        param_space: Dict,
        num_samples: int,
    ):
        path = (self.storage_path / expe_name).resolve().as_posix()

        trainer = TorchTrainer(
            self.train_func,
            scaling_config=ScalingConfig(
                num_workers=self.ressources_config.num_workers,
                use_gpu=self.ressources_config.use_gpu,
                resources_per_worker={
                    "CPU": self.ressources_config.cpus_per_worker,
                    "GPU": (
                        self.ressources_config.gpus_per_worker
                        if self.ressources_config.use_gpu
                        else 0
                    ),
                },
            ),
        )

        if tune.Tuner.can_restore(path) and not self.overwrite:
            tuner = tune.Tuner.restore(
                path, trainable=trainer, resume_errored=True
            )
        else:
            tuner = self.get_tuner(
                trainer=trainer,
                param_space=param_space,
                expe_name=expe_name,
                num_samples=num_samples,
            )

        return tuner.fit()

    def get_tuner(
        self,
        trainer: Trainer,
        param_space: Dict,
        expe_name: str,
        num_samples: int,
    ):
        tuner = tune.Tuner(
            trainer,
            param_space={"train_loop_config": param_space},
            tune_config=tune.TuneConfig(
                metric=self.tuner_config.metric,
                mode=self.tuner_config.mode,
                max_concurrent_trials=self.ressources_config.max_concurrent_trials,
                num_samples=num_samples,
                scheduler=(
                    ASHAScheduler(
                        grace_period=self.tuner_config.grace_period,
                    )
                    if self.tuner_config.grace_period
                    else None
                ),
            ),
            run_config=RunConfig(
                storage_path=self.storage_path.resolve().as_posix(),
                name=expe_name,
                checkpoint_config=CheckpointConfig(
                    num_to_keep=self.tuner_config.num_to_keep,
                    checkpoint_score_attribute=self.tuner_config.metric,
                    checkpoint_score_order=self.tuner_config.mode,
                ),
            ),
        )
        return tuner
