from dataclasses import asdict, dataclass
from functools import partial
from optparse import Option
from pathlib import Path
from typing import Dict, Optional

import ray
from adapters import AdapterConfig, init
from nltk import data
from ray import tune
from ray.train import CheckpointConfig, Result, RunConfig, ScalingConfig
from ray.train.huggingface.transformers import prepare_trainer
from ray.train.torch import TorchTrainer
from ray.tune.result_grid import ResultGrid
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import PREFIX_CHECKPOINT_DIR, Trainer

from expes import chat_template
from expes.callbacks import (
    LogParametersTrainedCallback,
    RayTrainReportCallback,
    TestModelEachEpochCallback,
)
from expes.config import (
    DataConfig,
    InferenceConfig,
    RessourcesConfig,
    TrainingConfig,
    TunerConfig,
)
from expes.datacollator import DataCollatorForSeq2SeqCausalLM
from expes.metric import compute_metrics


@dataclass()
class TunerResults:
    hp_search_results: Optional[ResultGrid] = None
    best_config_robustness_results: Optional[ResultGrid] = None
    best_config_inference_results: Optional[ResultGrid] = None


@dataclass
class TrainFuncFactories:

    training_config: TrainingConfig
    inference_config: Optional[InferenceConfig] = None

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
            tokenizer.eval_pred_manager = eval_pred_manager_cls(tokenizer=tokenizer)

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
        training_args = self.get_training_args(config)

        init(model)
        adapter_configs = config.adapter_configs
        for adapter_name, adapter_conf in adapter_configs.items():
            if isinstance(adapter_conf, str):
                model.load_adapter(adapter_conf, load_as=adapter_name)
            else:
                adapter_conf = AdapterConfig.load(adapter_conf)
                model.add_adapter(adapter_name, adapter_conf)

        model.active_adapters = config.adapter_activation or list(
            adapter_configs.keys()
        )

        if training_args.do_train:
            model.train_adapter(model.active_adapters)

        return model

    def get_datasets(self, config: TrainingConfig, tokenizer):
        return config.data_config.get_datasets(config, tokenizer)

    def get_datacollators(self, config: TrainingConfig, tokenizer, model):
        data_config: DataConfig = config.data_config
        if config.is_causal_lm:
            chat_template = config.chat_template
            tokenizer_checkpoint = config.tokenizer_checkpoint
            instruction_template_ids=chat_template.get_input_prefix_ids(tokenizer_checkpoint) or data_config.instruction_template_ids
            response_template_ids=chat_template.get_output_prefix_ids(tokenizer_checkpoint) or data_config.response_template_ids
            kwargs = dict(
                tokenizer=tokenizer,
                loss_completion_only=config.loss_completion_only,
                instruction_template=instruction_template_ids,
                response_template=response_template_ids,
            )
        
            return {
                "data_collator": DataCollatorForSeq2SeqCausalLM(
                    **kwargs,
                ),
                "eval_data_collator": DataCollatorForSeq2SeqCausalLM(
                    **kwargs,
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
        metric_fn = self.training_config.get_metrics_fn()

        return partial(
            compute_metrics, metrics_fn=metric_fn, tokenizer=tokenizer
        )

    def get_training_args(self, config: TrainingConfig):
        training_args = dict(
            output_dir=".",
            do_train=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            remove_unused_columns=False,
            predict_with_generate=True,
            include_for_metrics=["inputs"],
            push_to_hub=False,
            disable_tqdm=True,
            report_to="none",
        )
        training_args.update(**config.training_kwargs)
        return config.training_args_cls(**training_args)

    def get_trainer(self, config: TrainingConfig, **kwargs):
        test_dataset = kwargs.pop("test_dataset", None)
        trainer = config.trainer_cls(**kwargs)
        trainer.test_dataset = test_dataset
        trainer.add_callback(LogParametersTrainedCallback(trainer))
        if test_dataset is not None:
            trainer.add_callback(
                TestModelEachEpochCallback(trainer, test_dataset=test_dataset)
            )
        return trainer


    def __call__(self, config):
        data_config = self.training_config.data_config.__class__(
            **config.pop("data_config")
        )
        config = self.training_config.__class__(
            **config, data_config=data_config
        )
        tokenizer = self.get_tokenizer(config)
        model = self.get_model(config, tokenizer)
        data_collators = self.get_datacollators(
            config, tokenizer, model
        )

        compute_metrics = self.get_compute_metrics_fn(tokenizer)
        datasets = self.get_datasets(config, tokenizer)
        train_dataset = datasets.get("train")
        eval_dataset = datasets.get("validation")
        test_dataset = datasets.get("test")

        training_args = self.get_training_args(config)

        trainer = self.get_trainer(
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
        result_grid: ResultGrid = self.hp_search()
        best_result = result_grid.get_best_result(scope="all")
        best_checkpoint = best_result.get_best_checkpoint(
            metric=self.tuner_config.metric,
            mode=self.tuner_config.mode
        )
        best_config = best_result.config["train_loop_config"]

        robustness_results = (
            self._fit(
                expe_name=f"{self.expe_name}_robustness",
                param_space=best_config,
                num_samples=self.tuner_config.robustness_num_samples,
            )
            if self.tuner_config.robustness_num_samples 
            else None
        )
        
        inference_results = None
        if self.factories.inference_config is not None:
            _best_config: TrainingConfig = TrainingConfig(**best_config)
            checkpoint_path = Path(best_checkpoint.path)
            adapter_configs = _best_config.adapter_configs
            _best_config.adapter_configs = {
                adapter_name: (checkpoint_path / PREFIX_CHECKPOINT_DIR / adapter_name).resolve().as_posix()
                for adapter_name in adapter_configs
            }
            
            _best_config = _best_config.prepare_config_for_inference(self.factories.inference_config) 
            # print(_best_config) # OK

            self.tuner_config.metric = self.factories.inference_config.metric
            self.tuner_config.mode = self.factories.inference_config.mode
            inference_results = self._fit(
                expe_name=f"{self.expe_name}_test_best_model",
                param_space=asdict(_best_config),
                num_samples=self.factories.inference_config.n_samples,
            )

            

        return TunerResults(
            hp_search_results=result_grid,
            best_config_robustness_results=robustness_results,
            best_config_inference_results=inference_results,
        )

    def evaluate(self, trainer, split="eval"):
        metrics = {}
        datasets = getattr(trainer, f"{split}_dataset", {}) or {}
        for name, dataset in datasets.items():
            metrics.update(trainer.evaluate(eval_dataset=dataset, metric_key_prefix=f"{split}_{name}"))

        return metrics


    def train_func(self, config):
        trainer = self.factories(config)
        trainer.add_callback(RayTrainReportCallback())
        trainer = prepare_trainer(trainer)
        
        if trainer.args.do_train:
            trainer.train()
        elif trainer.args.do_eval:
            # TODO: eval with all datasets iteratively
            metrics = {
                **self.evaluate(trainer, "eval"),
                **self.evaluate(trainer, "test"),
            }
            if metrics: 
                ray.train.report(metrics=metrics, checkpoint=None)

    def hp_search(self):
        param_space = self.factories.get_hp_space()
        return self._fit(
            expe_name=f"{self.expe_name}_hp_search",
            param_space=param_space,
            num_samples=self.tuner_config.num_samples,
        )


    def retrain_with_best_config(self, best_config):
        return 

    def _fit(
        self,
        expe_name: str,
        param_space: Dict,
        num_samples: int = 0,
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
