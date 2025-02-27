import os
from copy import deepcopy
from dataclasses import asdict
from tempfile import TemporaryDirectory

import adapters
from adapters.composition import MultiTask
from ray import tune

from etr_fr_expes.dataset import (
    DS_KEY_ETR_FR,
    DS_KEY_ORANGESUM,
    DS_KEY_WIKILARGE_FR,
    prepare_datasets,
)
from etr_fr_expes.metric import etr_compute_metrics
from expes import RayTuner, TrainFuncFactories, TrainingConfig
from expes.config import DataConfig, TunerConfig


class BaseRayTunerTest:
    model_class = None
    model_config = None
    tokenizer_checkpoint = None
    training_config_special_kwargs = None
    data_config_special_kwargs = None

    tasks = [DS_KEY_ETR_FR, DS_KEY_ORANGESUM, DS_KEY_WIKILARGE_FR]
    configs_to_test = [
        (
            {
                "union_1": adapters.MultiTaskConfigUnion(
                    base_config=adapters.MTLLoRAConfig(
                        init_weights="bert",
                        r=tune.randint(2, 10),
                    ),
                    task_names=tasks,
                )
            },
            MultiTask(*tasks),
        ),
        (
            {
                "lora_1": adapters.LoRAConfig(
                    init_weights="bert",
                    r=tune.randint(2, 10),
                )
            },
            "lora_1",
        ),
    ]

    def get_training_config(self, adapter_configs, adapter_activation):
        training_config = TrainingConfig(
            data_config=DataConfig(
                get_dataset=prepare_datasets,
                n_samples=5,
                **self.data_config_special_kwargs,
            ),
            compute_metrics=etr_compute_metrics,
            model_class=self.model_class,
            model_config=self.model_config,
            tokenizer_checkpoint=self.tokenizer_checkpoint,
            tasks=self.tasks,
            adapter_configs=adapter_configs,
            adapter_activation=adapter_activation,
            training_kwargs={
                "learning_rate": tune.choice([0.001, 0.005, 0.323]),
                "num_train_epochs": 2,
            },
            generation_config={"max_new_tokens": 20},
            **self.training_config_special_kwargs or {},
        )
        return training_config

    def resolve_tune_search_space(self, config):
        if isinstance(config, dict):
            return {
                k: self.resolve_tune_search_space(v) for k, v in config.items()
            }
        elif isinstance(config, list):
            return [self.resolve_tune_search_space(v) for v in config]
        elif isinstance(config, tune.search.sample.Domain):
            return config.sample()
        else:
            return config

    def get_tuner(self, training_config, storage_path, expe_name):
        factories = TrainFuncFactories(training_config)
        tuner_config = TunerConfig(
            metric="eval_etr_fr_sari_rouge_bertf1_hmean",
            mode="max",
            num_samples=2,
            robustness_num_samples=2,
            num_to_keep=2,
        )
        return RayTuner(
            factories=factories,
            storage_path=storage_path,
            expe_name=expe_name,
            tuner_config=tuner_config,
        )

    def test_train_func(self):
        for adapter_configs, adapter_activation in self.configs_to_test:
            name = ";".join(list(adapter_configs.keys()))
            with self.subTest(name=name):
                with TemporaryDirectory() as tmpdirname:
                    training_config = deepcopy(
                        self.get_training_config(
                            adapter_configs, adapter_activation
                        )
                    )
                    tuner = self.get_tuner(
                        training_config, tmpdirname, "train_func_test"
                    )
                    os.chdir(tmpdirname)
                    tuner.train_func(
                        self.resolve_tune_search_space(asdict(training_config))
                    )

    def test_call(self):
        for adapter_configs, adapter_activation in self.configs_to_test:
            name = ";".join(list(adapter_configs.keys()))
            with self.subTest(name=name):
                with TemporaryDirectory() as tmpdirname:
                    training_config = deepcopy(
                        self.get_training_config(
                            adapter_configs, adapter_activation
                        )
                    )
                    tuner = self.get_tuner(
                        training_config, tmpdirname, "hp_search_test"
                    )
                    results = tuner()

                    with self.subTest(restype="hp_search_results"):
                        self.run_result_grid_tests(results.hp_search_results)

                    with self.subTest(restype="best_config_robustness_results"):
                        self.run_result_grid_tests(
                            results.best_config_robustness_results
                        )

    def run_result_grid_tests(self, result_grid):
        self.assertListEqual(result_grid.errors, [])

        required_metrics = [
            "loss",
            "epoch",
            "test_{task}_texts",
            "eval_{task}_texts",
            "test_{task}_loss",
            "eval_{task}_loss",
        ]
        for result in result_grid._results:
            for task in self.tasks:
                for required_metric in required_metrics:
                    self.assertIn(
                        required_metric.format(task=task), result.metrics
                    )

                test_texts = result.metrics[
                    "test_{task}_texts".format(task=task)
                ]
                eval_texts = result.metrics[
                    "eval_{task}_texts".format(task=task)
                ]
                self.assertNotEqual(test_texts["inputs"], eval_texts["inputs"])
                self.assertNotEqual(test_texts["labels"], eval_texts["labels"])
