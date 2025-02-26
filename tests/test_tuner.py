import os
import unittest
from copy import deepcopy
from dataclasses import asdict
from tempfile import TemporaryDirectory

from adapters.composition import MultiTask
from adapters.configuration.adapter_config import (
    MTLLoRAConfig,
    MultiTaskConfigUnion,
)
from ray import tune
from transformers import LlamaConfig, LlamaForCausalLM

from expes.chat_template import causal_chat_template
from expes.dataset import (
    DS_KEY_ETR_FR,
    DS_KEY_ORANGESUM,
    DS_KEY_WIKILARGE_FR,
    build_mtl_dataset,
    get_datasets,
)
from expes.tuner import RayTuner, RayTunerConfig
from expes.tuner_factory import TrainFuncFactories, TrainingConfig


def prepare_datasets(config: TrainingConfig, tokenizer):
    datasets = get_datasets(tasks=config.tasks)
    for split in datasets:
        for task in datasets[split]:
            datasets[split][task] = datasets[split][task].select(range(5))

    return build_mtl_dataset(datasets, stopping_strategy="concatenate")


class TestRayTuner(unittest.TestCase):
    tasks = [DS_KEY_ETR_FR, DS_KEY_ORANGESUM, DS_KEY_WIKILARGE_FR]
    tuning_config = TrainingConfig(
        prepare_dataset=prepare_datasets,
        model_class=LlamaForCausalLM,
        model_config=LlamaConfig(
            hidden_size=32,
            num_hidden_layers=5,
            num_attention_heads=4,
            intermediate_size=37,
            hidden_act="gelu",
        ),
        tokenizer_checkpoint="meta-llama/Llama-2-7b-hf",
        tasks=tasks,
        adapter_configs={
            "union1": MultiTaskConfigUnion(
                task_names=tasks,
                base_config=MTLLoRAConfig(r=2),
            )
        },
        adapter_activation=MultiTask(*tasks),
        pad_token="<pad>",
        chat_template=causal_chat_template,
        training_args={
            "learning_rate": 0.001,
            "num_train_epochs": 2,
        },
        generation_config={"max_new_tokens": 20},
    )

    def test_train_func(self):
        with TemporaryDirectory() as tmpdirname:
            factories = TrainFuncFactories(self.tuning_config)
            tuner_config = RayTunerConfig(
                metric="eval_etr_fr_sari_rouge_bertf1_hmean",
                mode="max",
                num_samples=2,
                robustness_num_samples=2,
                num_to_keep=2,
            )
            tuner = RayTuner(
                factories=factories,
                storage_path=tmpdirname,
                expe_name="test_hp_search",
                tuner_config=tuner_config,
            )
            os.chdir(tmpdirname)
            tuner.train_func(asdict(self.tuning_config))

    def test_call(self):
        with TemporaryDirectory() as tmpdirname:
            tuning_config = deepcopy(self.tuning_config)
            tuning_config.adapter_configs["union1"] = MultiTaskConfigUnion(
                task_names=self.tasks,
                base_config=MTLLoRAConfig(r=tune.randint(10, 50)),
            )
            tuning_config.training_args["learning_rate"] = tune.choice(
                [0.001, 0.005, 0.323]
            )
            factories = TrainFuncFactories(self.tuning_config)
            tuner_config = RayTunerConfig(
                metric="eval_etr_fr_sari_rouge_bertf1_hmean",
                mode="max",
                num_samples=2,
                robustness_num_samples=2,
                num_to_keep=2,
            )
            tuner = RayTuner(
                factories=factories,
                storage_path=tmpdirname,
                expe_name="test_hp_search",
                tuner_config=tuner_config,
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
