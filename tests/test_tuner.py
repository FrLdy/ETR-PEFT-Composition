import unittest
from tempfile import TemporaryDirectory

import torch
from adapters.composition import MultiTask
from adapters.configuration.adapter_config import (
    MTLLoRAConfig,
    MultiTaskConfigUnion,
)
from ray import tune
from transformers.models.llama.configuration_llama import LlamaConfig

from expes.chat_template import causal_chat_template
from expes.tuner import RayTuner
from expes.tuner_factory import TunerFactories, TuningConfig

from .utils import get_mtl_dataset


def prepare_datasets(config: TuningConfig, tokenizer):
    return get_mtl_dataset(
        config.tasks,
        [torch.randint(10, 15, (3,)).tolist() for _ in config.tasks],
    )


class TestRayTuner(unittest.TestCase):

    def test_hp_search(self):
        tasks = ["a", "b", "c"]
        tuning_config = TuningConfig(
            prepare_dataset=prepare_datasets,
            model_config=LlamaConfig(num_hidden_layers=2),
            tokenizer_checkpoint="meta-llama/Llama-2-7b-hf",
            tasks=tasks,
            adapter_configs={
                "union1": MultiTaskConfigUnion(
                    task_names=tasks,
                    base_config=MTLLoRAConfig(r=tune.randint(2, 14)),
                )
            },
            adapter_activation=MultiTask(*tasks),
            pad_token="<pad>",
            chat_template=causal_chat_template,
            training_args={
                "lr": tune.choice([0.001, 0.003, 0.005]),
            },
        )
        with TemporaryDirectory() as tmpdirname:
            factories = TunerFactories(tuning_config)
            tuner = RayTuner(factories)
            result = tuner.hp_search(
                storage_path=tmpdirname,
                metric="eval_a_dummy_score",
                mode="max",
                expe_name="test_hp_search",
                num_samples=2,
                num_to_keep=2,
            )
            self.assertListEqual(result.errors, [])
