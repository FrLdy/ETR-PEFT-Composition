import unittest
from dataclasses import asdict
from tempfile import TemporaryDirectory

import torch
from adapters.composition import MultiTask
from adapters.configuration.adapter_config import (
    MTLLoRAConfig,
    MultiTaskConfigUnion,
)
from ray import tune
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from expes.chat_template import causal_chat_template
from expes.dataset import build_mtl_dataset
from expes.tuner import RayTuner
from expes.tuner_factory import TunerFactories, TuningConfig

from .utils import get_mtl_dataset


def prepare_datasets(config: TuningConfig, tokenizer):
    dataset = get_mtl_dataset(
        config.tasks,
        [torch.randint(10, 15, (3,)).tolist() for _ in config.tasks],
    )
    dataset = build_mtl_dataset(
        dataset, config.tasks, stopping_strategy="concatenate"
    )
    return dataset


class TestRayTuner(unittest.TestCase):

    def test_hp_search(self):
        tasks = ["a", "b", "c"]
        tuning_config = TuningConfig(
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
            },
        )
        with TemporaryDirectory() as tmpdirname:
            factories = TunerFactories(tuning_config)
            tuner = RayTuner(factories)
            tuner.train_func(asdict(tuning_config))
            # result = tuner.hp_search(
            #     storage_path=tmpdirname,
            #     metric="eval_a_dummy_score",
            #     mode="max",
            #     expe_name="test_hp_search",
            #     num_samples=2,
            #     num_to_keep=2,
            # )
            # self.assertListEqual(result.errors, [])
