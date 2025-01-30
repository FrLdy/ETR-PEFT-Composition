import unittest

import torch
from transformers import MistralConfig
from transformers.testing_utils import torch_device

from expes.models.mistral import MistralForCausalLMAdapterModel
from expes.tests.models.base import ModelBaseTestMixin
from expes.tests.utils import ids_tensor


class TestMistral(ModelBaseTestMixin, unittest.TestCase):
    def build_model(self):
        config = MistralConfig(
            hidden_size=32,
            num_hidden_layers=5,
            num_attention_heads=8,
            intermediate_size=37,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
        )
        model = MistralForCausalLMAdapterModel(config)
        return model

    def inputs(self, bsz=1, n_tasks=0):
        inputs = {
            "input_ids": ids_tensor((bsz, 128), 1000).to(torch_device),
            "labels": ids_tensor((bsz, 128), 1000).to(torch_device),
        }
        if n_tasks > 0:
            inputs["task_ids"] = torch.randint(0, n_tasks, (bsz,))

        return inputs
