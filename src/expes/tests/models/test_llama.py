import unittest

import torch
from transformers import LlamaConfig
from transformers.testing_utils import torch_device

from expes.models.llama import LlamaForCausalLMAdapterModel
from expes.tests.models.base import ModelBaseTestMixin
from expes.tests.utils import ids_tensor


class TestLlama(ModelBaseTestMixin, unittest.TestCase):
    def build_model(self):
        config = LlamaConfig(
            hidden_size=32,
            num_hidden_layers=5,
            num_attention_heads=4,
            intermediate_size=37,
            hidden_act="gelu",
            pad_token_id=0,
        )
        model = LlamaForCausalLMAdapterModel(config)
        return model

    def inputs(self, bsz=1, n_tasks=0):
        inputs = {
            "input_ids": ids_tensor((bsz, 128), 1000).to(torch_device),
            "labels": ids_tensor((bsz, 128), 1000).to(torch_device),
        }
        if n_tasks > 0:
            inputs["task_ids"] = torch.randint(0, n_tasks, (bsz,))

        return inputs
