import unittest
from functools import partial

from transformers import MBartConfig

from expes.models.mbart import MBartForConditionalGenerationAdapterModel
from expes.tests.models.base import AdapterModelBaseTestMixin
from expes.tests.utils import get_trainable_param_names


class TestMBartAdapter(AdapterModelBaseTestMixin, unittest.TestCase):
    def build_model(self):
        model = MBartForConditionalGenerationAdapterModel.from_pretrained(
            "facebook/mbart-large-50"
        )
        return model

    def test_add_MTL_lora(self):
        union_name, adapter_names, model = self.build_MTL_lora_model()
        attn_matrices = model.adapters_config.get(
            union_name
        ).base_config.attn_matrices

        self.apply_test_on_attn_matrices(
            model.model.encoder,
            attn_matrices,
            partial(self.run_test_add_MTL_lora, adapter_names, union_name),
        )
        self.apply_test_on_attn_matrices(
            model.model.decoder,
            attn_matrices,
            partial(self.run_test_add_MTL_lora, adapter_names, union_name),
        )

    def test_del_MTL_lora(self):
        union_name, adapter_names, model = self.build_MTL_lora_model()
        attn_matrices = model.adapters_config.get(
            union_name
        ).base_config.attn_matrices
        model.delete_adapter(union_name)
        self.apply_test_on_attn_matrices(
            model.model.encoder,
            attn_matrices,
            partial(self.run_test_del_MTL_lora, adapter_names, union_name),
        )
        self.apply_test_on_attn_matrices(
            model.model.decoder,
            attn_matrices,
            partial(self.run_test_del_MTL_lora, adapter_names, union_name),
        )
