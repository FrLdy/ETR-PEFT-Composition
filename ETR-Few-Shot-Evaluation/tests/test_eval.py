import unittest

from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaConfig,
)
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.utils.dummy_pt_objects import AutoModelForCausalLM

from etr_fr.dataset import DS_KEY_ETR_FR, DS_KEY_ORANGESUM, DS_KEY_WIKILARGE_FR
from icl.eval import EvalLLM, EvalLLMConfig, EvalLocalLLM
from icl.prompts import PromptTemplate

TEST_CONFIGS = {
    "llama": LlamaConfig(
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
    )
}

class EvalLLMTest(EvalLocalLLM):
    def __init__(self, config: EvalLLMConfig) -> None:
        super().__init__(config)

    def _load_model(self):
        config = AutoConfig.from_pretrained(self.config.model_name)
        config.update(
            TEST_CONFIGS[self.config.model_name]
        )
        model = AutoModelForCausalLM.from_config(config)
        return model

class TestEvalLLM(unittest.TestCase):

    def test_load_datasets(self):
        datasets = self.eval_llm.load_datasets()
        self.assertEqual(set(["train", "test", "validation"]), set(datasets.keys()))
        for k in self.config.datasets:
            for _, dataset_dict in datasets.items():
                self.assertTrue(k in dataset_dict)

