import unittest
from functools import partial

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from expes.tuner_factory import TuningConfig


class TestTuningConfig(unittest.TestCase):
    partial_config = partial(
        TuningConfig,
        prepare_dataset=lambda x: x,
        model_class=PreTrainedModel,
    )

    def test_init(self):
        with self.assertRaises(AssertionError):
            self.partial_config()

        with self.assertRaises(AssertionError):
            self.partial_config(model_config=PretrainedConfig())

        model_checkpoint = "checkpoint"
        config = self.partial_config(model_checkpoint=model_checkpoint)
        self.assertTrue(config.tokenizer_checkpoint == model_checkpoint)

        tokenizer_checkpoint = "tokenizer_checkpoint"
        config = self.partial_config(
            model_checkpoint=model_checkpoint,
            tokenizer_checkpoint=tokenizer_checkpoint,
        )
        self.assertTrue(config.tokenizer_checkpoint == tokenizer_checkpoint)
        self.assertTrue(config.model_checkpoint == model_checkpoint)
