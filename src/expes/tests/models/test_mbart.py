import unittest

from transformers import MBartConfig

from expes.models.mbart import MBartForConditionalGenerationAdapterModel
from expes.tests.models.base import ModelBaseTestMixin


class TestMBart(ModelBaseTestMixin, unittest.TestCase):
    def build_model(self):
        model = MBartForConditionalGenerationAdapterModel.from_pretrained(
            "facebook/mbart-large-50"
        )
        return model
