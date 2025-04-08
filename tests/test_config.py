import unittest

from etr_fr_expes.config import get_inference_config
from etr_fr_expes.dataset import (
    DS_KEY_ETR_FR,
    DS_KEY_ETR_FR_POLITIC,
    DS_KEY_ORANGESUM,
    DS_KEY_WIKILARGE_FR,
)
from expes.config import InferenceConfig


class TestConfig(unittest.TestCase):
    def test_get_inference_config(self):
        train_tasks = [DS_KEY_ETR_FR, DS_KEY_ORANGESUM, DS_KEY_WIKILARGE_FR]

        inference_config = get_inference_config(train_tasks)

        expected_inference_config = InferenceConfig(
            validation_tasks=train_tasks,
            test_tasks=train_tasks,
            task_to_task_ids={
                DS_KEY_ETR_FR: 0,
                DS_KEY_ORANGESUM: 1,
                DS_KEY_WIKILARGE_FR: 2,
                DS_KEY_ETR_FR_POLITIC:0
            } 
        )

        self.assertEqual(inference_config, expected_inference_config) 
