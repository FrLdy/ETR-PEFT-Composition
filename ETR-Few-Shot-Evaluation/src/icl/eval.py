import logging
from typing import Type

from datasets import DatasetDict
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines import pipeline

from etr_fr.dataset import (
    AVAILABLE_DATASETS,
    DST_KEY,
    SRC_KEY,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from etr_fr.metrics import ETRMetrics
from icl.embedding_index import EmbeddingIndex

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # or INFO depending on your needs
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


class EvalLLM:
    def __init__(
        self,
        config: "EvalLLMConfig",
    ) -> None:
        self.config = config
        self.metrics = ETRMetrics(config.lang)
        self.llm = self.config.llm_cls(config)
        self.post_init()
        self.prompt_manager = self.config.prompt_manager_cls(self)

    def post_init(self):
        self.datasets = self.load_datasets()

    @property
    def run_name(self):
        return f"{self.config.model_name}"

    def load_datasets(self):
        datasets = {}
        tasks = self.config.tasks
        tasks_by_split = {
            TRAIN_SPLIT: self.config.train_tasks,
            VAL_SPLIT: self.config.eval_tasks,
            TEST_SPLIT: self.config.test_tasks,
        }
        for name in tasks:
            if name in AVAILABLE_DATASETS:
                dataset_dict = AVAILABLE_DATASETS.get(name)()
                for split, dataset in dataset_dict.items():
                    if name in tasks_by_split[split]:
                        datasets.setdefault(split, {})
                        datasets[split][name] = dataset

        return DatasetDict(datasets)

    def generate(self, **kwargs):
        src = kwargs.pop(SRC_KEY)
        prompt = self.prompt_manager.get_system_prompt(**kwargs)
        input = self.prompt_manager.get_user_prompt(src, **kwargs)
        pred = self.llm.generate(prompt=prompt, input=input)
        pred = self.prompt_manager.get_pred(pred)
        return pred

    def evaluate(self, prefix, task, dataset):
        inputs, labels, predictions = [], [], []

        for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
            input = sample[SRC_KEY]
            label = sample[DST_KEY]

            logger.debug(f"Sample {idx}: Input={input}, Label={label}")

            pred = self.generate(**sample, task=task)

            logger.debug(f"Sample {idx}: Prediction={pred}")

            inputs.append(input)
            labels.append(label)
            predictions.append(pred)

        metrics = self.metrics.compute(
            predictions=predictions, references=labels, sources=inputs
        )
        metrics.update(
            {
                "texts": {
                    "inputs": inputs,
                    "labels": labels,
                    "predictions": predictions,
                }
            }
        )

        metrics = {f"{prefix}_{task}_{k}": v for k, v in metrics.items()}
        return metrics

    def run(self):
        metrics = {}
        for task in self.config.eval_tasks:
            dataset = self.datasets[VAL_SPLIT][task]
            metrics.update(self.evaluate("eval", task, dataset))

        for task in self.config.test_tasks:
            dataset = self.datasets[TEST_SPLIT][task]
            metrics.update(self.evaluate("test", task, dataset))

        return metrics


class EvalLLMFewShot(EvalLLM):

    def post_init(self):
        super().post_init()
        self.embedding_indexes = self.load_embedding_indexes()

    def load_embedding_indexes(self):
        indexes_by_split = {}
        for split, dataset_dict in self.datasets.items():
            indexes_by_split.setdefault(split, {})
            for task, ds in list(dataset_dict.items()):
                index = EmbeddingIndex(
                    dataset_name=task, split=split, dataset=ds
                )
                try:
                    index.load_index()
                except AssertionError:
                    index.build_index()
                    index.cache_index()
                indexes_by_split[split][task] = index
                self.datasets[split][task] = index.dataset

        return indexes_by_split
