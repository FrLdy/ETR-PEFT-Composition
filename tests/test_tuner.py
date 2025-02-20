import unittest
from tempfile import TemporaryDirectory

import torch
from adapters import init
from adapters.composition import MultiTask
from adapters.configuration.adapter_config import (
    MTLLoRAConfig,
    MultiTaskConfigUnion,
)
from adapters.context import ForwardContext
from ray import tune
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from expes.adapter_trainer import AdapterTrainer
from expes.chat_template import ChatTemplate
from expes.datacollator import DataCollatorForSeq2SeqCausalLM
from expes.metric import TEXT_METRIC_KEY
from expes.training_args import TrainingArguments
from expes.tuner import RayTuner, TunerFactories

from .utils import get_dataset, lorem_ipsum_dataset


class LlamaTunerFactories(TunerFactories):
    checkpoint = "meta-llama/Llama-2-7b-hf"
    trainer_cls = AdapterTrainer
    training_args_cls = TrainingArguments

    def get_hp_space(self, **kwargs):
        return {
            "r": tune.randint(4, 50),
            "task_names": ["a", "b", "c"],
            "training_args": {"num_train_epochs": 2},
        }

    def get_tokenizer(self, config):

        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tokenizer = ChatTemplate().apply_to_tokenizer(tokenizer)

        return tokenizer

    def get_model(self, config, tokenizer):
        model_config = AutoConfig.from_pretrained(self.checkpoint)
        model_config.update(
            LlamaConfig(
                hidden_size=32,
                num_hidden_layers=5,
                num_attention_heads=4,
                intermediate_size=37,
                hidden_act="gelu",
            ).to_dict()
        )
        model = LlamaForCausalLM(model_config)
        model.resize_token_embeddings(len(tokenizer))

        ForwardContext.context_args.add("task_ids")
        init(model)
        adapter_name = "union_1"
        model.add_adapter(
            adapter_name,
            MultiTaskConfigUnion(
                base_config=MTLLoRAConfig(r=config["r"]),
                task_names=config["task_names"],
            ),
        )
        model.set_active_adapters(adapter_name)
        model.train_adapter(adapter_name)

        return model

    def get_datacollators(self, tokenizer, config, **kwargs):
        return {
            "data_collator": DataCollatorForSeq2SeqCausalLM(
                tokenizer=tokenizer,
                loss_completion_only=True,
            ),
            "eval_data_collator": DataCollatorForSeq2SeqCausalLM(
                tokenizer=tokenizer,
                loss_completion_only=True,
                eval_mode=True,
            ),
        }

    def get_train_dataset(self, config, tokenizer):
        n_tasks = len(config["task_names"])
        dataset = lorem_ipsum_dataset(25)
        dataset_size = len(dataset)
        dataset = dataset.add_column(
            "task_ids",
            torch.randint(0, n_tasks, (dataset_size,)).tolist(),
        )
        return dataset

    def get_eval_dataset(self, config, tokenizer):
        tasks = config["task_names"]
        eval_dataset = {t: get_dataset(task_id=i) for i, t in enumerate(tasks)}
        return eval_dataset

    def get_test_dataset(self, config, tokenizer):
        raise NotImplementedError

    def get_compute_metrics(self, tokenizer):
        def compute_metrics(eval_pred):
            if hasattr(tokenizer, "eval_pred_manager"):
                eval_pred = tokenizer.eval_pred_manager(eval_pred)

            return {
                "dummy_score": 1.0,
                TEXT_METRIC_KEY: {
                    "inputs": eval_pred.inputs,
                    "labels": eval_pred.label_ids,
                    "predictions": eval_pred.predictions,
                },
            }

        return compute_metrics


class TestRayTuner(unittest.TestCase):
    def test_hp_search(self):
        with TemporaryDirectory() as tmpdirname:
            ray_tune = RayTuner(LlamaTunerFactories())
            ray_tune.hp_search(
                storage_path=tmpdirname,
                metric="eval_a_dummy_score",
                mode="max",
                expe_name="test_hp_search",
                num_samples=4,
                num_to_keep=2,
            )
            __import__("pdb").set_trace()
