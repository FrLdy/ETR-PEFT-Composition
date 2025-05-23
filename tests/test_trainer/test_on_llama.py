import unittest

from fsspec.utils import tokenize
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
)

from expes.chat_template import ChatTemplate
from expes.datacollator import DataCollatorForSeq2SeqCausalLM

from .base import BaseTestTrainer


class TestLlama27BTrainer(unittest.TestCase, BaseTestTrainer):
    is_seq2seq = False
    checkpoint = "meta-llama/Llama-2-7b-hf"

    def get_model_tokenizer(self):

        config = AutoConfig.from_pretrained(self.checkpoint)
        config.update(
            LlamaConfig(
                hidden_size=32,
                num_hidden_layers=5,
                num_attention_heads=4,
                intermediate_size=37,
                hidden_act="gelu",
            ).to_dict()
        )
        model = LlamaForCausalLM(config)

        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tokenizer = ChatTemplate().apply_to_tokenizer(tokenizer)
        model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    def trainer_kwargs(self, **kwargs):
        tokenizer = kwargs.get("tokenizer")
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


class TestLlama323BTrainer(TestLlama27BTrainer):
    checkpoint = "meta-llama/Llama-3.2-3B"

    def trainer_kwargs(self, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        instruction_template = [14711, 5688, 25]
        response_template = [17010, 9442, 25]
        return {
            "data_collator": DataCollatorForSeq2SeqCausalLM(
                tokenizer=tokenizer,
                loss_completion_only=True,
                instruction_template=instruction_template,
                response_template=response_template,
            ),
            "eval_data_collator": DataCollatorForSeq2SeqCausalLM(
                tokenizer=tokenizer,
                loss_completion_only=True,
                eval_mode=True,
                instruction_template=instruction_template,
                response_template=response_template,
            ),
        }


class TestLlama318BTrainer(TestLlama27BTrainer):
    checkpoint = "meta-llama/Llama-3.1-8B"

    def trainer_kwargs(self, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        instruction_template = [14711, 5688, 25]
        response_template = [17010, 9442, 25]
        return {
            "data_collator": DataCollatorForSeq2SeqCausalLM(
                tokenizer=tokenizer,
                loss_completion_only=True,
                instruction_template=instruction_template,
                response_template=response_template,
            ),
            "eval_data_collator": DataCollatorForSeq2SeqCausalLM(
                tokenizer=tokenizer,
                loss_completion_only=True,
                eval_mode=True,
                instruction_template=instruction_template,
                response_template=response_template,
            ),
        }
