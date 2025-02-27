import unittest

from transformers import (
    AutoConfig,
    AutoTokenizer,
    MBartConfig,
    MBartForConditionalGeneration,
)
from transformers.data.data_collator import DataCollatorForSeq2Seq

from expes.eval_pred_manager import Seq2SeqEvalPredManager

from .base import BaseTestTrainer


class TestMBart(unittest.TestCase, BaseTestTrainer):
    is_seq2seq = True
    checkpoint = "facebook/mbart-large-50"

    def get_datasets(self, tasks=None, tokenizer=None):

        def tokenize_function(batch):
            inputs = tokenizer(
                batch["src"],
                truncation=True,
                max_length=128,
            )

            targets = tokenizer(
                text_target=batch["dst"],
                truncation=True,
                max_length=128,
            )
            inputs["labels"] = targets["input_ids"]

            return inputs

        return [
            dataset.map(
                tokenize_function, batched=True, remove_columns=["src", "dst"]
            )
            for dataset in super().get_datasets(
                tasks=tasks, tokenizer=tokenizer
            )
        ]

    def get_model_tokenizer(self):

        config = AutoConfig.from_pretrained(self.checkpoint)
        config.update(
            MBartConfig(
                d_model=16,
                encoder_layers=2,
                decoder_layers=2,
                encoder_attention_heads=4,
                decoder_attention_heads=4,
                encoder_ffn_dim=4,
                decoder_ffn_dim=4,
                vocab_size=250027,
            ).to_dict()
        )
        model = MBartForConditionalGeneration(config)

        tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint,
            src_lang="en_XX",
            tgt_lang="en_XX",
        )
        tokenizer.eval_pred_manager = Seq2SeqEvalPredManager(tokenizer)

        return model, tokenizer

    def trainer_kwargs(self, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        model = kwargs.get("model")
        return {
            "data_collator": DataCollatorForSeq2Seq(
                tokenizer=tokenizer, model=model
            )
        }
