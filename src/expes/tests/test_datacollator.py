import unittest

import numpy as np
from transformers.models.altclip.modeling_altclip import clip_loss
from transformers.models.auto.tokenization_auto import AutoTokenizer

from expes.chat_template import causal_chat_template
from expes.datacollator import DataCollatorForSeq2SeqCausalLM
from expes.tests.utils import lorem_ipsum_dataset
from expes.utils import replace_ignore_index


class TestDataCollatorForSeq2SeqCausalLM(unittest.TestCase):
    def build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1"
        )
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tokenizer = causal_chat_template.apply_to_tokenizer(tokenizer)
        return tokenizer

    def build_datacollator(self, eval_mode=True, loss_completion_only=False):
        collator = DataCollatorForSeq2SeqCausalLM(
            tokenizer=self.build_tokenizer(),
            eval_mode=eval_mode,
            loss_completion_only=loss_completion_only,
        )

        return collator

    def test_with_loss_completion_only(self):
        examples = list(lorem_ipsum_dataset(10))
        collator = self.build_datacollator(
            eval_mode=True, loss_completion_only=True
        )
        tokenizer = collator.tokenizer
        batch = collator(examples)
        labels = replace_ignore_index(
            batch["labels"],
            tokenizer.pad_token_id,
        )
        dec_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )
        assert all(
            label.strip() == example[collator.target_key].strip()
            for label, example in zip(dec_labels, examples)
        )

        dec_gen_inputs = tokenizer.batch_decode(
            batch["generation_input_ids"],
            skip_special_tokens=True,
        )
        assert all(
            gen_in.strip()
            == f"{tokenizer.input_prefix} {example[collator.source_key].strip()} {tokenizer.output_prefix}"
            for gen_in, example in zip(dec_gen_inputs, examples)
        )
