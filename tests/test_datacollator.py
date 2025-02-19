import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.models.auto.tokenization_auto import AutoTokenizer

from expes.chat_template import causal_chat_template
from expes.datacollator import DataCollatorForSeq2SeqCausalLM
from expes.utils import replace_ignore_index
from tests.utils import lorem_ipsum_dataset


class TestDataCollatorForSeq2SeqCausalLM(unittest.TestCase):

    def get_dataset(self):
        dataset = lorem_ipsum_dataset(10)
        dataset = dataset.add_column(
            "new_var", torch.randint(0, 10, (len(dataset),)).tolist()
        )
        return dataset

    def build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
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
        dataset = self.get_dataset()
        collator = self.build_datacollator(eval_mode=True, loss_completion_only=True)
        tokenizer = collator.tokenizer
        bsz = 2
        raw_dataloader = DataLoader(dataset, batch_size=bsz)
        dataloader = DataLoader(dataset, batch_size=bsz, collate_fn=collator)
        for raw_batch, batch in zip(raw_dataloader, dataloader):

            labels = replace_ignore_index(
                batch["labels"],
                tokenizer.pad_token_id,
            )
            dec_labels = tokenizer.batch_decode(
                labels,
                skip_special_tokens=True,
            )
            dec_inputs = tokenizer.batch_decode(
                batch["input_ids"],
                skip_special_tokens=True,
            )
            dec_gen_inputs = tokenizer.batch_decode(
                batch["generation_input_ids"],
                skip_special_tokens=True,
            )

            # check if additional inputs are kept
            self.assertTrue(
                set(raw_batch.keys())
                - set([collator.source_key, collator.target_key]).intersection(
                    set(batch.keys())
                )
            )

            for k, v in batch.items():
                print(k)
                self.assertTrue(isinstance(v, torch.Tensor))

            for (
                raw_input,
                raw_label,
                dec_input,
                dec_gen_input,
                dec_label,
            ) in zip(
                raw_batch["src"],
                raw_batch["dst"],
                dec_inputs,
                dec_gen_inputs,
                dec_labels,
            ):
                self.assertTrue(dec_label.strip() == raw_label.strip())
                message = [
                    {"role": "user", "content": raw_input},
                    {"role": "assistant", "content": raw_label},
                ]

                self.assertTrue(
                    dec_input + tokenizer.eos_token
                    == tokenizer.apply_chat_template(
                        message, add_generation_prompt=False, tokenize=False
                    ).strip()
                )

                self.assertTrue(
                    dec_gen_input
                    == tokenizer.apply_chat_template(
                        [message[0]], add_generation_prompt=True, tokenize=False
                    ).strip()
                )
