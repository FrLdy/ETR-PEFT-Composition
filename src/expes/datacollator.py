import numpy as np
from pandas.core.computation.parsing import token
from transformers.data.data_collator import DataCollatorForLanguageModeling
from trl import DataCollatorForCompletionOnlyLM


class DataCollatorForSeq2SeqCausalLM:
    def __init__(
        self,
        tokenizer,
        eval_mode=False,
        loss_completion_only=False,
        source_key="src",
        target_key="dst",
    ):
        self.tokenizer = tokenizer
        self.eval_mode = eval_mode
        self.loss_completion_only = loss_completion_only
        self.source_key = source_key
        self.target_key = target_key

        # Precompute instruction and response token IDs
        if loss_completion_only:
            self.causal_lm_collator = DataCollatorForCompletionOnlyLM(
                tokenizer=self.tokenizer,
                response_template=tokenizer.output_prefix,
                instruction_template=tokenizer.input_prefix,
            )
        else:
            self.causal_lm_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            )

    def _encode_template(self, template):
        return self.tokenizer.encode(template, add_special_tokens=False)

    def __call__(self, examples):
        texts, texts_eval = zip(*[self._process_example(ex) for ex in examples])

        # Tokenize input text with right padding for training
        self.causal_lm_collator.tokenizer.padding_side = "right"
        batch = self.tokenizer(list(texts), padding=True)
        batch = self.causal_lm_collator(batch["input_ids"])
        if self.eval_mode:
            self.tokenizer.padding_side = "left"
            batch_eval = self.tokenizer(
                list(texts_eval), return_tensors="pt", padding=True
            )
            batch.update(
                {
                    "generation_input_ids": batch_eval["input_ids"],
                    "generation_attention_mask": batch_eval["attention_mask"],
                }
            )

        return batch

    def _process_example(self, example):
        return self._apply_chat_template(
            example[self.source_key], example[self.target_key]
        )

    def _apply_chat_template(self, source, target):
        message = [
            {"role": "user", "content": source},
            {"role": "assistant", "content": target},
        ]
        return (
            self.tokenizer.apply_chat_template(
                message, add_generation_prompt=False, tokenize=False
            ).strip(),
            self.tokenizer.apply_chat_template(
                [message[0]], add_generation_prompt=True, tokenize=False
            ).strip(),
        )
