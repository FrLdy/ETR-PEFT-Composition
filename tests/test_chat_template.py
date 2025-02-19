import unittest

import lorem
from transformers.models.auto.tokenization_auto import AutoTokenizer

from expes.chat_template import causal_chat_template


class TestChatTemplate(unittest.TestCase):
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1"
    )
    tokenizer.chat_template = causal_chat_template.chat_template

    def build_message(self, input, answer=None):
        messages = [
            {
                "role": "user",
                "content": input,
            },
            *([{"role": "assistant", "content": answer}] if answer else []),
        ]
        return messages

    def test_apply_template_with_answer(self):
        input_text = lorem.sentence()
        answer = lorem.sentence()
        input = self.build_message(input=input_text, answer=answer)
        formatted = self.tokenizer.apply_chat_template(
            input,
            tokenize=False,
            input_prefix=causal_chat_template.input_prefix,
            output_prefix=causal_chat_template.output_prefix,
        ).strip()

        assert input_text in formatted
        assert answer in formatted
        expected = f"{causal_chat_template.input_prefix} {input_text} {causal_chat_template.output_prefix} {answer} {self.tokenizer.eos_token}"
        assert formatted == expected

    def test_apply_template_without_answer(self):
        input_text = lorem.sentence()
        input = self.build_message(input=input_text)
        formatted = self.tokenizer.apply_chat_template(
            input,
            tokenize=False,
            input_prefix=causal_chat_template.input_prefix,
            output_prefix=causal_chat_template.output_prefix,
        ).strip()

        assert input_text in formatted
        expected = f"{causal_chat_template.input_prefix} {input_text}"
        assert formatted == expected

    def test_apply_template_without_answer_with_generation_prompt(self):
        input_text = lorem.sentence()
        input = self.build_message(input=input_text)
        formatted = self.tokenizer.apply_chat_template(
            input,
            tokenize=False,
            add_generation_prompt=True,
            input_prefix=causal_chat_template.input_prefix,
            output_prefix=causal_chat_template.output_prefix,
        ).strip()

        assert input_text in formatted
        expected = f"{causal_chat_template.input_prefix} {input_text} {causal_chat_template.output_prefix}"
        assert formatted == expected
