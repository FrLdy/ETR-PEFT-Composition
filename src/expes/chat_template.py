from dataclasses import dataclass
from functools import partial

from expes.eval_pred_manager import ChatEvalPredictionManager


class ChatTemplate:
    input_prefix = "### Input:"
    output_prefix = "### Output:"
    chat_template = """
        {%- for message in messages %}
            {%- if message['role'] == 'user' %}
                {{- input_prefix + ' ' + message['content'] }}
            {%- elif message['role'] == 'assistant' %}
                {{- ' ' + output_prefix + ' ' + message['content'] + ' ' + eos_token }}
            {%- endif %}

            {%- if add_generation_prompt %}
                {{- ' ' + output_prefix }}
            {%- endif %}
        {%- endfor %}
    """

    def apply_to_tokenizer(self, tokenizer):
        tokenizer.chat_template = self.chat_template
        tokenizer.input_prefix = self.input_prefix
        tokenizer.output_prefix = self.output_prefix

        tokenizer.apply_chat_template = partial(
            tokenizer.apply_chat_template,
            input_prefix=tokenizer.input_prefix,
            output_prefix=tokenizer.output_prefix,
        )

        tokenizer.eval_pred_manager = ChatEvalPredictionManager(self, tokenizer)

        return tokenizer


causal_chat_template = ChatTemplate()
