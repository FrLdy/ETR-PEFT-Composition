from functools import partial

CHAT_TEMPLATE_IDS = {
    "meta-llama/Llama-3.1-8B": {
        "### Input:": [14711, 5688, 25],
        "### Output:": [17010, 9442, 25],
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "### Input:": [14711, 5688, 25],
        "### Output:": [17010, 9442, 25],
    }
}


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

        return tokenizer

    def get_input_prefix_ids(self, checkpoint):
        return CHAT_TEMPLATE_IDS.get(checkpoint, {}).get(self.input_prefix, None)

    def get_output_prefix_ids(self, checkpoint):
        return CHAT_TEMPLATE_IDS.get(checkpoint, {}).get(self.output_prefix, None)



causal_chat_template = ChatTemplate()

