from dataclasses import dataclass


@dataclass
class ChatTemplate:
    input_prefix = "### Input:"
    output_prefix = "### Output:"
    chat_template = """
        {%- for message in messages %}
            {%- if message['role'] == 'user' %}
                {{- '### Input:' + message['content'] + '### Output:'}}
            {%- elif message['role'] == 'assistant' %}
                {{- ' '  + message['content'] + ' ' + eos_token }}
            {%- endif %}
        {%- endfor %}
    """


causal_chat_template = ChatTemplate()
