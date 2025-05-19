from typing import Literal

import torch
from transformers import AutoTokenizer, pipeline


class LLM:
    def __init__(self, config: "EvalLLMConfig") -> None:
        self.config = config
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()

    def create_prompt(self, input):
        prompt_template = self.config.icl_config.prompt_template
        return prompt_template.format(input=input)

    def load_model(self): ...

    def load_tokenizer(self): ...

    def generate(self, prompt, input): ...


class LocalLLM(LLM):
    def load_model(self):
        model = pipeline(
            "text-generation",
            model=self.config.model_name,
            device_map="auto",
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "use_flash_attention_2": True,
            },
        )
        return model

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        return tokenizer

    def generate(self, prompt, input):
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {"role": "user", "content": input},
        ]

        res = self.model(
            messages,
            max_new_tokens=3000,
        )
        res = res[0]["generated_text"][-1]["content"]
        return res


LLM_CLASSES = {
    c.__name__: c
    for c in [
        LocalLLM,
    ]
}
LLMClassName = Literal[tuple(LLM_CLASSES.keys())]
