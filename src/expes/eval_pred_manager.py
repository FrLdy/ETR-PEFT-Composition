from copy import deepcopy
from functools import partial

import numpy as np


def replace_ignore_index(tokens, substitution_token, ignore_index=-100):
    tokens = np.where(tokens != ignore_index, tokens, substitution_token)
    return tokens


class EvalPredManager: ...


class Seq2SeqEvalPredManager(EvalPredManager):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def replace_special_tokens(
        self, eval_pred, substitution_token, ignore_index=-100
    ):
        replace_fn = partial(
            replace_ignore_index,
            substitution_token=substitution_token,
            ignore_index=ignore_index,
        )
        new_eval_pred = deepcopy(eval_pred)
        new_eval_pred.predictions = replace_fn(eval_pred.predictions)
        new_eval_pred.label_ids = replace_fn(eval_pred.label_ids)
        if eval_pred.inputs is not None:
            new_eval_pred.inputs = replace_fn(eval_pred.inputs)

        return new_eval_pred

    def __call__(self, eval_pred):
        decode = partial(self.tokenizer.batch_decode, skip_special_tokens=True)
        eval_pred = self.replace_special_tokens(
            eval_pred, self.tokenizer.pad_token_id, -100
        )

        eval_pred.label_ids = decode(eval_pred.label_ids)
        eval_pred.predictions = decode(eval_pred.predictions)
        if eval_pred.inputs is not None:
            eval_pred.inputs = decode(eval_pred.inputs)

        return eval_pred


class ChatEvalPredictionManager(Seq2SeqEvalPredManager):
    def __init__(self, chat_template, tokenizer):
        super().__init__(tokenizer)
        self.chat_template = chat_template

    @property
    def output_prefix(self):
        return self.chat_template.output_prefix

    @property
    def input_prefix(self):
        return self.chat_template.input_prefix

    def get_pred(self, text):
        return text.split(self.output_prefix, 1)[-1].strip()

    def get_label(self, text):
        return self.get_pred(text)

    def get_input(self, text):
        return (
            text.split(self.output_prefix, 1)[0]
            .replace(self.input_prefix, "")
            .strip()
        )

    def get_inputs(self, texts):
        return [self.get_input(text) for text in texts]

    def get_predictions(self, texts):
        return [self.get_pred(text) for text in texts]

    def get_labels(self, texts):
        return self.get_predictions(texts)

    def __call__(self, eval_pred):
        eval_pred = super().__call__(eval_pred)

        eval_pred.predictions = self.get_predictions(eval_pred.predictions)
        eval_pred.label_ids = self.get_labels(eval_pred.label_ids)
        eval_pred.inputs = (
            self.get_inputs(eval_pred.inputs)
            if eval_pred.inputs is not None
            else eval_pred.inputs
        )

        return eval_pred
