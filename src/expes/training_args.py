from dataclasses import field
from typing import Optional, Union

from transformers.generation.configuration_utils import GenerationConfig
from transformers.training_args import (
    TrainingArguments as BaseTrainingArguments,
)
from transformers.training_args_seq2seq import (
    Seq2SeqTrainingArguments as BaseSeq2SeqTrainingArguments,
)


class TrainingArguments(BaseTrainingArguments):
    predict_with_generate: bool = field(
        default=False,
        metadata={
            "help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."
        },
    )

    generation_config: Optional[Union[dict, str, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": (
                "The GenerationConfig that will be used during prediction. Args from this config ",
                "will have higher priority than model's generation config. Anything not set by this config ",
                "will fallback to `model.generation_config`.",
            )
        },
    )


class Seq2SeqTrainingArguments(BaseSeq2SeqTrainingArguments): ...
