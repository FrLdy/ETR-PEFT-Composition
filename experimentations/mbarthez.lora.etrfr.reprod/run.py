from adapters.configuration.adapter_config import LoRAConfig
from ray import tune
from transformers import AutoModelForSeq2SeqLM

from etr_fr_expes.config import ETRDataConfig, ETRTrainingConfig, ETRTunerConfig
from etr_fr_expes.dataset import DS_KEY_ETR_FR, DS_KEY_ETR_FR_POLITIC
from etr_fr_expes.hyperparameters.default import (
    default_training_kwargs,
)
from etr_fr_expes.hyperparameters.utils import sample_from_adapter
from etr_fr_expes.metric import METRIC_KEY_SRB
from expes.cli import tuner_cli
from expes.tuner import TrainFuncFactories


def lora_config_grid_search(adapter_name):
    return {
        adapter_name: LoRAConfig(
            attn_matrices=["q", "k", "v"],
            r=128,
            output_lora=True,
            dropout=0.05,
            alpha=sample_from_adapter(adapter_name=adapter_name, param="r"),
        )
    }

MAIN_DS_KEY = DS_KEY_ETR_FR
# To be completed or import an predefined
training_config = ETRTrainingConfig(
    train_tasks=[MAIN_DS_KEY],
    validation_tasks=[MAIN_DS_KEY],
    test_tasks=[MAIN_DS_KEY],
    is_causal_lm=False,
    data_config=ETRDataConfig(
        sampling_strategy="balanced",
        tokenize_dataset=True,
        input_max_length=512,
        output_max_length=256,
    ),
    adapter_configs=lora_config_grid_search(f"lora_{MAIN_DS_KEY}"),
    adapter_activation=f"lora_{MAIN_DS_KEY}",
    model_checkpoint="facebook/mbart-large-50",
    model_class=AutoModelForSeq2SeqLM,
    tokenizer_kwargs={"src_lang": "fr_XX", "tgt_lang": "fr_XX"},
    generation_config={"max_new_tokens": 256, "num_beams": 4},
    training_kwargs={
        "learning_rate":tune.grid_search([5e-5]),
        **default_training_kwargs(),
        "num_train_epochs": 30,
    }
)
tuner_config = ETRTunerConfig(
    metric=f"eval_{MAIN_DS_KEY}_{METRIC_KEY_SRB}",
    mode="max",
)

if __name__ == "__main__":
    tuner_cls = tuner_cli()
    tuner = tuner_cls(
        tuner_config=tuner_config,
        factories=TrainFuncFactories(training_config),
    )
    tuner()
