from adapters.configuration.adapter_config import LoRAConfig
from ray import tune
from transformers import AutoModelForSeq2SeqLM

from etr_fr_expes.config import ETRDataConfig, ETRTrainingConfig, ETRTunerConfig
from etr_fr_expes.dataset import DS_KEY_ETR_FR, DS_KEY_ETR_FR_POLITIC
from etr_fr_expes.hyperparameters.default import (
    default_training_kwargs,
    training_kwargs_grid_search,
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
            dropout=0.1,
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
    model_checkpoint="moussaKam/mbarthez",
    model_class=AutoModelForSeq2SeqLM,
    generation_config={"max_new_tokens": 256, "num_beams": 4},
    training_kwargs={
        **default_training_kwargs(),
        "learning_rate": 5e-5,
        "num_train_epochs": 25,
    },
)
tuner_config = ETRTunerConfig(
    metric=f"eval_{MAIN_DS_KEY}_{METRIC_KEY_SRB}",
    mode="max",
    robustness_num_samples=0
)

if __name__ == "__main__":
    tuner_cls = tuner_cli()
    tuner = tuner_cls(
        tuner_config=tuner_config,
        factories=TrainFuncFactories(training_config),
    )
    tuner()
