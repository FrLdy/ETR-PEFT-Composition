from transformers import AutoModelForSeq2SeqLM

from etr_fr_expes.config import ETRDataConfig, ETRTrainingConfig, ETRTunerConfig
from etr_fr_expes.dataset import (
    DS_KEY_ETR_FR,
    DS_KEY_ETR_FR_POLITIC,
    DS_KEY_ORANGESUM,
)
from etr_fr_expes.hyperparameters.default import (
    default_training_kwargs,
    training_kwargs_grid_search,
)
from etr_fr_expes.hyperparameters.lora_sta import lora_config_grid_search
from etr_fr_expes.metric import (
    METRIC_KEY_ROUGEL,
)
from expes.cli import tuner_cli
from expes.tuner import TrainFuncFactories

MAIN_DS_KEY = DS_KEY_ORANGESUM
# To be completed or import an predefined
training_config = ETRTrainingConfig(
    train_tasks=[MAIN_DS_KEY],
    validation_tasks=[MAIN_DS_KEY],
    test_tasks=[MAIN_DS_KEY],
    is_causal_lm=False,
    data_config=ETRDataConfig(
        tokenize_dataset=True,
        input_max_length=400,
        output_max_length=150,
    ),
    adapter_configs=lora_config_grid_search(f"lora_{MAIN_DS_KEY}"),
    adapter_activation=f"lora_{MAIN_DS_KEY}",
    model_checkpoint="facebook/mbart-large-50",
    model_class=AutoModelForSeq2SeqLM,
    tokenizer_kwargs={"src_lang": "fr_XX", "tgt_lang": "fr_XX"},
    generation_config={"max_new_tokens": 150, "num_beams": 4},
    training_kwargs={
        **training_kwargs_grid_search(),
        **default_training_kwargs(),
        "num_train_epochs": 20,
    },
)
tuner_config = ETRTunerConfig(
    metric=f"eval_{MAIN_DS_KEY}_{METRIC_KEY_ROUGEL}",
    mode="max",
)

if __name__ == "__main__":
    tuner_cls = tuner_cli()
    tuner = tuner_cls(
        tuner_config=tuner_config,
        factories=TrainFuncFactories(training_config),
    )
    tuner()
