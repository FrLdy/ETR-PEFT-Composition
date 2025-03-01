from transformers import AutoModelForSeq2SeqLM

from etr_fr_expes.config import ETRDataConfig, ETRTrainingConfig, ETRTunerConfig
from etr_fr_expes.dataset import DS_KEY_ETR_FR
from etr_fr_expes.hyperparameters import (
    default_training_kwargs,
    lora_config_grid_search,
    training_kwargs_grid_search,
)
from expes.cli import tuner_cli
from expes.tuner import TrainFuncFactories

# To be completed or import an predefined
training_config = ETRTrainingConfig(
    train_tasks=[DS_KEY_ETR_FR],
    validation_tasks=[DS_KEY_ETR_FR],
    test_tasks=[DS_KEY_ETR_FR],
    is_causal_lm=False,
    data_config=ETRDataConfig(
        sampling_strategy="balanced",
        tokenize_dataset=True,
        input_max_length=512,
        output_max_length=256,
    ),
    adapter_configs=lora_config_grid_search("lora_etr_fr"),
    adapter_activation="lora_etr_fr",
    model_checkpoint="facebook/mbart-large-50",
    model_class=AutoModelForSeq2SeqLM,
    tokenizer_kwargs={"src_lang": "fr_XX", "tgt_lang": "fr_XX"},
    generation_config={"max_new_tokens": 240, "num_beams": 4},
    training_kwargs={
        **training_kwargs_grid_search(),
        **default_training_kwargs(),
        "num_train_epochs": 5,
    },
)
tuner_config = ETRTunerConfig()

if __name__ == "__main__":
    tuner_cls = tuner_cli()
    tuner = tuner_cls(
        tuner_config=tuner_config,
        factories=TrainFuncFactories(training_config),
    )
    tuner()
