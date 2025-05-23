from transformers import AutoModelForSeq2SeqLM

from etr_fr_expes.config import ETRDataConfig, ETRTrainingConfig, ETRTunerConfig
from etr_fr_expes.dataset import (
    DS_KEY_WIKILARGE_FR,
)
from etr_fr_expes.hyperparameters.default import (
    default_training_kwargs,
    training_kwargs_grid_search,
)
from etr_fr_expes.hyperparameters.lora_sta import lora_config_grid_search
from etr_fr_expes.metric import METRIC_KEY_BLEU, METRIC_KEY_SARI, METRIC_KEY_SRB
from expes.cli import tuner_cli
from expes.tuner import TrainFuncFactories

MAIN_DS_KEY = DS_KEY_WIKILARGE_FR

training_config = ETRTrainingConfig(
    train_tasks=[MAIN_DS_KEY],
    validation_tasks=[MAIN_DS_KEY],
    test_tasks=[MAIN_DS_KEY],
    is_causal_lm=False,
    data_config=ETRDataConfig(
        tokenize_dataset=True,
        input_max_length=200,
        output_max_length=100,
    ),
    adapter_configs=lora_config_grid_search("lora_wikilarge_fr"),
    adapter_activation="lora_wikilarge_fr",
    model_checkpoint="moussaKam/mbarthez",
    model_class=AutoModelForSeq2SeqLM,
    tokenizer_kwargs={"src_lang": "fr_XX", "tgt_lang": "fr_XX"},
    generation_config={"max_new_tokens": 100, "num_beams": 4},
    training_kwargs={
        **training_kwargs_grid_search(),
        **default_training_kwargs(),
        "num_train_epochs": 8,
    },
)
tuner_config = ETRTunerConfig(
    metric=f"eval_{MAIN_DS_KEY}_{METRIC_KEY_SARI}",
    mode="max",
)

if __name__ == "__main__":
    tuner_cls = tuner_cli()
    tuner = tuner_cls(
        tuner_config=tuner_config,
        factories=TrainFuncFactories(training_config),
    )
    tuner()
