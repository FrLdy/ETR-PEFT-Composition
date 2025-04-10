from adapters import MultiTask
from transformers import AutoModelForSeq2SeqLM

from etr_fr_expes.config import (
    ETRDataConfig,
    ETRTrainingConfig,
    ETRTunerConfig,
    get_inference_config,
)
from etr_fr_expes.dataset import (
    DS_KEY_ETR_FR,
    DS_KEY_ORANGESUM,
    DS_KEY_WIKILARGE_FR,
)
from etr_fr_expes.hyperparameters.default import (
    default_training_kwargs,
    training_kwargs_grid_search,
)
from etr_fr_expes.hyperparameters.lora_sta import mtllora_config_grid_search
from etr_fr_expes.metric import METRIC_KEY_SRB
from expes.cli import tuner_cli
from expes.tuner import TrainFuncFactories

MAIN_DS_KEY = DS_KEY_ETR_FR
# To be completed or import an predefined
train_tasks=[MAIN_DS_KEY, DS_KEY_WIKILARGE_FR]
adapter_names = [f"lora_{task}" for task in train_tasks]
training_config = ETRTrainingConfig(
    train_tasks=train_tasks,
    validation_tasks=[MAIN_DS_KEY],
    test_tasks=[MAIN_DS_KEY],
    is_causal_lm=False,
    data_config=ETRDataConfig(
        sampling_strategy="balanced",
        tokenize_dataset=True,
        input_max_length=512,
        output_max_length=256,
    ),
    adapter_configs=mtllora_config_grid_search(adapter_names),
    adapter_activation=MultiTask(*adapter_names),
    model_checkpoint="moussaKam/mbarthez",
    model_class=AutoModelForSeq2SeqLM,
    tokenizer_kwargs={"src_lang": "fr_XX", "tgt_lang": "fr_XX"},
    generation_config={"max_new_tokens": 256, "num_beams": 4},
    training_kwargs={
        **training_kwargs_grid_search(),
        **default_training_kwargs(),
        "num_train_epochs": 25,
        "ddp_find_unused_parameters": False,
    },
)
tuner_config = ETRTunerConfig(
    metric=f"eval_{MAIN_DS_KEY}_{METRIC_KEY_SRB}",
    mode="max",
    robustness_num_samples=0,
)

inference_config = get_inference_config(training_config.train_tasks)

if __name__ == "__main__":
    tuner_cls = tuner_cli()
    tuner = tuner_cls(
        tuner_config=tuner_config,
        factories=TrainFuncFactories(training_config, inference_config),
    )
    tuner()
