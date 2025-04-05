
from adapters import MultiTask
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from etr_fr_expes.config import ETRDataConfig, ETRTrainingConfig, ETRTunerConfig
from etr_fr_expes.dataset import (
    DS_KEY_ETR_FR,
    DS_KEY_ORANGESUM,
    DS_KEY_WIKILARGE_FR,
)
from etr_fr_expes.hyperparameters.default import (
    llm_default_training_kwargs,
    training_kwargs_grid_search,
)
from etr_fr_expes.hyperparameters.lora_sta import (
    lora_config_grid_search,
    mtllora_config_grid_search,
)
from etr_fr_expes.metric import METRIC_KEY_SRB
from expes.chat_template import ChatTemplate
from expes.cli import tuner_cli
from expes.tuner import TrainFuncFactories

MAIN_DS_KEY = DS_KEY_ETR_FR 
MAIN_METRIC_KEY = METRIC_KEY_SRB

train_tasks=[MAIN_DS_KEY, DS_KEY_ORANGESUM, DS_KEY_WIKILARGE_FR]
adapter_names = [f"lora_{task}" for task in train_tasks]
training_config = ETRTrainingConfig(
    train_tasks=train_tasks,
    validation_tasks=[MAIN_DS_KEY],
    test_tasks=[MAIN_DS_KEY],
    is_causal_lm=True,
    data_config=ETRDataConfig(),
    chat_template=ChatTemplate(),
    pad_token="<pad>",
    adapter_configs=mtllora_config_grid_search(adapter_names),
    adapter_activation=MultiTask(*adapter_names),
    model_checkpoint="meta-llama/Llama-3.1-8B",
    model_class=AutoModelForCausalLM,
    generation_config={"max_new_tokens": 100, "num_beams": 4},
    training_kwargs={
        **training_kwargs_grid_search(),
        **llm_default_training_kwargs(),
        "num_train_epochs": 6,
        "bf16":True,
    },
)
tuner_config = ETRTunerConfig(
    metric=f"eval_{MAIN_DS_KEY}_{MAIN_METRIC_KEY}",
    mode="max",
    robustness_num_samples=0,
)

if __name__ == "__main__":
    tuner_cls = tuner_cli()
    tuner = tuner_cls(
        tuner_config=tuner_config,
        factories=TrainFuncFactories(training_config),
    )
    tuner()
