from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Dict, List, Optional, Type, Union

from adapters import AdapterCompositionBlock, AdapterConfig, init
from adapters.configuration import adapter_config
from ray.util import pdb
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

from expes.adapter_trainer import AdapterTrainer, Seq2SeqAdapterTrainer
from expes.callbacks import (
    LogParametersTrainedCallback,
    TestModelEachEpochCallback,
)
from expes.chat_template import ChatTemplate
from expes.datacollator import DataCollatorForSeq2SeqCausalLM
from expes.dataset import DS_KEY_ETR_FR, DS_KEY_ORANGESUM, DS_KEY_WIKILARGE_FR
from expes.eval_pred_manager import Seq2SeqEvalPredManager
from expes.metric import FALCMetrics, compute_metrics
from expes.trainer import Seq2SeqTrainer, Trainer
from expes.training_args import Seq2SeqTrainingArguments, TrainingArguments

HPS_KEY_GLOBAL = "global_config"
HPS_KEY_ADAPTER_CONFIGS = "adapter_configs"
HPS_KEY_TRAINING_ARGS = "trainer_config"


@dataclass
class TuningConfig:
    prepare_dataset: Callable
    model_class: Union[type, Type[PreTrainedModel]]
    is_causal_lm: bool = True
    model_config: Optional[PretrainedConfig] = None
    model_checkpoint: Optional[str] = None
    tokenizer_checkpoint: Optional[str] = None
    tasks: List[str] = field(
        default_factory=lambda: [
            DS_KEY_ETR_FR,
            DS_KEY_WIKILARGE_FR,
            DS_KEY_ORANGESUM,
        ]
    )
    adapter_configs: Dict = field(default_factory=dict)
    adapter_activation: Optional[Union[str, AdapterCompositionBlock]] = None
    generation_config: Dict = field(default_factory=dict)
    pad_token: Optional[str] = None
    chat_template: Optional[ChatTemplate] = None
    training_args: Dict = field(default_factory=dict)
    trainer_cls: Optional[Type[Trainer]] = None
    training_args_cls: Optional[Type[TrainingArguments]] = None

    def __post_init__(self):
        assert self.model_config or self.model_checkpoint
        if self.model_checkpoint is None:
            assert self.tokenizer_checkpoint
        if (
            self.tokenizer_checkpoint is None
            and self.model_checkpoint is not None
        ):
            self.tokenizer_checkpoint = self.model_checkpoint

        if self.trainer_cls is None:
            if self.adapter_configs is not None:
                self.trainer_cls = (
                    AdapterTrainer
                    if self.is_causal_lm
                    else Seq2SeqAdapterTrainer
                )
            else:
                self.trainer_cls = (
                    Trainer if self.is_causal_lm else Seq2SeqTrainer
                )

        if self.training_args_cls is None:
            if issubclass(self.trainer_cls, Trainer):
                self.training_args_cls = TrainingArguments
            elif issubclass(self.trainer_cls, Seq2SeqTrainer):
                self.training_args_cls = Seq2SeqTrainingArguments


@dataclass
class TunerFactories:

    tuning_config: TuningConfig = field()

    def get_hp_space(self, **kwargs):
        return asdict(self.tuning_config)

    def add_pad_token(self, config, tokenizer):
        return (
            config.pad_token is not None
            and tokenizer.pad_token is None
            and config.pad_token not in tokenizer.get_vocab()
        )

    def get_tokenizer(self, config):
        checkpoint = config.tokenizer_checkpoint or config.model_checkpoint
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        if self.add_pad_token(config, tokenizer):
            tokenizer.add_special_tokens({"pad_token": config.pad_token})
        if config.chat_template is not None:
            tokenizer = config.chat_template.apply_to_tokenizer(tokenizer)
        else:
            tokenizer.eval_pred_manager = Seq2SeqEvalPredManager(tokenizer)

        return tokenizer

    def get_model(self, config, tokenizer):
        if config.model_config:
            model = config.model_class(config.model_config)
        else:
            model = config.model_class.from_pretrained(config.model_checkpoint)

        if config.pad_token:
            model.resize_token_embeddings(len(tokenizer))
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        if config.adapter_configs and config.adapter_activation:
            model = self.setup_adapters(model, config)

        if config.generation_config:
            unused_params = model.generation_config.update(
                **config.generation_config
            )
            assert unused_params == {}

        return model

    def setup_adapters(self, model, config):
        init(model)
        adapter_configs = config.adapter_configs
        for adapter_name, adapter_conf in adapter_configs.items():
            adapter_conf = AdapterConfig.load(adapter_conf)
            model.add_adapter(adapter_name, adapter_conf)
        model.active_adapters = config.adapter_activation or list(
            adapter_configs.keys()
        )
        model.train_adapter(model.active_adapters)

        return model

    def get_datasets(self, config: TuningConfig, tokenizer):
        return config.prepare_dataset(config, tokenizer)

    def get_datacollators(self, config, tokenizer, model):
        if config.is_causal_lm:
            return {
                "data_collator": DataCollatorForSeq2SeqCausalLM(
                    tokenizer=tokenizer,
                    loss_completion_only=True,
                ),
                "eval_data_collator": DataCollatorForSeq2SeqCausalLM(
                    tokenizer=tokenizer,
                    loss_completion_only=True,
                    eval_mode=True,
                ),
            }
        return {
            "data_collator": DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                return_tensors="pt",
            )
        }

    def get_compute_metrics(self, tokenizer):
        metric_fn = FALCMetrics(lang="fr")

        return partial(
            compute_metrics, metrics_fn=metric_fn, tokenizer=tokenizer
        )

    def get_training_args(self, config):
        return config.training_args_cls(
            output_dir=".",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            remove_unused_columns=False,
            predict_with_generate=True,
            include_for_metrics=["inputs"],
            push_to_hub=False,
            disable_tqdm=True,
            report_to="none",
            **config.training_args,
        )

    def get_trainer(self, config: TuningConfig, **kwargs):
        test_dataset = kwargs.pop("test_dataset", None)
        trainer = config.trainer_cls(**kwargs)
        trainer.add_callback(LogParametersTrainedCallback(trainer))
        if test_dataset is not None:
            trainer.add_callback(
                TestModelEachEpochCallback(trainer, test_dataset=test_dataset)
            )
        return trainer
