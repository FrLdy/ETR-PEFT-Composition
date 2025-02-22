from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Dict, List, Optional, Type

from adapters.configuration.adapter_config import AdapterConfig
from adapters.context import ForwardContext
from adapters.wrappers.model import init
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

from expes.adapter_trainer import AdapterTrainer, Seq2SeqAdapterTrainer
from expes.chat_template import ChatTemplate
from expes.datacollator import DataCollatorForSeq2SeqCausalLM
from expes.dataset import (
    DS_KEY_ETR_FR,
    DS_KEY_ORANGESUM,
    DS_KEY_WIKILARGE_FR,
    base_mtl_dataset,
)
from expes.eval_pred_manager import Seq2SeqEvalPredManager
from expes.metric import FALCMetrics, compute_metrics
from expes.trainer import Seq2SeqTrainer, Trainer
from expes.training_args import Seq2SeqTrainingArguments, TrainingArguments

HPS_KEY_GLOBAL = "global_config"
HPS_KEY_ADAPTER_CONFIGS = "adapter_configs"
HPS_KEY_TRAINING_ARGS = "trainer_config"


@dataclass
class TuningConfig:
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
    training_args: Dict = field(default_factory=dict)
    adapter_configs: Dict = field(default_factory=dict)
    pad_token: Optional[str] = None
    chat_template: Optional[ChatTemplate] = None
    trainer_cls: Type[Trainer] = field(default=Trainer)
    training_args_cls: Type[TrainingArguments] = field(
        default=TrainingArguments
    )

    def __post_init__(self):
        if self.tokenizer_checkpoint is None:
            self.tokenizer_checkpoint = self.model_checkpoint

        if self.adapter_configs is not None:
            self.trainer_cls = (
                AdapterTrainer if self.is_causal_lm else Seq2SeqAdapterTrainer
            )
        else:
            self.trainer_cls = Trainer if self.is_causal_lm else Seq2SeqTrainer

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

        if self.add_pad_token(tokenizer):
            tokenizer.add_special_tokens({"pad_token": config.pad_token})

        if self.chat_template:
            tokenizer = self.chat_template.apply_to_tokenizer(tokenizer)
        else:
            tokenizer.eval_pred_manager = Seq2SeqEvalPredManager(tokenizer)

        return tokenizer

    def get_model(self, config, tokenizer):
        if config.model_config:
            model = AutoModel.from_config(config.model_config)
        else:
            model = AutoModel.from_pretrained(config.model_checkpoint)

        if self.add_pad_token(config, tokenizer):
            model.resize_token_embeddings(len(tokenizer))

        if config.adapter_configs:
            model = self.setup_adapters(model, config)

        return model

    def setup_adapters(self, model, config):
        ForwardContext.context_args.add("task_ids")
        init(model)
        configs = config.adapter_configs
        adapter_name = list(configs.keys())[0]
        adapter_config = AdapterConfig.load(configs[adapter_name])
        model.add_adapter(adapter_name, adapter_config, set_active=True)
        return model

    def get_datasets(self, config, tokenizer):
        interleave = config["data"]["interleave"]
        dataset = base_mtl_dataset(self.tasks, interleave)
        return dataset

    def get_datacollators(self, tokenizer, model):
        if self.is_causal_lm:
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
        return self.config.training_args_cls(
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
