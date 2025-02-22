from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional

import adapters
from adapters.context import ForwardContext
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils.dummy_pt_objects import Seq2SeqTrainer

from expes.chat_template import ChatTemplate
from expes.datacollator import DataCollatorForSeq2SeqCausalLM
from expes.dataset import (
    DS_KEY_ETR_FR,
    DS_KEY_ORANGESUM,
    DS_KEY_WIKILARGE_FR,
    base_mtl_dataset,
)
from expes.eval_pred_manager import Seq2SeqEvalPredManager
from expes.metric import TEXT_METRIC_KEY, FALCMetrics, compute_metrics
from expes.trainer import Trainer
from expes.training_args import Seq2SeqTrainingArguments, TrainingArguments

HPS_KEY_GLOBAL = "global_config"
HPS_KEY_ADAPTER_CONFIGS = "adapter_configs"
HPS_KEY_TRAINER_CONFIG = "trainer_config"


@dataclass
class TunerFactories:

    tasks: List[str] = [DS_KEY_ETR_FR, DS_KEY_WIKILARGE_FR, DS_KEY_ORANGESUM]

    is_causal_lm = True
    model_config = None
    model_checkpoint = None

    tokenizer_checkpoint = None
    add_pad_token = True
    pad_token: str = "<pad>"
    chat_template = ChatTemplate()

    trainer_cls = None
    trainer_config: Dict = field(default_factory=dict)
    adpater_configs: Dict = field(default_factory=dict)
    setup_adapters: Optional[Callable] = None

    @property
    def training_args_cls(self):
        if isinstance(self.trainer_cls, Trainer):
            return TrainingArguments
        elif isinstance(self.trainer_cls, Seq2SeqTrainer):
            return Seq2SeqTrainingArguments

    @property
    def global_hps(self):
        return {
            "model_checkpoint": self.model_checkpoint,
            "tokenizer_checkpoint": self.tokenizer_checkpoint
            or self.model_checkpoint,
            "tasks": self.tasks,
        }

    def get_hp_space(self, **kwargs):
        hp_space = {}
        hp_space.update({HPS_KEY_GLOBAL: self.global_hps})
        if self.adapter_configs:
            hp_space.update({HPS_KEY_ADAPTER_CONFIGS: self.adapter_configs})
        if self.trainer_config:
            hp_space.update({HPS_KEY_TRAINER_CONFIG: self.trainer_config})

        return hp_space

    def get_tokenizer(self, config):
        checkpoint = self.tokenizer_checkpoint or self.model_checkpoint
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if (
            self.add_pad_token
            and tokenizer.pad_token is None
            and self.pad_token not in tokenizer.get_vocab()
        ):
            tokenizer.add_special_tokens({"pad_token": self.pad_token})
        if self.chat_template:
            tokenizer = self.chat_template.apply_to_tokenizer(tokenizer)
        else:
            tokenizer.eval_pred_manager = Seq2SeqEvalPredManager(tokenizer)
        return tokenizer

    def get_model(self, config, tokenizer):
        if self.model_config:
            model = AutoModel.from_config(self.model_config)
        else:
            model = AutoModel.from_pretrained(self.model_checkpoint)
        if self.add_pad_token:
            model.resize_token_embeddings(len(tokenizer))
        if self.adapter_configs:
            self.setup_adapters(model, config.get(HPS_KEY_ADAPTER_CONFIGS))
        return model

    def get_datasets(self, config, tokenizer):
        interleave = config["data"]["interleave"]
        dataset = base_mtl_dataset(self.tasks, interleave)
        return dataset

    def get_datacollators(self, tokenizer, config, **kwargs):
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
        # TODO : add for seq2seq
        return {}

    def get_compute_metrics(self, tokenizer):
        metric_fn = FALCMetrics(lang="fr")

        return partial(
            compute_metrics, metrics_fn=metric_fn, tokenizer=tokenizer
        )

    def get_training_args(self, config):
        return self.training_args_cls(
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
            **config.get(HPS_KEY_TRAINER_CONFIG, {}),
        )

    def get_trainer(self, **kwargs):
        trainer = self.trainer_cls(**kwargs)
        return trainer
