from dataclasses import dataclass
from functools import partial

from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils.dummy_pt_objects import Seq2SeqTrainer

from expes.chat_template import ChatTemplate
from expes.datacollator import DataCollatorForSeq2SeqCausalLM
from expes.dataset import build_mtl_dataset
from expes.eval_pred_manager import Seq2SeqEvalPredManager
from expes.metric import TEXT_METRIC_KEY, FALCMetrics
from expes.trainer import Trainer
from expes.training_args import Seq2SeqTrainingArguments, TrainingArguments


def compute_metrics(eval_pred, metrics_fn, tokenizer):
    eval_pred = tokenizer.eval_pred_manager(eval_pred)
    inputs, labels, predictions = (
        eval_pred.inputs,
        eval_pred.label_ids,
        eval_pred.predictions,
    )
    metrics = metrics_fn(
        predictions=predictions, references=labels, sources=inputs
    )

    metrics.update(
        {
            TEXT_METRIC_KEY: {
                "inputs": inputs,
                "labels": labels,
                "predictions": predictions,
            },
        }
    )

    return metrics


def get_tokenizer(checkpoint, add_pad_token=False, chat_template=None):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if add_pad_token:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if chat_template:
        tokenizer = chat_template.apply_to_tokenizer(tokenizer)
    else:
        tokenizer.eval_pred_manager = Seq2SeqEvalPredManager(tokenizer)
    return tokenizer


def get_model(checkpoint, tokenizer, resize_token_embeddings=False):
    model = AutoModel.from_pretrained(checkpoint)
    if resize_token_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    return model


@dataclass
class TunerFactories:

    is_causal_lm = False
    trainer_cls = None
    model_config = None
    model_checkpoint = None
    tokenizer_checkpoint = None
    add_pad_token = True
    chat_template = ChatTemplate()

    @property
    def training_args_cls(self):
        if isinstance(self.trainer_cls, Trainer):
            return TrainingArguments
        elif isinstance(self.trainer_cls, Seq2SeqTrainer):
            return Seq2SeqTrainingArguments

    def get_hp_space(self, **kwargs):
        return {
            "global_config": {
                "model_checkpoint": self.model_checkpoint,
                "tokenizer_checkpoint": self.tokenizer_checkpoint
                or self.model_checkpoint,
                "add_pad_token": self.add_pad_token,
            }
        }

    def get_tokenizer(self, config):
        checkpoint = (self.tokenizer_checkpoint or self.model_checkpoint,)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if self.add_pad_token:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
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
        return model

    def get_train_dataset(self, config, tokenizer):
        raise NotImplementedError

    def get_eval_dataset(self, config, tokenizer):
        raise NotImplementedError

    def get_test_dataset(self, config, tokenizer):
        raise NotImplementedError

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
            **config.get("training_args", {}),
        )

    def get_trainer(self, **kwargs):
        trainer = self.trainer_cls(**kwargs)
        return trainer


class MTLTunerFactories(TunerFactories):
    def get_train_dataset(self, config, tokenizer):
        build_mtl_dataset({"etr": ..., "orangesum": ...})

    def get_eval_dataset(self, config, tokenizer):
        return super().get_eval_dataset(config, tokenizer)

    def get_test_dataset(self, config, tokenizer):
        return super().get_test_dataset(config, tokenizer)
