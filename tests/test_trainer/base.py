from copy import deepcopy
from functools import partial
from tempfile import TemporaryDirectory

import adapters
import torch
from adapters.heads.language_modeling import CausalLMHead
from transformers.trainer_utils import EvalPrediction

from expes.adapter_trainer import AdapterTrainer, Seq2SeqAdapterTrainer
from expes.callbacks import (
    LogParametersTrainedCallback,
    SavePredictionsCallback,
)
from expes.metric import TEXT_METRIC_KEY
from expes.training_args import TrainingArguments

from ..utils import lorem_ipsum_dataset


class BaseTestTrainer:
    is_seq2seq = False
    configs_to_test = [
        (
            adapters.MultiTaskConfigUnion(
                base_config=adapters.MTLLoRAConfig(),
                task_names=["a", "b", "c"],
            ),
            ["loras.a.", "loras.b.", "loras.c."],
        ),
        (
            adapters.LoRAConfig(),
            ["loras.{name}."],
        ),
        (
            adapters.MTLLoRAConfig(),
            ["loras.{name}."],
        ),
    ]

    def get_dataset(self, adapter_config=None, task_id=None):
        dataset = lorem_ipsum_dataset(25)
        dataset_size = len(dataset)
        if hasattr(adapter_config, "task_names"):
            dataset = dataset.add_column(
                "task_ids",
                torch.randint(
                    0, len(adapter_config.task_names), (dataset_size,)
                ).tolist(),
            )
        if task_id is not None:
            dataset = dataset.add_column(
                "task_ids", torch.tensor([task_id] * dataset_size).tolist()
            )
        return dataset

    def trainings_run(
        self,
        model,
        output_dir,
        tokenizer,
        adapter_config,
        lr=1.0,
        num_train_epochs=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        **kwargs,
    ):
        # setup dataset
        train_dataset = self.get_dataset(adapter_config)
        if hasattr(adapter_config, "task_names"):
            eval_dataset = {
                t: self.get_dataset(task_id=i)
                for i, t in enumerate(adapter_config.task_names)
            }
        else:
            eval_dataset = self.get_dataset(adapter_config)

        training_args = TrainingArguments(
            output_dir=output_dir,
            do_train=True,
            eval_strategy="epoch",
            learning_rate=lr,
            num_train_epochs=num_train_epochs,
            use_cpu=True,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            remove_unused_columns=False,
            predict_with_generate=True,
            include_for_metrics=["inputs"],
        )

        # evaluate
        trainer = (
            AdapterTrainer if not self.is_seq2seq else Seq2SeqAdapterTrainer
        )(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=partial(
                self.run_test_compute_metrics,
                tokenizer=tokenizer,
            ),
            **kwargs,
        )
        trainer.add_callback(SavePredictionsCallback)
        trainer.add_callback(LogParametersTrainedCallback(trainer))
        trainer.train()

    def run_train_test(
        self,
        model,
        state_dict_pre,
        adapter_name,
        filter_keys,
    ):
        def has_tied_embeddings(k):
            tied_embeddings = (
                hasattr(model.config, "tie_word_embeddings")
                and model.config.tie_word_embeddings
            )
            if hasattr(model, "heads"):
                is_tied_layer = (
                    isinstance(model.heads[adapter_name], CausalLMHead)
                    and "heads.{}.{}.weight".format(
                        adapter_name,
                        len(model.heads[adapter_name]._modules) - 1,
                    )
                    in k
                )
            else:
                is_tied_layer = True
            return tied_embeddings and is_tied_layer

        adapters_with_change, base_with_change = False, False
        for (k1, v1), (k2, v2) in zip(
            state_dict_pre.items(), model.state_dict().items()
        ):
            # move both to the same device to avoid device mismatch errors
            v1, v2 = v1.to(v2.device), v2
            if (
                any(key.format(name=adapter_name) in k1 for key in filter_keys)
                or adapter_name in k1
            ) and not has_tied_embeddings(k1):
                adapters_with_change |= not torch.equal(v1, v2)
            else:
                base_with_change |= not torch.equal(v1, v2)
        self.assertTrue(adapters_with_change)
        self.assertFalse(base_with_change)

    def test_trainer(self):
        for adapter_config, filter_keys in self.configs_to_test:
            model, tokenizer = self.get_model_tokenizer()
            with self.subTest(
                model_class=model.__class__.__name__,
                config=adapter_config.__class__.__name__,
            ):
                with TemporaryDirectory() as tmpdirname:
                    if adapter_config is not None:
                        adapter_name = adapter_config.__class__.__name__
                        if isinstance(
                            adapter_config, adapters.MultiTaskConfigUnion
                        ):
                            adapters.ForwardContext.context_args.add("task_ids")
                        adapters.init(model)
                        model.add_adapter(adapter_name, adapter_config)
                        model.train_adapter(adapter_name)

                    state_dict_pre = deepcopy(model.state_dict())

                    self.trainings_run(
                        model,
                        tmpdirname,
                        tokenizer,
                        adapter_config,
                        **self.trainer_kwargs(tokenizer),
                    )
                    self.run_train_test(
                        model,
                        state_dict_pre,
                        adapter_name,
                        filter_keys,
                    )

    def run_test_compute_metrics(self, eval_pred, tokenizer):
        if hasattr(tokenizer, "eval_pred_manager"):
            eval_pred: EvalPrediction = tokenizer.eval_pred_manager(eval_pred)

        return {
            "dummy_score": 1.0,
            TEXT_METRIC_KEY: {
                "inputs": eval_pred.inputs,
                "labels": eval_pred.label_ids,
                "predictions": eval_pred.predictions,
            },
        }
