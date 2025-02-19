from functools import partial
from tempfile import TemporaryDirectory

import adapters
import torch

from expes.adapter_trainer import AdapterTrainer, Seq2SeqAdapterTrainer
from expes.training_args import TrainingArguments

from ..utils import lorem_ipsum_dataset


class BaseTestTrainer:
    is_seq2seq = False
    configs_to_test = [
        adapters.MultiTaskConfigUnion(
            base_config=adapters.MTLLoRAConfig(),
            task_names=["a", "b", "c"],
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
        trainer.train()

    def test_trainer(self):
        for adapter_config in self.configs_to_test:
            model, tokenizer = self.get_model_tokenizer()
            if adapter_config is not None:
                adapter_name = adapter_config.__class__.__name__
                if isinstance(adapter_config, adapters.MultiTaskConfigUnion):
                    adapters.ForwardContext.context_args.add("task_ids")
                adapters.init(model)
                model.add_adapter(adapter_name, adapter_config)
                model.train_adapter(adapter_name)
            with self.subTest(
                model_class=model.__class__.__name__,
                config=adapter_config.base_config.__class__.__name__,
            ):

                with TemporaryDirectory() as tmpdirname:
                    self.trainings_run(
                        model,
                        tmpdirname,
                        tokenizer,
                        adapter_config,
                        **self.trainer_kwargs(tokenizer),
                    )
