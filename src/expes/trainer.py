import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import datasets
import torch
from datasets import IterableDataset
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedModel
from transformers import Seq2SeqTrainer as BaseSeq2SeqTrainer
from transformers import Trainer as BaseTrainer
from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import unwrap_model
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    is_sagemaker_mp_enabled,
    logging,
)
from transformers.utils.import_utils import is_datasets_available

from adapters.composition import AdapterCompositionBlock, Fuse
from adapters.trainer import AdapterTrainerCallback

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


logger = logging.get_logger(__name__)


class Trainer(BaseTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        eval_data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[
            Union[Dataset, IterableDataset, "datasets.Dataset"]
        ] = None,
        eval_dataset: Optional[
            Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]
        ] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[
            Optional[torch.optim.Optimizer],
            Optional[torch.optim.lr_scheduler.LambdaLR],
        ] = (None, None),
        optimizer_cls_and_kwargs: Optional[
            Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]
        ] = None,
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            processing_class,
            model_init,
            compute_loss_func,
            compute_metrics,
            callbacks,
            optimizers,
            optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics,
        )

        self.eval_data_collator = (
            eval_data_collator
            if eval_data_collator is not None
            else self.data_collator
        )

    def get_eval_dataloader(
        self, eval_dataset: Optional[Union[str, Dataset]] = None
    ) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = (
            eval_dataset if isinstance(eval_dataset, str) else "eval"
        )
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(
                self._eval_dataloaders[dataloader_key]
            )

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset if eval_dataset is not None else self.eval_dataset
        )
        data_collator = self.eval_data_collator

        if is_datasets_available() and isinstance(
            eval_dataset, datasets.Dataset
        ):
            eval_dataset = self._remove_unused_columns(
                eval_dataset, description="evaluation"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="evaluation"
            )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = (
                self.args.dataloader_prefetch_factor
            )

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        data_collator = self.eval_data_collator

        if is_datasets_available() and isinstance(
            test_dataset, datasets.Dataset
        ):
            test_dataset = self._remove_unused_columns(
                test_dataset, description="test"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="test"
            )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(test_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(test_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = (
                self.args.dataloader_prefetch_factor
            )

        # We use the same batch_size as for eval.
        return self.accelerator.prepare(
            DataLoader(test_dataset, **dataloader_params)
        )


class Seq2SeqTrainer(BaseSeq2SeqTrainer, Trainer): ...


class AdapterTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        adapter_names: Optional[List[List[str]]] = None,
        optimizers: Tuple[
            torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR
        ] = (None, None),
        preprocess_logits_for_metrics: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = None,
    ):
        if model is not None:
            model_quantized = getattr(model, "is_quantized", False)
            model.is_quantized = False
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=(
                [AdapterTrainerCallback(self)] + callbacks
                if callbacks
                else [AdapterTrainerCallback(self)]
            ),
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        if model is not None:
            model.is_quantized = model_quantized

        if adapter_names is not None:
            self.model.set_active_adapters(adapter_names)
        # Set the defaults for loading/ saving model & adapters
        if isinstance(self.model, PreTrainedModel):
            model_frozen = getattr(self.model.base_model, "model_frozen", False)
        else:
            model_frozen = False
        if model_frozen and self.model.active_adapters:
            # Check if training AdapterFusion
            self.train_adapter_fusion = (
                isinstance(self.model.active_adapters, Fuse)
                or isinstance(
                    self.model.active_adapters, AdapterCompositionBlock
                )
                and any(
                    [
                        isinstance(child, Fuse)
                        for child in self.model.active_adapters.children
                    ]
                )
            )
        if self.model.active_adapters is None:
            raise ValueError(
                "Expected a model with an active adapter setup."
                "If you want to fully finetune the model use the Trainer class."
            )
        if (
            self.label_names is None or len(self.label_names) < 1
        ) and self.model.active_head is not None:
            all_label_names = set()
            for head in self.model._active_heads:
                all_label_names |= set(self.model.heads[head].get_label_names())
            self.label_names = list(all_label_names)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = (
            self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        )

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            if hasattr(self.model, "config") and hasattr(
                self.model.config, "adapters"
            ):
                match_str = r"adapter_fusion_layer\..*\.value"
                decay_parameters = [
                    name
                    for name in decay_parameters
                    if not re.match(match_str, name)
                ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = (
                Trainer.get_optimizer_cls_and_kwargs(self.args)
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = (
            output_dir if output_dir is not None else self.args.output_dir
        )
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict
                )
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
                )
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_all_adapters(output_dir)
            if self.train_adapter_fusion:
                self.model.save_all_adapter_fusions(output_dir)
            if hasattr(self.model, "heads"):
                self.model.save_all_heads(output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load_from_checkpoint(self, resume_from_checkpoint):
        args = self.args
        if os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
            logger.info(f"Loading model from {resume_from_checkpoint}).")

        if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
            config = PretrainedConfig.from_json_file(
                os.path.join(resume_from_checkpoint, CONFIG_NAME)
            )
            checkpoint_version = config.transformers_version
            if (
                checkpoint_version is not None
                and checkpoint_version != __version__
            ):
                logger.warn(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if args.deepspeed:
            # will be resumed in deepspeed_init
            pass
        else:
            adapter_loaded = False
            if os.path.isdir(resume_from_checkpoint):
                adapter_loaded = self._load_adapters(resume_from_checkpoint)
                self._load_adapter_fusions(resume_from_checkpoint)
                # Save all heads for a model with heads
                if hasattr(self.model, "heads"):
                    self._load_heads(resume_from_checkpoint)

            if not adapter_loaded:
                raise Exception(
                    "Can't find a valid checkpoint at {}".format(
                        resume_from_checkpoint
                    )
                )

    def _load_adapters(self, resume_from_checkpoint):
        adapter_loaded = False
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if (
                    "," not in file_name
                    and "adapter_config.json"
                    in os.listdir(
                        os.path.join(resume_from_checkpoint, file_name)
                    )
                ):
                    self.model.load_adapter(
                        os.path.join(
                            os.path.join(resume_from_checkpoint, file_name)
                        )
                    )
                    adapter_loaded = True
        return adapter_loaded

    def _load_adapter_fusions(self, resume_from_checkpoint):
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if "," in file_name:
                    self.model.load_adapter_fusion(
                        os.path.join(resume_from_checkpoint, file_name)
                    )

    def _load_heads(self, resume_from_checkpoint):
        for file_name in os.listdir(resume_from_checkpoint):
            if os.path.isdir(os.path.join(resume_from_checkpoint, file_name)):
                if "," not in file_name and "head_config.json" in os.listdir(
                    os.path.join(resume_from_checkpoint, file_name)
                ):
                    self.model.load_head(
                        os.path.join(resume_from_checkpoint, file_name)
                    )

    def _load_best_model(self):
        model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        logger.info(
            f"Loading best adapter(s) from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
        )
        # attempt to re-load all adapters from checkpoint
        for adapter in model.adapters_config.adapters:
            adapter_dir = os.path.join(
                self.state.best_model_checkpoint, adapter
            )
            if os.path.exists(adapter_dir):
                model.load_adapter(adapter_dir)
                model.adapter_to(adapter, device=self.args.device)
        if self.train_adapter_fusion:
            logger.info(
                f"Loading best adapter fusion(s) from {self.state.best_model_checkpoint} (score:"
                f" {self.state.best_metric})."
            )
            # attempt to re-load all adapter fusions from checkpoint
            for fusion in model.adapters_config.fusions:
                fusion_dir = os.path.join(
                    self.state.best_model_checkpoint, fusion
                )
                if os.path.exists(fusion_dir):
                    model.load_adapter_fusion(fusion_dir)
                    model.adapter_fusion_to(fusion, device=self.args.device)


class Seq2SeqAdapterTrainer(AdapterTrainer, Seq2SeqTrainer):
    pass
