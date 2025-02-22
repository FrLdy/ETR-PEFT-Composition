import copy
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import datasets
import torch
import torch.nn as nn
from packaging import version
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import Seq2SeqTrainer  # noqa
from transformers import PreTrainedModel
from transformers import Trainer as HFTrainer
from transformers.data.data_collator import DataCollator
from transformers.debug_utils import DebugOption
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput,
    speed_metrics,
)
from transformers.utils import logging
from transformers.utils.import_utils import (
    is_datasets_available,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
)

from expes.training_args import TrainingArguments

if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse(
        "1.10"
    )

    from .trainer_pt_utils import smp_forward_only, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(
        XLA_FSDPV2_MIN_VERSION
    )
else:
    IS_XLA_FSDPV2_POST_2_2 = False

logger = logging.get_logger(__name__)


class Trainer(HFTrainer):
    # inspired from https://github.com/huggingface/transformers/pull/32346
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Optional[TrainingArguments] = None,
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
            else (
                eval_dataset if eval_dataset is not None else self.eval_dataset
            )
        )
        data_collator = self.data_collator
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

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.
                <Tip>
                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.
                </Tip>
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
            gen_kwargs:
                Additional `generate` specific kwargs.
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multipe eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=(
                        _eval_dataset if override else eval_dataset_name
                    ),
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                    **gen_kwargs,
                )
                metrics.update(dataset_metrics)
            return metrics

        # Set generation-related kwargs
        if self.args.predict_with_generate:
            if self.args.generation_config is None:
                # We assume the model can generate if predict-with-generate is True
                # Therefore, generation_config should be available
                self.gen_config = self.model.generation_config
            elif isinstance(self.args.generation_config, GenerationConfig):
                gen_config = self.args.generation_config
                self.gen_config = copy.deepcopy(
                    gen_config
                )  # copy so we don't modify args.gen_config in-place
            else:
                # That means `args.generation_config` is passed as a Dict
                self.gen_config = self.model.generation_config
                _ = self.gen_config.update(**self.args.generation_config)
            unused_kwargs = self.gen_config.update(**gen_kwargs)
            if unused_kwargs:
                logger.warning_once(
                    f"Following generation related kwargs were passed to `evaluate` but not used by `generate()`: "
                    f"{' '.join(unused_kwargs.keys())} .",
                    "Make sure there are no typos in the passed kwargs or do not pass unused kwargs.",
                )

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader)
        start_time = time.time()
        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=(
                True if self.compute_metrics is None else None
            ),
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[
                f"{metric_key_prefix}_jit_compilation_time"
            ]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[
                f"{metric_key_prefix}_model_preparation_time"
            ]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        self.log(output.metrics)
        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return output.metrics

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.
        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)
            gen_kwargs:
                Additional `generate` specific kwargs.
        <Tip>
        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.
        </Tip>
        Returns: *NamedTuple* A namedtuple with the following keys:
            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # Set generation-related kwargs
        if self.args.predict_with_generate:
            if self.args.generation_config is None:
                # We assume the model can generate if predict-with-generate is True
                # Therefore, generation_config should be available
                self.gen_config = self.model.generation_config
            elif isinstance(self.args.generation_config, GenerationConfig):
                gen_config = self.args.generation_config
                self.gen_config = copy.deepcopy(
                    gen_config
                )  # copy so we don't modify args.gen_config in-place
            else:
                # That means `args.generation_config` is passed as a Dict
                self.gen_config = self.model.generation_config
                _ = self.gen_config.update(**self.args.generation_config)
            unused_kwargs = self.gen_config.update(**gen_kwargs)
            if unused_kwargs:
                logger.warning_once(
                    f"Following generation related kwargs were passed to `evaluate` but not used by `generate()`: "
                    f"{' '.join(unused_kwargs.keys())} .",
                    "Make sure there are no typos in the passed kwargs or do not pass unused kwargs.",
                )

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()
        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            test_dataloader,
            description="Prediction",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[
                f"{metric_key_prefix}_jit_compilation_time"
            ]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[
                f"{metric_key_prefix}_model_preparation_time"
            ]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        self.control = self.callback_handler.on_predict(
            self.args, self.state, self.control, output.metrics
        )
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return PredictionOutput(
            predictions=output.predictions,
            label_ids=output.label_ids,
            metrics=output.metrics,
        )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            gen_kwargs:
                Additional `generate` specific kwargs.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )

        # Prioroty: gen_kwargs > args.gen_config > model.generation_config > default GenerationConfig()
        if self.args.predict_with_generate:
            gen_config = self.gen_config
            default_synced_gpus = (
                True if is_deepspeed_zero3_enabled() else False
            )
            synced_gpus = gen_kwargs.get("synced_gpus", default_synced_gpus)
            if len(gen_kwargs) > 0:
                unused_kwargs = gen_config.update(**gen_kwargs)
                if unused_kwargs:
                    logger.warning_once(
                        "Following generation related kwargs were passed to `prediction_step` but not "
                        f"used by `generate()`: {' '.join(unused_kwargs.keys())} .",
                        "Make sure there are no typos in the passed kwargs or do not pass unused kwargs.",
                    )

        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []
        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(
                tuple(inputs.get(name) for name in self.label_names)
            )
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        # If the `generation_input_ids` was passed in inputs, the model can generate and we need to modify
        # input keys. Otherwise, we don't know the `prompt` to generate from
        if self.args.predict_with_generate and not prediction_loss_only:
            generation_inputs = inputs.copy()
            if "generation_input_ids" in generation_inputs:
                # get inputs that are related to text and contain only generation prompt
                generation_only_inputs = {
                    k.replace("generation_", ""): v
                    for k, v in generation_inputs.items()
                    if "generation_" in k
                }

                # get common inputs that are not related to text, e.g. pixel-values
                gen_keys = generation_only_inputs.keys()
                generation_inputs_common = {
                    k: v
                    for k, v in generation_inputs.items()
                    if k.replace("generation_", "") not in gen_keys
                    and "generation" not in k
                }
                generated_tokens = model.generate(
                    **generation_inputs_common,
                    **generation_only_inputs,
                    generation_config=gen_config,
                    synced_gpus=synced_gpus,
                )
            else:
                raise ValueError(
                    "`predict_with_generate` is set to `True` but no inputs are passed for generation. ",
                    "Make sure you have `generation_input_ids` and `generation_attention_mask`.",
                )

        # clean up inputs for loss from generation related input tensors if there are any before doing `forward`
        inputs = {k: v for k, v in inputs.items() if "generation_" not in k}
        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(
                            v
                            for k, v in raw_outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]
                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(
                            v
                            for k, v in raw_outputs.items()
                            if k not in ignore_keys
                        )
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(
                            model, inputs, return_outputs=True
                        )
                    loss = loss.mean().detach()
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys
                        )
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]
        if prediction_loss_only:
            return (loss, None, None)

        if self.args.predict_with_generate and not prediction_loss_only:
            return (loss, generated_tokens, labels)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        return (loss, logits, labels)
