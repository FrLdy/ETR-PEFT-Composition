import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Union

import ray
import transformers
from datasets import Dataset
from pandas import DataFrame
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train import Checkpoint
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from expes.metric import METRIC_KEY_TEXTS
from expes.trainer import Trainer
from expes.training_args import TrainingArguments


class LogParametersTrainedCallback(TrainerCallback):

    def __init__(self, trainer: Trainer) -> None:
        self.trainer = trainer

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):

        model = kwargs["model"]
        summary = model.adapter_summary(as_dict=True)
        prefix = "params_summary"
        metrics = {
            f"{prefix}_{infos['name'].lower().replace(' ', '_')}_{info}": v
            for infos in summary
            for info, v in infos.items()
            if info != "name"
        }

        self.trainer.log(metrics)


class SavePredictionsCallback(TrainerCallback):
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        metrics = kwargs["metrics"]
        texts_key = self.get_texts_key(metrics.keys())
        texts = metrics[texts_key]
        df = DataFrame(texts)

        output_dir = Path(args.output_dir)
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        output_dir = output_dir / checkpoint_folder / "generations"
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = texts_key.replace(f"_{METRIC_KEY_TEXTS}", "") + ".json"
        df.to_json(output_dir / file_name)

    def get_texts_key(self, metrics_keys):
        return list(
            filter(lambda x: x.endswith(METRIC_KEY_TEXTS), metrics_keys)
        )[0]


class TestModelEachEpochCallback(TrainerCallback):
    def __init__(
        self, trainer: Trainer, test_dataset: Union[Dataset, Dict[str, Dataset]]
    ) -> None:
        self.trainer = trainer
        self.test_dataset = test_dataset

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.trainer.evaluate(
            eval_dataset=self.test_dataset, metric_key_prefix="test"
        )
        control.should_evaluate = True
        control.should_log = True


class RayTrainReportCallback(TrainerCallback):
    """A simple callback to report checkpoints and metrics to Ray Train.

    This callback is a subclass of `transformers.TrainerCallback
    <https://huggingface.co/docs/transformers/main/en/main_classes/callback#transformers.TrainerCallback>`_
    and overrides the `TrainerCallback.on_save()` method. After
    a new checkpoint get saved, it fetches the latest metric dictionary
    from `TrainerState.log_history` and reports it with the latest checkpoint
    to Ray Train.

    Checkpoints will be saved in the following structure::

        checkpoint_00000*/   Ray Train Checkpoint
        └─ checkpoint/       Hugging Face Transformers Checkpoint

    For customized reporting and checkpointing logic, implement your own
    `transformers.TrainerCallback` following this user
    guide: :ref:`Saving and Loading Checkpoints <train-dl-saving-checkpoints>`.

    Note that users should ensure that the logging, evaluation, and saving frequencies
    are properly configured so that the monitoring metric is always up-to-date
    when `transformers.Trainer` saves a checkpoint.

    Suppose the monitoring metric is reported from evaluation stage:

    Some valid configurations:
        - evaluation_strategy == save_strategy == "epoch"
        - evaluation_strategy == save_strategy == "steps", save_steps % eval_steps == 0

    Some invalid configurations:
        - evaluation_strategy != save_strategy
        - evaluation_strategy == save_strategy == "steps", save_steps % eval_steps != 0

    """

    CHECKPOINT_NAME = "checkpoint"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        record_extra_usage_tag(
            TagKey.TRAIN_TRANSFORMERS_RAYTRAINREPORTCALLBACK, "1"
        )

    def on_save(self, args, state, control, **kwargs):
        """Event called after a checkpoint save."""
        with TemporaryDirectory() as tmpdir:
            # Aggregate all the logged metrics
            metrics = {}
            for log in state.log_history:
                metrics.update(log)

            # Copy ckpt files and construct a Ray Train Checkpoint
            source_ckpt_path = transformers.trainer.get_last_checkpoint(
                args.output_dir
            )
            if source_ckpt_path is not None:
                target_ckpt_path = Path(tmpdir, self.CHECKPOINT_NAME).as_posix()
                shutil.copytree(source_ckpt_path, target_ckpt_path)
                checkpoint = Checkpoint.from_directory(tmpdir)
            else:
                checkpoint = None

            # Report latest metrics and checkpoint to Ray Train
            ray.train.report(metrics=metrics, checkpoint=checkpoint)
