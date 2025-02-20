from pathlib import Path

from datasets.naming import filename_prefix_for_name
from pandas import DataFrame
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from expes.metric import TEXT_METRIC_KEY
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
        file_name = texts_key.replace(f"_{TEXT_METRIC_KEY}", "") + ".json"
        df.to_json(output_dir / file_name)

    def get_texts_key(self, metrics_keys):
        return list(
            filter(lambda x: x.endswith(TEXT_METRIC_KEY), metrics_keys)
        )[0]
