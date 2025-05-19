from dataclasses import asdict, dataclass
from typing import Type

from datasets.utils.py_utils import Path
from ray import tune
from ray.train import RunConfig

from icl.config import EvalLLMConfig, deserialize_eval_llm_config
from icl.eval import EvalLLM


@dataclass
class TunerConfig:
    storage_path: Path
    expe_name: str
    n_cpus: int = 5
    n_gpus: int = 1
    n_samples: int = 5


def tuner(
    tuner_config: TunerConfig,
    eval_config: EvalLLMConfig,
    eval_llm_cls: Type[EvalLLM] = EvalLLM,
):
    def objective(config):
        config = deserialize_eval_llm_config(config)
        eval_llm = eval_llm_cls(config)
        metrics = eval_llm.run()
        return metrics

    experiment_path = (
        tuner_config.storage_path.resolve() / tuner_config.expe_name
    )
    experiment_path_str = experiment_path.as_posix()

    hp_space = asdict(eval_config)

    fn = tune.with_resources(
        objective,
        {
            "cpu": tuner_config.n_cpus,
            "gpu": tuner_config.n_gpus,
        },
    )

    if experiment_path.exists():
        print(
            f"[INFO] Restoring Ray Tune experiment from {experiment_path_str}"
        )
        tuner = tune.Tuner.restore(
            path=experiment_path_str,
            trainable=fn,
        )
    else:
        print(
            f"[INFO] Starting new Ray Tune experiment at {experiment_path_str}"
        )
        tuner = tune.Tuner(
            fn,
            param_space=hp_space,
            run_config=RunConfig(
                storage_path=tuner_config.storage_path.resolve().as_posix(),
                name=tuner_config.expe_name,
            ),
            tune_config=tune.TuneConfig(
                num_samples=tuner_config.n_samples,
                max_concurrent_trials=1,
            ),
        )

    return tuner.fit()
