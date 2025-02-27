from expes.cli import tuner_cli
from expes.tuner import (
    RayTuner,
    RessourcesConfig,
    TrainFuncFactories,
    TrainingConfig,
)

# To be completed or import an predefined
hps_config = TrainingConfig()

if __name__ == "__main__":
    args = tuner_cli()
    tuner = RayTuner(
        **args.init,
        tuner_config=RessourcesConfig(**args.tuner_config),
        factories=TrainFuncFactories(hps_config),
    )
    tuner()
