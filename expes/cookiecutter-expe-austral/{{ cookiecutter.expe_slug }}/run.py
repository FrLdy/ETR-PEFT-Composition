from expes.cli import tuner_cli
from expes.tuner import RayTuner, RayTunerConfig
from expes.tuner_factory import TrainFuncFactories, TrainingConfig

# To be completed or import an predefined
hps_config = TrainingConfig()

if __name__ == "__main__":
    args = tuner_cli()
    tuner = RayTuner(
        **args.init,
        tuner_config=RayTunerConfig(**args.tuner_config),
        factories=TrainFuncFactories(hps_config),
    )
    tuner()
