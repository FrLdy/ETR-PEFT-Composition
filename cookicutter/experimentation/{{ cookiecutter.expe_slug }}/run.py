from expes.cli import tuner_cli
from expes.config import TunerConfig
from expes.tuner import TrainFuncFactories, TrainingConfig

# To be completed or import an predefined
training_config = TrainingConfig()
tuner_config = TunerConfig()

if __name__ == "__main__":
    tuner_cls = tuner_cli()
    tuner = tuner_cls(
        tuner_config=tuner_config,
        factories=TrainFuncFactories(training_config),
    )
    tuner()
