import jsonargparse

from expes.tuner import RayTuner, RayTunerConfig


def tuner_cli(
    ray_tuner_cls=RayTuner,
    ray_tuner_config_cls=RayTunerConfig,
):
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(
        ray_tuner_cls,
        "init",
        skip=["factories", "tuner_config"],
    )
    parser.add_class_arguments(
        ray_tuner_config_cls,
        "tuner_config",
    )
    args = parser.parse_args()
    return args
