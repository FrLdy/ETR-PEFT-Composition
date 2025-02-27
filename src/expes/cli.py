import jsonargparse

from expes.config import RessourcesConfig
from expes.tuner import RayTuner


def tuner_cli(
    ray_tuner_cls=RayTuner,
    ressources_config_cls=RessourcesConfig,
):
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(
        ray_tuner_cls,
        None,
        skip=["factories", "tuner_config"],
    )
    args = parser.parse_args()
    return args
