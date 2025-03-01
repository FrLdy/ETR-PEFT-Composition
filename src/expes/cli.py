from functools import partial

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
    return partial(
        ray_tuner_cls,
        storage_path=args.storage_path,
        expe_name=args.expe_name,
        ressources_config=RessourcesConfig(**args.ressources_config),
    )
