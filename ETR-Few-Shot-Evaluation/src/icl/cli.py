from functools import partial

import jsonargparse

from icl.tuner import TunerConfig, tuner


def tuner_cli():
    parser = jsonargparse.ArgumentParser()
    parser.add_dataclass_arguments(TunerConfig)
    args = parser.parse_args()
    tuner_config = args
    return partial(tuner, tuner_config=tuner_config)
