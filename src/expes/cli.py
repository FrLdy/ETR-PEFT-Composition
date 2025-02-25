import jsonargparse

from expes.tuner import RayTuner


def cli():
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(RayTuner, None, skip=["factories"])
    parser.add_method_arguments(RayTuner, "hp_search", "hp_search")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cli()
