from ray import tune


def sample_from_adapter(adapter_name, param):

    return tune.sample_from(
        lambda spec: spec["train_loop_config"]["adapter_configs"][adapter_name][
            param
        ]
    )
