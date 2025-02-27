from functools import partial

from ray import tune


def sample_from_adapter(adapter_name, adapter_param):

    return tune.sample_from(
        lambda spec: spec["train_loop_config"]["adapter_configs"][adapter_name][
            adapter_param
        ]
    )


from adapters.configuration.adapter_config import LoRAConfig

from etr_fr_expes.hyperparameters import HpsGridSearch


def lora_config(adapter_name):
    return {
        adapter_name: LoRAConfig(
            r=HpsGridSearch.lora_r,
            attn_matrices=HpsGridSearch.lora_attn_matrices,
            alpha=HPS,
        )
    }


class LoRAGridSearch:
    dropout = tune.grid_search([0.0, 0.05, 0.1])
    learning_rate = tune.grid_search([1e-5, 2e-5, 5e-5, 1e-4])
    attn_matrices = tune.grid_search(
        [
            ["q"],
            ["k"],
            ["v"],
            ["q", "k"],
            ["q", "v"],
            ["k", "v"],
            ["q", "k", "v"],
        ]
    )
    r = tune.grid_search([8, 16, 32, 64, 128])
    output_lora = tune.grid_search([True, False])
    alpha = partial(sample_from_adapter, adapter_param="r")
