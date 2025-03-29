from adapters.configuration.adapter_config import LoRAConfig
from ray import tune

from etr_fr_expes.hyperparameters.utils import sample_from_adapter


def lora_config_grid_search(adapter_name):
    return {
        adapter_name: LoRAConfig(
            attn_matrices=["q", "k", "v"],
            r=tune.grid_search([128]),
            output_lora=True,
            alpha=sample_from_adapter(adapter_name=adapter_name, param="r"),
        )
    }
