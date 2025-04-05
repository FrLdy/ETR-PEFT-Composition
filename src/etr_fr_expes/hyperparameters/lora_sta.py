from random import sample

from adapters import LoRAConfig, MTLLoRAConfig
from ray import tune

from etr_fr_expes.hyperparameters.utils import sample_from_adapter


def lora_config_grid_search(adapter_name):
    return {
        adapter_name: LoRAConfig(
            attn_matrices=["q", "k", "v"],
            r=128,
            output_lora=True,
            alpha=sample_from_adapter(adapter_name=adapter_name, param="r"),
        )
    }

def mtllora_config_grid_search(tasks):
    def config(task, base_task=None):
        return MTLLoRAConfig(
            attn_matrices=["q", "k", "v"],
            n_up_projection=tune.grid_search([1, 2, 3]) if base_task is None else sample_from_adapter(adapter_name=base_task, param="n_up_projection"),
            r=128,
            output_lora=True,
            alpha=sample_from_adapter(adapter_name=task, param="r"),

        )

    base_task, tasks = tasks[0], tasks[1:]
    return {
        base_task: config(base_task),
        **{task: config(task, base_task=base_task) for task in tasks}
    }
