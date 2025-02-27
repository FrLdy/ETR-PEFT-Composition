from adapters.configuration.adapter_config import LoRAConfig

from etr_fr_expes.hyperparameters import HpsGridSearch


def lora_config(adapter_name):
    return {
        adapter_name: LoRAConfig(
            r=HpsGridSearch.lora_r,
            attn_matrices=HpsGridSearch.lora_attn_matrices,
        )
    }
