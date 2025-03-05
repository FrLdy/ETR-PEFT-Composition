from adapters.configuration.adapter_config import LoRAConfig
from ray import tune


def sample_from_adapter(adapter_name, adapter_param):

    return tune.sample_from(
        lambda spec: spec["train_loop_config"]["adapter_configs"][adapter_name][
            adapter_param
        ]
    )


def default_training_kwargs():
    return dict(
        weight_decay=1e-1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
    )


def training_kwargs_grid_search():
    return dict(
        learning_rate=tune.grid_search([2e-5, 5e-5, 1e-4]),
    )


def lora_config_grid_search(adapter_name):
    return {
        adapter_name: LoRAConfig(
            attn_matrices=tune.grid_search(
                [
                    ["q", "k"],
                    ["q", "v"],
                    ["k", "v"],
                    ["q", "k", "v"],
                ]
            ),
            r=tune.grid_search([16, 32, 64, 128]),
            output_lora=tune.grid_search([True, False]),
            alpha=sample_from_adapter(
                adapter_name=adapter_name, adapter_param="r"
            ),
        )
    }
