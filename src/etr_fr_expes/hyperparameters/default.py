from ray import tune


def default_training_kwargs():
    return dict(
        weight_decay=1e-1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
    )

def llm_default_training_kwargs():
    return {
        **default_training_kwargs(),
        **dict(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
        )
    }


def training_kwargs_grid_search():
    return dict(
        learning_rate=tune.grid_search([1e-5, 2e-5, 5e-5, 1e-4]),
    )
