import unittest

import torch
from transformers.testing_utils import require_torch, torch_device

from adapters import init
from adapters.composition import BatchSplit, MultiTaskLearning
from adapters.configuration.adapter_config import (
    LoRAConfig,
    MTLConfigUnion,
    MTLLoRAConfig,
)
from expes.tests.utils import get_trainable_param_names, ids_tensor


@require_torch
class ModelBaseTestMixin:

    def build_model(self): ...

    def build_adapter_model(self):
        model = self.build_model()
        init(model)
        return model

    def build_lora_model(self, adapter_name, trainable=True):
        model = self.build_adapter_model()
        lora_config = LoRAConfig()
        model.add_adapter(adapter_name, lora_config)
        model.active_adapters = adapter_name
        if trainable:
            model.train_adapter(adapter_name)

        return adapter_name, model

    def build_loras_mtl_compositions(self, trainable=True):
        model = self.build_adapter_model()
        lora_config = LoRAConfig()
        names = ["lora1", "lora2", "lora3"]
        for name in names:
            model.add_adapter(name, lora_config)

        model.active_adapters = MultiTaskLearning(*names)
        if trainable:
            model.train_adapter(model.active_adapters)

        return names, model

    def inputs(self, bsz=1, n_tasks=0):
        inputs = {
            "input_ids": ids_tensor((bsz, 128), 1000).to(torch_device),
            "labels": ids_tensor((bsz, 50), 1000).to(torch_device),
        }
        if n_tasks > 0:
            inputs["task_ids"] = torch.randint(0, n_tasks, (bsz,))

        return inputs

    def training_pass(self, model, inputs=None):
        inputs = inputs or self.inputs()
        loss = model(**inputs).loss
        loss.backward()

    def batched_training_pass(self, model, inputs=None):
        inputs = inputs or self.inputs(bsz=5)
        self.training_pass(model, inputs)

    def test_add_lora(self):
        adapter_name = "lora1"
        _, model = self.build_lora_model(adapter_name)
        trainable_parameters = get_trainable_param_names(model)
        assert all(
            adapter_name in param_name for param_name in trainable_parameters
        )

    def test_fwd_lora(self):
        _, model = self.build_lora_model("lora1")
        self.training_pass(model)
        self.batched_training_pass(model)

    def test_fwd_loras_batch_split(self):
        adapter_names, model = self.build_loras_mtl_compositions()
        model.active_adapters = BatchSplit(
            *adapter_names, batch_sizes=[2, 2, 2]
        )
        self.training_pass(model, self.inputs(6))

    def test_add_mtl_loras(self):
        adapter_names, model = self.build_loras_mtl_compositions()
        trainable_parameters = get_trainable_param_names(model)
        assert all(
            any(adapter_name in param_name for adapter_name in adapter_names)
            for param_name in trainable_parameters
        )

    def test_fwd_loras_mtl_split(self):
        _, model = self.build_loras_mtl_compositions()
        n_tasks = len(model.active_adapters)
        self.training_pass(model, self.inputs(1, n_tasks))
        self.batched_training_pass(model, self.inputs(5, n_tasks))

    def build_MTL_lora_model(self, trainable=True):
        model = self.build_adapter_model()
        config = MTLConfigUnion(
            task_names=["lora1", "lora2", "lora3"],
            base_config=MTLLoRAConfig(
                n_up_projection=3,
            ),
        )
        union_name = "mtl-lora"
        model.add_adapter(union_name, config)
        model.active_adapters = MultiTaskLearning(*config.task_names)

        if trainable:
            model.train_adapter(model.active_adapters)

        return union_name, config.task_names, model

    def test_add_MTL_lora(self):
        union_name, adapter_names, model = self.build_MTL_lora_model()
        attn_matices = model.adapters_config.get(
            union_name
        ).base_config.attn_matrices
        for layer in model.model.layers:
            for attn_mat in attn_matices:
                trainable_parameters = list(
                    get_trainable_param_names(
                        getattr(layer.self_attn, f"{attn_mat}_proj")
                    )
                )
                assert all(
                    f"loras.{adapter_name}.weights" in trainable_parameters
                    for adapter_name in adapter_names
                )
                assert all(
                    any(
                        f"loras.{adapter_name}.task_specific" in param_name
                        for param_name in trainable_parameters
                    )
                    for adapter_name in adapter_names
                )
                assert all(
                    f"shared_parameters.{union_name}.{param_name}"
                    for param_name in ["lora_A", "lora_B"]
                )
