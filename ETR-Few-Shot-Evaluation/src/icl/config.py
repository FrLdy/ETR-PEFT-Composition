import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Type

from datasets.utils.py_utils import Literal

from etr_fr.dataset import (
    DS_KEY_ETR_FR,
    DS_KEY_ETR_FR_POLITIC,
    DS_KEY_ORANGESUM,
    DS_KEY_WIKILARGE_FR,
)
from icl.llm import LLM, LLM_CLASSES, LLMClassName, LocalLLM
from icl.models import ExamplesOrdering
from icl.prompt_manager import (
    PROMPT_MANAGER_CLASSES,
    COTPromptManager,
    FewShotPromptManager,
    PromptManager,
    PromptManagerClassName,
)
from icl.prompts import PromptTemplate, PromptTemplates

DEFAULT_TASK_TO_NAME = {
    DS_KEY_ETR_FR: "FALC",
    DS_KEY_ETR_FR_POLITIC: "FALC",
    DS_KEY_ORANGESUM: "Résumé",
    DS_KEY_WIKILARGE_FR: "Simplification",
}


@dataclass
class ICLConfig:
    prompt_template: PromptTemplate
    mode: str


@dataclass
class ZeroShotConfig(ICLConfig):
    mode: str = "zero_shot"
    prompt_template: PromptTemplate = field(
        default_factory=lambda: PromptTemplates.ZERO_SHOT.value
    )


@dataclass
class COTConfig(ICLConfig):
    mode: str = "cot"
    prompt_template: PromptTemplate = field(
        default_factory=lambda: PromptTemplates.COT.value
    )


@dataclass
class FewShotConfig(ICLConfig):
    mode: str = "few_shot"
    k: int = 3
    rag: bool = True
    examples_ordering: ExamplesOrdering = "sort_similarity_ascending"
    task_to_name: Dict[str, str] = field(
        default_factory=lambda: DEFAULT_TASK_TO_NAME
    )
    prompt_template: PromptTemplate = field(
        default_factory=lambda: PromptTemplates.FEW_SHOT.value
    )


@dataclass
class EvalLLMConfig:
    model_name: str
    tokenizer_name: str
    icl_config: ICLConfig
    llm_cls_name: LLMClassName = LocalLLM.__name__
    prompt_manager_cls_name: PromptManagerClassName = PromptManager.__name__
    generation_config: Optional[Dict] = None
    train_tasks: List[str] = field(default_factory=list)
    test_tasks: List[str] = field(default_factory=list)
    eval_tasks: List[str] = field(default_factory=list)
    lang: Literal["fr", "en"] = "fr"

    @property
    def llm_cls(self):
        return LLM_CLASSES.get(self.llm_cls_name)

    @property
    def prompt_manager_cls(self):
        return PROMPT_MANAGER_CLASSES.get(self.prompt_manager_cls_name)

    @property
    def tasks(self):
        return set(
            [
                *self.train_tasks,
                *self.eval_tasks,
                *self.test_tasks,
            ]
        )


EvalLocalLLMConfig = partial(
    EvalLLMConfig,
)

EvalLocalLLMZeroShotConfig = partial(
    EvalLocalLLMConfig,
)

EvalLocalLLMRAGConfig = partial(
    EvalLocalLLMConfig,
    prompt_manager_cls_name=FewShotPromptManager.__name__,
)

EvalLocalLLCOTConfig = partial(
    EvalLocalLLMConfig,
    prompt_manager_cls_name=COTPromptManager.__name__,
)


def make_icl_config(icl_data: Dict[str, Any]) -> ICLConfig:
    mode = icl_data.get("mode")
    icl_data["prompt_template"] = PromptTemplate(
        **icl_data.pop("prompt_template")
    )

    mode_to_config = {
        "zero_shot": ZeroShotConfig,
        "few_shot": FewShotConfig,
        "cot": COTConfig,
    }

    config_class = mode_to_config.get(mode)
    if config_class is None:
        raise ValueError(f"Unknown ICL mode: {mode}")

    return config_class(**icl_data)


def deserialize_eval_llm_config(data: Dict[str, Any]) -> EvalLLMConfig:
    icl_data = data.pop("icl_config")
    icl_config = make_icl_config(icl_data)

    return EvalLLMConfig(
        icl_config=icl_config,
        **data,
    )
