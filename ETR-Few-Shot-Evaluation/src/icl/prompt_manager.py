import re
from typing import Dict, List, Literal

from etr_fr.dataset import DST_KEY, SRC_KEY, TRAIN_SPLIT
from icl.embedding_index import SRC_EMBEDDING_KEY, EmbeddingIndex, Shot
from icl.eval import EvalLLM, EvalLLMFewShot
from icl.models import Shots


class PromptManager:
    def __init__(self, eval_llm: EvalLLM) -> None:
        self.eval_llm = eval_llm
        self.config = eval_llm.config.icl_config
        self.prompt_template = self.config.prompt_template

    def get_user_prompt(self, input, **kwargs):
        return self.prompt_template.input_prompt.format(input=input, **kwargs)

    def get_system_prompt(self, **kwargs):
        return self.prompt_template.system_prompt.format(**kwargs)

    def get_pred(self, pred):
        return pred


class COTPromptManager(PromptManager):
    def get_pred(self, pred):
        template = self.prompt_template.output_prefix
        match = re.search(
            f"{template}(.*?){template}",
            pred,
            re.DOTALL,
        )
        cleaned = match.group(1) if match else pred.replace(template, "")
        # cleaned = re.sub(rf"^{template}\s*", "", pred)
        return cleaned


class FewShotPromptManager(COTPromptManager):
    def __init__(self, eval_llm: EvalLLMFewShot) -> None:
        assert isinstance(eval_llm, EvalLLMFewShot)
        super().__init__(eval_llm)
        self.train_indexes: Dict[str, EmbeddingIndex] = (
            eval_llm.embedding_indexes[TRAIN_SPLIT]
        )

    def get_shots(self, src_embedding):
        k = self.config.k
        train_tasks = self.eval_llm.config.train_tasks
        shots = []
        for task in train_tasks:
            index = self.train_indexes[task]
            shots += index.search(src_embedding, k).shots

        return Shots(shots=shots, task_order=train_tasks)

    def get_system_prompt(self, **kwargs):
        shots = self.get_shots(kwargs.get(SRC_EMBEDDING_KEY))
        shots = shots.order(self.config.examples_ordering)
        if self.config.task_to_name is not None:
            shots = shots.rename_tasks(self.config.task_to_name)

        shot_prompts = "\n".join(
            [
                self.prompt_template.shot_template.format(
                    i=i + 1,
                    task=shot.task,
                    input=shot.content[SRC_KEY],
                    output=shot.content[DST_KEY],
                )
                for i, shot in enumerate(shots)
            ]
        )

        return self.prompt_template.system_prompt.format(shots=shot_prompts)

    def get_user_prompt(self, input, **kwargs):
        task = kwargs["task"]
        if self.config.task_to_name is not None:
            task = self.config.task_to_name.get(task, task)
        return super().get_user_prompt(input, task=task)


PROMPT_MANAGER_CLASSES = {
    c.__name__: c
    for c in [
        PromptManager,
        COTPromptManager,
        FewShotPromptManager,
    ]
}

PromptManagerClassName = Literal[tuple(PROMPT_MANAGER_CLASSES.keys())]
