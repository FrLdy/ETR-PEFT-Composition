from ray import tune

from etr_fr.dataset import DS_KEY_ETR_FR, DS_KEY_ETR_FR_POLITIC
from icl.backbones import MISTRAL_7B_INSTRUCT
from icl.cli import tuner_cli
from icl.config import (
    EvalLocalLLMRAGConfig,
    FewShotConfig,
)
from icl.eval import EvalLLMFewShot

model_id = MISTRAL_7B_INSTRUCT
eval_config = EvalLocalLLMRAGConfig(
    model_name=model_id,
    tokenizer_name=model_id,
    train_tasks=[DS_KEY_ETR_FR],
    eval_tasks=[DS_KEY_ETR_FR],
    test_tasks=[DS_KEY_ETR_FR, DS_KEY_ETR_FR_POLITIC],
    icl_config=FewShotConfig(
        k=tune.grid_search([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        examples_ordering="shuffle",
    ),
)
eval_llm_cls = EvalLLMFewShot

if __name__ == "__main__":
    tuner = tuner_cli()
    tuner(
        eval_config=eval_config,
        eval_llm_cls=eval_llm_cls,
    )
