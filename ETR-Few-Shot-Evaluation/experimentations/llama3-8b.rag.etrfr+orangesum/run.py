from ray import tune

from etr_fr.dataset import (
    DS_KEY_ETR_FR,
    DS_KEY_ETR_FR_POLITIC,
    DS_KEY_ORANGESUM,
)
from icl.backbones import LLAMA3_8B_INSTRUCT
from icl.cli import tuner_cli
from icl.config import (
    EvalLocalLLMRAGConfig,
    FewShotConfig,
)
from icl.eval import EvalLLMFewShot
from icl.hp_search import MTLRAGConfig

model_id = LLAMA3_8B_INSTRUCT
eval_config = MTLRAGConfig(
    model_name=model_id,
    tokenizer_name=model_id,
    train_tasks=[DS_KEY_ETR_FR, DS_KEY_ORANGESUM],
)
eval_llm_cls = EvalLLMFewShot

if __name__ == "__main__":
    tuner = tuner_cli()
    tuner(
        eval_config=eval_config,
        eval_llm_cls=eval_llm_cls,
    )
