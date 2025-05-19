from etr_fr.dataset import (
    DS_KEY_ETR_FR,
    DS_KEY_ORANGESUM,
    DS_KEY_WIKILARGE_FR,
)
from icl.backbones import MISTRAL_7B_INSTRUCT
from icl.cli import tuner_cli
from icl.eval import EvalLLMFewShot
from icl.hp_search import MTLRAGConfig

model_id = MISTRAL_7B_INSTRUCT
eval_config = MTLRAGConfig(
    model_name=model_id,
    tokenizer_name=model_id,
    train_tasks=[DS_KEY_ETR_FR, DS_KEY_ORANGESUM, DS_KEY_WIKILARGE_FR],
)
eval_llm_cls = EvalLLMFewShot

if __name__ == "__main__":
    tuner = tuner_cli()
    tuner(
        eval_config=eval_config,
        eval_llm_cls=eval_llm_cls,
    )
