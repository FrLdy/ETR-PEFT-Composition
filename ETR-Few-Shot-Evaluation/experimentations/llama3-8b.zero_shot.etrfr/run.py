from etr_fr.dataset import DS_KEY_ETR_FR, DS_KEY_ETR_FR_POLITIC
from icl.backbones import LLAMA3_8B_INSTRUCT
from icl.cli import tuner_cli
from icl.config import EvalLocalLLMConfig, ZeroShotConfig
from icl.eval import EvalLLM

model_id = LLAMA3_8B_INSTRUCT
eval_config = EvalLocalLLMConfig(
    model_name=model_id,
    tokenizer_name=model_id,
    eval_tasks=[DS_KEY_ETR_FR],
    test_tasks=[DS_KEY_ETR_FR, DS_KEY_ETR_FR_POLITIC],
    icl_config=ZeroShotConfig(),
)
eval_llm_cls = EvalLLM

if __name__ == "__main__":
    tuner = tuner_cli()
    tuner(
        eval_config=eval_config,
        eval_llm_cls=eval_llm_cls,
    )
