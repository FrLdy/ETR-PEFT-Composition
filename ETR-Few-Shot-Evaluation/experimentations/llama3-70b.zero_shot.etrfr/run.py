from etr_fr.dataset import DS_KEY_ETR_FR, DS_KEY_ETR_FR_POLITIC
from icl.cli import tuner_cli
from icl.config import EvalLLMConfig, ZeroShotConfig
from icl.eval import EvalLocalLLM

model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
eval_config = EvalLLMConfig(
    model_name=model_id,
    tokenizer_name=model_id,
    eval_tasks=[DS_KEY_ETR_FR],
    test_tasks=[DS_KEY_ETR_FR, DS_KEY_ETR_FR_POLITIC],
    icl_config=ZeroShotConfig(),
)
eval_llm_cls = EvalLocalLLM

if __name__ == "__main__":
    tuner = tuner_cli()
    tuner(
        eval_config=eval_config,
        eval_llm_cls=eval_llm_cls,
    )
