from icl.cli import tuner_cli
from icl.config import EvalLLMConfig
from icl.eval import EvalLocalLLM

eval_config = EvalLLMConfig()
eval_llm_cls = EvalLocalLLM

if __name__ == "__main__":
    tuner = tuner_cli()
    tuner(
        eval_config=eval_config,
        eval_llm_cls=eval_llm_cls,
    )
