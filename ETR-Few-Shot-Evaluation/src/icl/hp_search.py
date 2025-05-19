from functools import partial

from ray import tune

from etr_fr.dataset import DS_KEY_ETR_FR, DS_KEY_ETR_FR_POLITIC
from icl.config import EvalLocalLLMRAGConfig, FewShotConfig

MTLRAGConfig = partial(
    EvalLocalLLMRAGConfig,
    eval_tasks=[DS_KEY_ETR_FR],
    test_tasks=[DS_KEY_ETR_FR, DS_KEY_ETR_FR_POLITIC],
    icl_config=FewShotConfig(
        k=tune.grid_search([1, 2, 3]),
        examples_ordering=tune.grid_search(
            ["alternating_tasks", "successive_tasks", "shuffle"]
        ),
    ),
)
