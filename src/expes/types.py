from typing import Literal

SamplingStrategy = Literal[
    "concatenate",
    "first_exhausted",
    "all_exhausted",
    "balanced",
]
