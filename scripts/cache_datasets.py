import logging

from datasets.config import HF_DATASETS_CACHE
from datasets.utils.py_utils import Path

from etr_fr_expes.dataset import (
    AVAILABLE_DATASETS,
    DS_KEY_ETR_FR,
    DS_KEY_WIKILARGE_FR,
)

DATASET_DIR = Path(__file__).parents[1] / "data"
DATASETS_LOCATIONS = {
    DS_KEY_ETR_FR: DATASET_DIR / "etr-fr/",
    DS_KEY_WIKILARGE_FR: DATASET_DIR / "wikilarge-fr/",
}

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

if __name__ == "__main__":
    for name, loader in AVAILABLE_DATASETS.items():
        loc = DATASETS_LOCATIONS.get(name)
        if loc is not None:
            loc = loc.resolve(strict=True).as_posix()
        logging.info(f"Caching {name} dataset in {HF_DATASETS_CACHE}")
        _ = loader(location=loc)
        logging.info(f"{name} dataset in cache.")
