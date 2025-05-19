import logging
from pathlib import Path

from jsonargparse import ActionConfigFile, ArgumentParser

from etr_fr.dataset import AVAILABLE_DATASETS
from icl.embedding_index import (
    EMBEDDING_INDEX_DIR,
    EmbeddingIndex,
    EmbeddingIndexBuilderConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)


def create_index(
    overwrite,
    dataset_name,
    dataset,
    index_builder_config,
):
    for split, ds in dataset.items():
        logging.info(f"Processing dataset '{dataset_name}' split '{split}'.")

        embedding_index = EmbeddingIndex(
            dataset_name=dataset_name,
            split=split,
            dataset=ds,
        )

        try:
            logging.info("Attempting to load existing index.")
            embedding_index.load_index()
        except AssertionError:
            logging.warning("Index not found or invalid. Rebuilding index.")
            embedding_index.build_index(index_builder_config)
            path = embedding_index.cache_index()
            embedding_index.load_index()
            logging.info("Index built and saved in {}.".format(path))
        else:
            if overwrite:
                logging.info("Overwriting existing index as requested.")
                embedding_index.cache_index()
            else:
                logging.info(
                    "Index loaded successfully. Skipping save due to overwrite=False."
                )


def main(
    overwrite: bool,
    index_builder_config: EmbeddingIndexBuilderConfig,
):
    logging.info("Starting index creation process.")
    EMBEDDING_INDEX_DIR.absolute().mkdir(exist_ok=True)
    for ds_name, loader in AVAILABLE_DATASETS.items():
        logging.info(f"Loading dataset: {ds_name}")
        dataset = loader()
        create_index(
            overwrite,
            ds_name,
            dataset,
            index_builder_config,
        )
    logging.info("All datasets processed.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to config file (yaml/json)",
    )

    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="Whether to overwrite existing indexes",
    )

    parser.add_class_arguments(
        EmbeddingIndexBuilderConfig, "index_builder_config"
    )

    args = parser.parse_args()
    main(
        overwrite=args.overwrite,
        index_builder_config=args.index_builder_config,
    )
