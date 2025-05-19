import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type

import faiss
from datasets import Dataset
from sentence_transformers import SentenceTransformer

from etr_fr import dataset
from icl.models import Shot, Shots

DEFAULT_CACHE_DIR = Path("~/.cache")
DEFAULT_EMBEDDING_INDEX_DIR = DEFAULT_CACHE_DIR / "embedding_index"
EMBEDDING_INDEX_DIR = Path(
    os.getenv("EMBEDDING_INDEX_DIR", DEFAULT_EMBEDDING_INDEX_DIR)
).expanduser()

SRC_EMBEDDING_KEY = "src_embedding"


@dataclass
class EmbeddingIndexBuilderConfig:
    model_name: str = "jinaai/jina-embeddings-v3"
    index_cls: Type[faiss.Index] = faiss.IndexFlatIP

    @property
    def config_name(self):
        return "+".join([self.model_name, self.index_cls.__name__.lower()])


DEFAULT_EMBEDDING_INDEX_BUILDER_CONFIG = EmbeddingIndexBuilderConfig()


class EmbeddingIndexBuilder:
    _model_instances = {}

    def __init__(self, config: EmbeddingIndexBuilderConfig) -> None:
        self.config = config

        if config.model_name not in self._model_instances:
            instance = SentenceTransformer(
                config.model_name, trust_remote_code=True
            )
            self._model_instances[config.model_name] = instance

        self.model = self._model_instances.get(config.model_name)

    def __call__(self, texts):
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        dim = embeddings.shape[1]
        index = self.config.index_cls(dim)
        index.add(embeddings)

        return index


@dataclass
class EmbeddingIndex:
    dataset_name: str
    split: str
    dataset: Dataset
    index: Optional[faiss.Index] = None

    @property
    def name(self):
        return f"{self.dataset_name}-{self.split}"

    @property
    def index_cache_path(self):
        return EMBEDDING_INDEX_DIR / f"{self.name}.index"

    def set_index(self, index):
        self.index = index
        self._add_embeddings_colum()

    def build_index(
        self, builder_config=DEFAULT_EMBEDDING_INDEX_BUILDER_CONFIG
    ):
        index_builder = EmbeddingIndexBuilder(config=builder_config)
        texts = self.dataset[dataset.SRC_KEY]
        index = index_builder(texts)
        self.set_index(index)

    def cache_index(self):
        path = self.index_cache_path
        assert self.index is not None
        faiss.write_index(self.index, path.as_posix())
        return path

    def _add_embeddings_colum(self):
        key = SRC_EMBEDDING_KEY
        embeddings = [
            self.index.reconstruct(i) for i in range(self.index.ntotal)
        ]
        self.dataset = self.dataset.add_column(key, embeddings)
        self.dataset.set_format(
            type="numpy", columns=[key], output_all_columns=True
        )

    def load_index(self):
        path = self.index_cache_path
        assert path.exists()
        index = faiss.read_index(path.as_posix())
        assert len(self.dataset) == index.ntotal
        self.set_index(index)
        return self.index

    def search(self, query_embedding, k):
        distances, idxs = self.index.search(query_embedding[None, :], k=k)
        distances = distances.tolist()[0]
        idxs = idxs.tolist()[0]
        examples = [self.dataset[i] for i in idxs]
        return Shots(
            [
                Shot(
                    content=ex,
                    similariry=sim,
                    task=self.dataset_name,
                )
                for ex, sim in zip(examples, distances)
            ]
        )
