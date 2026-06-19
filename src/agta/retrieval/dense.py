import numpy as np
from pynndescent import NNDescent
from agta.models import TripRecord

_st_model = None
_model_name = "all-MiniLM-L6-v2"

def set_embedding_model(name: str):
    global _model_name, _st_model
    _model_name = name
    _st_model = None

def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(_model_name)
    return _st_model

def embed(texts):
    return _get_st_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)

def record_to_text(r: TripRecord) -> str:
    return f"{r.time} {r.from_activity} to {r.to_activity} by {r.mode}, {r.distance_km}km, {r.duration_min}min"


class DenseRetriever:
    def __init__(self):
        self.records: list[TripRecord] = []
        self.embeddings: np.ndarray | None = None
        self.index: NNDescent | None = None
        self._dirty = True

    def add(self, record: TripRecord):
        self.records.append(record)
        self._dirty = True

    def _rebuild_index(self):
        if not self.records:
            return
        texts = [record_to_text(r) for r in self.records]
        self.embeddings = embed(texts)
        if len(self.records) >= 10:
            self.index = NNDescent(self.embeddings, metric="cosine", n_neighbors=min(10, len(self.records)))
        self._dirty = False

    def score(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        if not self.records:
            return []
        if self._dirty:
            self._rebuild_index()
        query_embedding = embed([query])
        if self.index and len(self.records) >= 10:
            indices, distances = self.index.query(query_embedding, k=min(k, len(self.records)))
            return [(i, 1 - d) for i, d in zip(indices[0], distances[0])]
        else:
            sims = (self.embeddings @ query_embedding.T).flatten()
            top_indices = np.argsort(-sims)[:k]
            return [(i, sims[i]) for i in top_indices]