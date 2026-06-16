import numpy as np
from agta.models import TripRecord
from agta.retrieval.dense import DenseRetriever
from agta.retrieval.sparse import SparseRetriever


class HybridRetriever:
    def __init__(self, dense_weight: float = 0.4, sparse_weight: float = 0.3, recency_weight: float = 0.3):
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.recency_weight = recency_weight

    def add(self, record: TripRecord):
        self.dense.add(record)
        self.sparse.add(record)

    @property
    def records(self) -> list[TripRecord]:
        return self.dense.records

    def retrieve(self, query: str, current_day: int, k: int = 3) -> list[TripRecord]:
        if not self.records:
            return []

        n = len(self.records)
        dense_scores = np.zeros(n)
        sparse_scores = np.zeros(n)
        recency_scores = np.zeros(n)

        for i, s in self.dense.score(query, k=n):
            dense_scores[i] = s
        for i, s in self.sparse.score(query, k=n):
            sparse_scores[i] = s

        for i, r in enumerate(self.records):
            days_ago = max(current_day - r.day, 0)
            recency_scores[i] = np.exp(-0.3 * days_ago)

        if dense_scores.max() > 0:
            dense_scores /= dense_scores.max()
        if sparse_scores.max() > 0:
            sparse_scores /= sparse_scores.max()

        combined = (self.dense_weight * dense_scores
                    + self.sparse_weight * sparse_scores
                    + self.recency_weight * recency_scores)

        top_indices = np.argsort(-combined)[:k]
        return [self.records[i] for i in top_indices if combined[i] > 0]
