import numpy as np
import pandas as pd
from searcharray import SearchArray
from agta.models import TripRecord


def record_to_text(r: TripRecord) -> str:
    return f"{r.time} {r.from_activity} to {r.to_activity} by {r.mode}, {r.distance_km}km, {r.duration_min}min"


class SparseRetriever:
    def __init__(self):
        self.records: list[TripRecord] = []
        self._indexed: SearchArray | None = None
        self._dirty = True

    def add(self, record: TripRecord):
        self.records.append(record)
        self._dirty = True

    def _rebuild_index(self):
        if not self.records:
            return
        texts = pd.Series([record_to_text(r) for r in self.records])
        self._indexed = SearchArray.index(texts)
        self._dirty = False

    def score(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        if not self.records:
            return []
        if self._dirty:
            self._rebuild_index()
        tokens = query.lower().split()
        scores = np.zeros(len(self.records))
        for token in tokens:
            scores += self._indexed.array.bm25(token)
        top_indices = np.argsort(-scores)[:k]
        return [(i, scores[i]) for i in top_indices if scores[i] > 0]
