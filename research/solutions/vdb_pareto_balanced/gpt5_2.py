import numpy as np
import faiss
import os
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)
        self.M = int(kwargs.get("M", 32))
        self.ef_construction = int(kwargs.get("ef_construction", 200))
        self.ef_search = int(kwargs.get("ef_search", 200))
        self.num_threads = int(kwargs.get("num_threads", max(1, (os.cpu_count() or 8))))

        try:
            faiss.omp_set_num_threads(self.num_threads)
        except Exception:
            pass

        self.index = faiss.IndexHNSWFlat(self.dim, self.M)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search

    def add(self, xb: np.ndarray) -> None:
        xb = np.ascontiguousarray(xb, dtype=np.float32)
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        xq = np.ascontiguousarray(xq, dtype=np.float32)
        # Ensure efSearch is at least k to avoid early cutoffs
        if getattr(self.index, "hnsw", None) is not None:
            if self.index.hnsw.efSearch < max(k, self.ef_search):
                self.index.hnsw.efSearch = max(k, self.ef_search)
        D, I = self.index.search(xq, int(k))
        return D, I