import os
from typing import Tuple

import numpy as np

try:
    import faiss
except Exception:
    faiss = None


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)
        self.M = int(kwargs.get("M", 32))
        self.ef_construction = int(kwargs.get("ef_construction", 100))
        self.ef_search = int(kwargs.get("ef_search", 64))
        self._index = None

        # Configure FAISS threading if available
        if faiss is not None:
            try:
                n_threads = kwargs.get("num_threads", None)
                if n_threads is None:
                    n_threads = os.cpu_count() or 8
                n_threads = max(1, int(n_threads))
                faiss.omp_set_num_threads(n_threads)
            except Exception:
                pass

    def _ensure_index(self):
        if self._index is None:
            if faiss is None:
                raise RuntimeError("faiss is required for this index.")
            index = faiss.IndexHNSWFlat(self.dim, self.M)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = max(self.ef_search, 1)
            self._index = index

    def add(self, xb: np.ndarray) -> None:
        self._ensure_index()
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32, copy=False)
        xb = np.ascontiguousarray(xb)
        self._index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Index is empty. Call add() before search().")
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32, copy=False)
        xq = np.ascontiguousarray(xq)
        # Ensure efSearch >= k for HNSW correctness
        try:
            self._index.hnsw.efSearch = max(self.ef_search, int(k))
        except Exception:
            pass
        D, I = self._index.search(xq, int(k))
        return D, I