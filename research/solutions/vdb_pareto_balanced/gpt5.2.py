import os
import numpy as np
from typing import Tuple

try:
    import faiss  # type: ignore
except Exception as _e:  # pragma: no cover
    faiss = None


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)

        self.M = int(kwargs.get("M", 32))
        self.ef_construction = int(kwargs.get("ef_construction", 200))
        self.ef_search = int(kwargs.get("ef_search", 512))
        self.num_threads = int(kwargs.get("num_threads", min(8, os.cpu_count() or 1)))

        self._index = None

        if faiss is not None:
            try:
                faiss.omp_set_num_threads(self.num_threads)
            except Exception:
                pass

    def _ensure_index(self):
        if self._index is not None:
            return
        if faiss is None:
            raise RuntimeError("faiss is required but not available")

        idx = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
        idx.hnsw.efConstruction = self.ef_construction
        try:
            idx.hnsw.search_bounded_queue = True
        except Exception:
            pass
        self._index = idx

    def add(self, xb: np.ndarray) -> None:
        xb = np.ascontiguousarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim})")
        self._ensure_index()
        self._index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("index is empty; call add() first")

        xq = np.ascontiguousarray(xq, dtype=np.float32)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim})")
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        self._index.hnsw.efSearch = max(self.ef_search, k)
        D, I = self._index.search(xq, k)
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I