import os
import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)

        # Parameters with sensible defaults for high recall and reasonable build time
        self.M = int(kwargs.get("M", 32))
        self.ef_construction = int(kwargs.get("ef_construction", 120))
        self.ef_search = int(kwargs.get("ef_search", 320))
        self.metric = kwargs.get("metric", "l2")

        # Set FAISS threading
        n_threads = kwargs.get("num_threads", None)
        if n_threads is None:
            n_threads = os.cpu_count() or 8
        try:
            faiss.omp_set_num_threads(int(n_threads))
        except Exception:
            pass

        # Build HNSW index
        if self.metric.lower() == "l2":
            self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
        else:
            # Default to L2 if unknown
            self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)

        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search

    def add(self, xb: np.ndarray) -> None:
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32, copy=False)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError("xb must have shape (N, dim) with dim matching initialization")
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32, copy=False)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError("xq must have shape (nq, dim) with dim matching initialization")
        if k <= 0:
            raise ValueError("k must be positive")
        # Ensure efSearch is set (allow override via attribute)
        self.index.hnsw.efSearch = getattr(self, "ef_search", self.index.hnsw.efSearch)
        distances, indices = self.index.search(xq, k)
        return distances, indices