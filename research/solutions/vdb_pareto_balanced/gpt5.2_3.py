import os
from typing import Tuple, Optional

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None


class YourIndexClass:
    def __init__(
        self,
        dim: int,
        **kwargs,
    ):
        if faiss is None:
            raise ImportError("faiss is required for this solution")

        self.dim = int(dim)

        M = int(kwargs.get("M", 32))
        ef_construction = int(kwargs.get("ef_construction", kwargs.get("efConstruction", 200)))
        ef_search = int(kwargs.get("ef_search", kwargs.get("efSearch", 512)))

        threads = kwargs.get("threads", kwargs.get("n_threads", None))
        if threads is None:
            threads = os.cpu_count() or 1
        self.threads = int(max(1, threads))
        faiss.omp_set_num_threads(self.threads)

        self.index = faiss.IndexHNSWFlat(self.dim, M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

    def add(self, xb: np.ndarray) -> None:
        xb = np.ascontiguousarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim}), got {xb.shape}")
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        xq = np.ascontiguousarray(xq, dtype=np.float32)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim}), got {xq.shape}")

        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        D, I = self.index.search(xq, k)
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I