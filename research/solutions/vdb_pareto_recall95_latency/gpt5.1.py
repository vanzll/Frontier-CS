import numpy as np
from typing import Tuple
import os

import faiss


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim

        M = int(kwargs.get("M", 32))
        ef_construction = int(
            kwargs.get("ef_construction", kwargs.get("efConstruction", 200))
        )
        ef_search = int(kwargs.get("ef_search", kwargs.get("efSearch", 64)))

        num_threads = kwargs.get("num_threads", None)
        if num_threads is None:
            cpu_count = os.cpu_count() or 1
            num_threads = min(8, cpu_count)
        self.num_threads = int(num_threads)
        faiss.omp_set_num_threads(self.num_threads)

        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

    def add(self, xb: np.ndarray) -> None:
        if xb is None:
            return
        xb = np.asarray(xb, dtype="float32")
        xb = np.ascontiguousarray(xb)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim})")
        faiss.omp_set_num_threads(self.num_threads)
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        xq = np.asarray(xq, dtype="float32")
        xq = np.ascontiguousarray(xq)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim})")
        if k <= 0:
            raise ValueError("k must be positive")

        ntotal = self.index.ntotal
        nq = xq.shape[0]

        if ntotal == 0:
            D_empty = np.full((nq, k), np.inf, dtype="float32")
            I_empty = np.full((nq, k), -1, dtype="int64")
            return D_empty, I_empty

        search_k = min(k, ntotal)
        faiss.omp_set_num_threads(self.num_threads)
        D, I = self.index.search(xq, search_k)

        if search_k < k:
            D_full = np.full((nq, k), np.inf, dtype="float32")
            I_full = np.full((nq, k), -1, dtype="int64")
            D_full[:, :search_k] = D
            I_full[:, :search_k] = I
            D, I = D_full, I_full

        if D.dtype != np.float32:
            D = D.astype("float32", copy=False)
        if I.dtype != np.int64:
            I = I.astype("int64", copy=False)

        return D, I