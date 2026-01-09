import os
from typing import Tuple
import numpy as np

try:
    import faiss
except Exception as e:
    faiss = None


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)
        self.M = int(kwargs.get("M", 24))
        self.ef_construction = int(kwargs.get("ef_construction", 80))
        self.ef_search = int(kwargs.get("ef_search", 24))
        self.n_threads = int(kwargs.get("n_threads", min(8, os.cpu_count() or 1)))

        if faiss is None:
            raise RuntimeError("faiss is required but could not be imported")

        faiss.omp_set_num_threads(self.n_threads)

        metric = kwargs.get("metric", "l2")
        if metric in ("l2", "L2", faiss.METRIC_L2):
            self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
        elif metric in ("ip", "IP", faiss.METRIC_INNER_PRODUCT):
            self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)

        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        try:
            self.index.hnsw.search_bounded_queue = True
        except Exception:
            pass

        self.ntotal = 0

    def add(self, xb: np.ndarray) -> None:
        if xb is None:
            return
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32, copy=False)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            xb = np.ascontiguousarray(xb.reshape(-1, self.dim), dtype=np.float32)
        else:
            xb = np.ascontiguousarray(xb, dtype=np.float32)

        faiss.omp_set_num_threads(self.n_threads)
        self.index.add(xb)
        self.ntotal = int(self.index.ntotal)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if k <= 0:
            nq = 0 if xq is None else int(xq.shape[0])
            return np.empty((nq, 0), dtype=np.float32), np.empty((nq, 0), dtype=np.int64)

        if xq.dtype != np.float32:
            xq = xq.astype(np.float32, copy=False)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            xq = np.ascontiguousarray(xq.reshape(-1, self.dim), dtype=np.float32)
        else:
            xq = np.ascontiguousarray(xq, dtype=np.float32)

        faiss.omp_set_num_threads(self.n_threads)
        self.index.hnsw.efSearch = self.ef_search

        D, I = self.index.search(xq, int(k))

        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I