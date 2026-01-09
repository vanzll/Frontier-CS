import os
import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    faiss = None


class YourIndexClass:
    def __init__(
        self,
        dim: int,
        **kwargs,
    ):
        self.dim = int(dim)
        self.M = int(kwargs.get("M", 32))
        self.ef_construction = int(kwargs.get("ef_construction", 200))
        self.ef_search = int(kwargs.get("ef_search", 512))
        self.num_threads = int(kwargs.get("num_threads", 0))  # 0 => auto

        self._ntotal = 0
        self.index = None

        if faiss is None:
            raise RuntimeError("faiss is required but could not be imported")

        if self.num_threads <= 0:
            self.num_threads = min(8, os.cpu_count() or 1)
        try:
            faiss.omp_set_num_threads(self.num_threads)
        except Exception:
            pass

        self.index = faiss.IndexHNSWFlat(self.dim, self.M)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search

    def add(self, xb: np.ndarray) -> None:
        if xb is None or xb.size == 0:
            return
        xb = np.ascontiguousarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim})")
        self.index.add(xb)
        self._ntotal += xb.shape[0]

    def search(self, xq: np.ndarray, k: int):
        k = int(k)
        if k <= 0:
            nq = 0 if xq is None else int(xq.shape[0])
            return np.empty((nq, 0), dtype=np.float32), np.empty((nq, 0), dtype=np.int64)

        if xq is None or xq.size == 0:
            return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64)

        xq = np.ascontiguousarray(xq, dtype=np.float32)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim})")

        nq = xq.shape[0]
        if self._ntotal == 0:
            D = np.full((nq, k), np.inf, dtype=np.float32)
            I = np.full((nq, k), -1, dtype=np.int64)
            return D, I

        try:
            self.index.hnsw.efSearch = max(self.ef_search, k * 8)
        except Exception:
            pass

        D, I = self.index.search(xq, k)
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I