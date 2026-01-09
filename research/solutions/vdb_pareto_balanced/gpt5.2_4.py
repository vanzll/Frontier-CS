import os
import numpy as np
from typing import Tuple

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)

        self.M = int(kwargs.get("M", 32))
        self.ef_construction = int(kwargs.get("ef_construction", 150))
        self.ef_search = int(kwargs.get("ef_search", 512))
        self.add_chunk_size = int(kwargs.get("add_chunk_size", 200_000))

        self.num_threads = int(kwargs.get("num_threads", max(1, min(8, os.cpu_count() or 1))))
        self._index = None
        self._ntotal = 0

        if faiss is not None:
            try:
                faiss.omp_set_num_threads(self.num_threads)
            except Exception:
                pass

    def add(self, xb: np.ndarray) -> None:
        if xb is None:
            return
        xb = np.asarray(xb)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim})")
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32, copy=False)
        xb = np.ascontiguousarray(xb)

        n = xb.shape[0]
        if n == 0:
            return

        if faiss is None:
            if not hasattr(self, "_xb"):
                self._xb = xb.copy()
            else:
                self._xb = np.vstack([self._xb, xb])
            self._ntotal = int(self._xb.shape[0])
            return

        if self._index is None:
            try:
                index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
            except Exception:
                index = faiss.IndexHNSWFlat(self.dim, self.M)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            try:
                index.verbose = False
            except Exception:
                pass
            self._index = index

        if self.add_chunk_size > 0 and n > self.add_chunk_size:
            for i in range(0, n, self.add_chunk_size):
                self._index.add(xb[i : i + self.add_chunk_size])
        else:
            self._index.add(xb)

        self._ntotal += int(n)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        xq = np.asarray(xq)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim})")
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32, copy=False)
        xq = np.ascontiguousarray(xq)

        nq = xq.shape[0]
        if nq == 0:
            return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64)

        if self._ntotal == 0:
            D = np.full((nq, k), np.float32(np.inf), dtype=np.float32)
            I = np.full((nq, k), -1, dtype=np.int64)
            return D, I

        if faiss is None:
            xb = self._xb
            xq_norm = (xq * xq).sum(axis=1, keepdims=True)
            xb_norm = (xb * xb).sum(axis=1, keepdims=True).T
            dist = xq_norm + xb_norm - 2.0 * (xq @ xb.T)
            if k == 1:
                I = dist.argmin(axis=1).astype(np.int64)[:, None]
                D = dist[np.arange(nq), I[:, 0]].astype(np.float32)[:, None]
                return D, I
            idx = np.argpartition(dist, kth=min(k - 1, dist.shape[1] - 1), axis=1)[:, :k]
            row = np.arange(nq)[:, None]
            dsel = dist[row, idx]
            order = np.argsort(dsel, axis=1)
            I = idx[row, order].astype(np.int64)
            D = dist[row, I].astype(np.float32)
            return D, I

        self._index.hnsw.efSearch = max(self.ef_search, k * 32)
        D, I = self._index.search(xq, k)
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I