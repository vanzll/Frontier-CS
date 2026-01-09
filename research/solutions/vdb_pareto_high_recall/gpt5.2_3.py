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
        self.ef_construction = int(kwargs.get("ef_construction", 200))
        self.ef_search = int(kwargs.get("ef_search", 512))

        self.num_threads = kwargs.get("num_threads", None)
        if self.num_threads is None:
            try:
                self.num_threads = max(1, min(8, os.cpu_count() or 1))
            except Exception:
                self.num_threads = 8
        self.num_threads = int(self.num_threads)

        self._index = None
        self._xb_fallback = None

        if faiss is not None:
            try:
                faiss.omp_set_num_threads(self.num_threads)
            except Exception:
                pass

            self._index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
            h = self._index.hnsw
            try:
                h.efConstruction = self.ef_construction
            except Exception:
                pass
            try:
                h.efSearch = self.ef_search
            except Exception:
                pass
            try:
                h.search_bounded_queue = True
            except Exception:
                pass

    def add(self, xb: np.ndarray) -> None:
        xb = np.asarray(xb)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim})")

        if xb.dtype != np.float32 or not xb.flags["C_CONTIGUOUS"]:
            xb = np.ascontiguousarray(xb, dtype=np.float32)

        if self._index is not None:
            self._index.add(xb)
        else:
            if self._xb_fallback is None:
                self._xb_fallback = xb.copy()
            else:
                self._xb_fallback = np.vstack((self._xb_fallback, xb))

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        xq = np.asarray(xq)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim})")

        if xq.dtype != np.float32 or not xq.flags["C_CONTIGUOUS"]:
            xq = np.ascontiguousarray(xq, dtype=np.float32)

        if self._index is not None:
            nt = int(self._index.ntotal)
            kk = min(k, nt) if nt > 0 else k
            D, I = self._index.search(xq, kk)

            if kk < k:
                nq = xq.shape[0]
                D_pad = np.full((nq, k), np.inf, dtype=np.float32)
                I_pad = np.full((nq, k), -1, dtype=np.int64)
                D_pad[:, :kk] = D
                I_pad[:, :kk] = I
                return D_pad, I_pad

            return D.astype(np.float32, copy=False), I.astype(np.int64, copy=False)

        # Fallback exact search (slow; only used if faiss is unavailable)
        xb = self._xb_fallback
        if xb is None or xb.shape[0] == 0:
            nq = xq.shape[0]
            return np.full((nq, k), np.inf, dtype=np.float32), np.full((nq, k), -1, dtype=np.int64)

        N = xb.shape[0]
        kk = min(k, N)

        xq2 = np.sum(xq * xq, axis=1, keepdims=True).astype(np.float32, copy=False)
        xb2 = np.sum(xb * xb, axis=1, keepdims=True).T.astype(np.float32, copy=False)
        dist = xq2 + xb2 - 2.0 * (xq @ xb.T)

        idx = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
        dsel = dist[np.arange(dist.shape[0])[:, None], idx]
        order = np.argsort(dsel, axis=1)
        I = idx[np.arange(idx.shape[0])[:, None], order].astype(np.int64, copy=False)
        D = dsel[np.arange(dsel.shape[0])[:, None], order].astype(np.float32, copy=False)

        if kk < k:
            nq = xq.shape[0]
            D_pad = np.full((nq, k), np.inf, dtype=np.float32)
            I_pad = np.full((nq, k), -1, dtype=np.int64)
            D_pad[:, :kk] = D
            I_pad[:, :kk] = I
            return D_pad, I_pad

        return D, I