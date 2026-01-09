import os
import numpy as np
from typing import Tuple, Optional

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
        self.dim = int(dim)

        self.nlist = int(kwargs.get("nlist", 32768))
        self.nprobe = int(kwargs.get("nprobe", 512))

        self.quantizer_M = int(kwargs.get("quantizer_M", 32))
        self.quantizer_efConstruction = int(kwargs.get("quantizer_efConstruction", 200))
        self.quantizer_efSearch = int(kwargs.get("quantizer_efSearch", max(1024, self.nprobe * 2)))

        self.train_size = int(kwargs.get("train_size", 200000))
        self.niter = int(kwargs.get("niter", 20))
        self.threads = int(kwargs.get("threads", 8))

        self._index = None
        self._quantizer = None
        self._clustering_index = None
        self._trained = False
        self._ntotal = 0

        if faiss is None:  # pragma: no cover
            raise RuntimeError("faiss is required for this solution")

        try:
            faiss.omp_set_num_threads(max(1, self.threads))
        except Exception:
            pass

        self._build_index()

    def _build_index(self) -> None:
        d = self.dim
        self._quantizer = faiss.IndexHNSWFlat(d, self.quantizer_M, faiss.METRIC_L2)
        self._quantizer.hnsw.efConstruction = self.quantizer_efConstruction
        self._quantizer.hnsw.efSearch = self.quantizer_efSearch

        self._index = faiss.IndexIVFFlat(self._quantizer, d, self.nlist, faiss.METRIC_L2)
        self._index.nprobe = self.nprobe

        self._clustering_index = faiss.IndexFlatL2(d)
        try:
            self._index.clustering_index = self._clustering_index
        except Exception:
            pass

        try:
            cp = self._index.cp
            try:
                cp.niter = self.niter
            except Exception:
                pass
            try:
                cp.max_points_per_centroid = 256
            except Exception:
                pass
            try:
                cp.min_points_per_centroid = 10
            except Exception:
                pass
        except Exception:
            pass

    def _as_float32_c(self, x: np.ndarray) -> np.ndarray:
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        return x

    def _select_train_set(self, xb: np.ndarray, train_size: int) -> np.ndarray:
        n = xb.shape[0]
        ts = min(train_size, n)
        if ts <= 0:
            return xb[:0].copy()
        step = max(1, n // ts)
        xt = xb[::step][:ts]
        if xt.shape[0] < min(ts, n):
            need = min(ts, n) - xt.shape[0]
            if need > 0:
                xt = np.vstack([xt, xb[-need:]])
        return self._as_float32_c(xt)

    def add(self, xb: np.ndarray) -> None:
        xb = self._as_float32_c(xb)
        if xb.size == 0:
            return

        if not self._trained:
            ts = max(self.train_size, self.nlist * 2)
            xt = self._select_train_set(xb, ts)
            self._index.train(xt)
            self._trained = True

            try:
                self._quantizer.hnsw.efSearch = max(self.quantizer_efSearch, self.nprobe * 2, 1024)
            except Exception:
                pass
            self._index.nprobe = self.nprobe

        self._index.add(xb)
        self._ntotal += xb.shape[0]

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if k <= 0:
            nq = int(xq.shape[0])
            return np.empty((nq, 0), dtype=np.float32), np.empty((nq, 0), dtype=np.int64)

        xq = self._as_float32_c(xq)

        try:
            faiss.omp_set_num_threads(max(1, self.threads))
        except Exception:
            pass

        self._index.nprobe = self.nprobe
        try:
            self._quantizer.hnsw.efSearch = max(self.quantizer_efSearch, self.nprobe * 2, 1024)
        except Exception:
            pass

        D, I = self._index.search(xq, int(k))
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)

        if np.any(I < 0):
            bad = I < 0
            if bad.any():
                I = I.copy()
                D = D.copy()
                I[bad] = 0
                D[bad] = np.float32(3.4028235e38)

        return D, I