import os
import numpy as np
from typing import Tuple

try:
    import faiss  # type: ignore
except Exception as _e:
    faiss = None


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)

        self.nlist = int(kwargs.get("nlist", 16384))
        self.nprobe = int(kwargs.get("nprobe", 384))
        self.train_size = int(kwargs.get("train_size", 200000))
        self.niter = int(kwargs.get("niter", 20))
        self.max_points_per_centroid = int(kwargs.get("max_points_per_centroid", 256))
        self.seed = int(kwargs.get("seed", 12345))

        self.n_threads = kwargs.get("n_threads", None)
        if self.n_threads is None:
            self.n_threads = os.cpu_count() or 8
        self.n_threads = int(self.n_threads)

        self._index = None
        self._ntotal = 0

        if faiss is None:
            self._xb = None
        else:
            try:
                faiss.omp_set_num_threads(self.n_threads)
            except Exception:
                pass

            quantizer = faiss.IndexFlatL2(self.dim)
            self._index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)
            self._set_training_params()
            self._index.nprobe = min(self.nprobe, self.nlist)

    def _set_training_params(self) -> None:
        if self._index is None:
            return
        try:
            cp = self._index.cp
            try:
                cp.seed = self.seed
            except Exception:
                pass
            cp.niter = self.niter
            cp.max_points_per_centroid = self.max_points_per_centroid
            try:
                cp.min_points_per_centroid = 5
            except Exception:
                pass
        except Exception:
            pass

    def _ensure_float32(self, x: np.ndarray) -> np.ndarray:
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        return x

    def add(self, xb: np.ndarray) -> None:
        xb = self._ensure_float32(xb)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim})")

        if faiss is None:
            if self._xb is None:
                self._xb = xb.copy()
            else:
                self._xb = np.vstack((self._xb, xb))
            self._ntotal = int(self._xb.shape[0])
            return

        if self._index is None:
            raise RuntimeError("FAISS index not initialized")

        if not self._index.is_trained:
            n = xb.shape[0]
            ts = min(self.train_size, n)
            rng = np.random.default_rng(self.seed)
            idx = rng.integers(0, n, size=ts, endpoint=False, dtype=np.int64)
            train_x = np.ascontiguousarray(xb[idx])
            self._index.train(train_x)

        self._index.add(xb)
        self._ntotal += int(xb.shape[0])

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")
        xq = self._ensure_float32(xq)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim})")

        if faiss is None:
            if self._xb is None or self._xb.shape[0] == 0:
                D = np.full((xq.shape[0], k), np.inf, dtype=np.float32)
                I = np.full((xq.shape[0], k), -1, dtype=np.int64)
                return D, I
            xb = self._xb
            xq_norm = (xq * xq).sum(axis=1, keepdims=True)
            xb_norm = (xb * xb).sum(axis=1, keepdims=True).T
            dots = xq @ xb.T
            dist = xq_norm + xb_norm - 2.0 * dots
            idx = np.argpartition(dist, kth=min(k - 1, dist.shape[1] - 1), axis=1)[:, :k]
            row = np.arange(dist.shape[0])[:, None]
            dsel = dist[row, idx]
            order = np.argsort(dsel, axis=1)
            I = idx[row, order].astype(np.int64, copy=False)
            D = dsel[row, order].astype(np.float32, copy=False)
            return D, I

        if self._index is None or self._index.ntotal == 0:
            D = np.full((xq.shape[0], k), np.inf, dtype=np.float32)
            I = np.full((xq.shape[0], k), -1, dtype=np.int64)
            return D, I

        self._index.nprobe = min(self.nprobe, self.nlist)
        D, I = self._index.search(xq, k)
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I