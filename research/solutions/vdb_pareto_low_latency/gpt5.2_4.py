import os
from typing import Tuple, Optional

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


def _as_f32_c(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    if not x.flags.c_contiguous:
        x = np.ascontiguousarray(x)
    return x


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)

        self.nlist = int(kwargs.get("nlist", 4096))
        self.nprobe = int(kwargs.get("nprobe", 8))
        self.train_size = int(kwargs.get("train_size", 200_000))
        self.add_chunk_size = int(kwargs.get("add_chunk_size", 200_000))

        self.threads = int(kwargs.get("threads", min(8, os.cpu_count() or 1)))
        self._rng_seed = int(kwargs.get("seed", 12345))

        self._index = None
        self._fallback_xb: Optional[np.ndarray] = None
        self._ntotal = 0

        if faiss is not None:
            faiss.omp_set_num_threads(self.threads)
            quantizer = faiss.IndexFlatL2(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)
            index.nprobe = self.nprobe

            # Try to reduce training time a bit if supported
            try:
                index.cp.niter = int(kwargs.get("niter", 15))
            except Exception:
                pass
            try:
                index.cp.max_points_per_centroid = int(kwargs.get("max_points_per_centroid", 256))
            except Exception:
                pass
            try:
                index.cp.min_points_per_centroid = int(kwargs.get("min_points_per_centroid", 5))
            except Exception:
                pass

            self._index = index

    def add(self, xb: np.ndarray) -> None:
        if xb is None:
            return
        xb = _as_f32_c(xb)
        if xb.size == 0:
            return
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim}), got {xb.shape}")

        if self._index is None:
            if self._fallback_xb is None:
                self._fallback_xb = xb.copy()
            else:
                self._fallback_xb = np.vstack((self._fallback_xb, xb))
            self._ntotal = int(self._fallback_xb.shape[0])
            return

        if not self._index.is_trained:
            n = xb.shape[0]
            ts = min(self.train_size, n)

            if ts < self.nlist:
                self.nlist = max(1, min(self.nlist, max(1, int(np.sqrt(max(1, n))))))
                quantizer = faiss.IndexFlatL2(self.dim)
                new_index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)
                new_index.nprobe = self.nprobe
                try:
                    new_index.cp.niter = getattr(self._index.cp, "niter", 15)
                except Exception:
                    pass
                self._index = new_index

            rng = np.random.default_rng(self._rng_seed)
            if ts == n:
                train_x = xb
            else:
                idx = rng.integers(0, n, size=ts, dtype=np.int64)
                train_x = xb[idx]
            train_x = _as_f32_c(train_x)
            self._index.train(train_x)

        cs = max(1, self.add_chunk_size)
        for i in range(0, xb.shape[0], cs):
            self._index.add(xb[i : i + cs])

        self._ntotal += int(xb.shape[0])

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        xq = _as_f32_c(xq)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim}), got {xq.shape}")
        nq = xq.shape[0]

        if self._index is None:
            if self._fallback_xb is None or self._fallback_xb.shape[0] == 0:
                D = np.full((nq, k), np.inf, dtype=np.float32)
                I = np.full((nq, k), -1, dtype=np.int64)
                return D, I
            xb = self._fallback_xb
            xb = _as_f32_c(xb)
            x2 = (xq * xq).sum(axis=1, keepdims=True)
            b2 = (xb * xb).sum(axis=1, keepdims=True).T
            ip = xq @ xb.T
            dist = x2 + b2 - 2.0 * ip
            kk = min(k, xb.shape[0])
            I = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
            row = np.arange(nq)[:, None]
            dsel = dist[row, I]
            order = np.argsort(dsel, axis=1)
            I = I[row, order]
            D = dsel[row, order].astype(np.float32, copy=False)
            if kk < k:
                I2 = np.full((nq, k), -1, dtype=np.int64)
                D2 = np.full((nq, k), np.inf, dtype=np.float32)
                I2[:, :kk] = I
                D2[:, :kk] = D
                return D2, I2
            return D[:, :k], I[:, :k].astype(np.int64, copy=False)

        faiss.omp_set_num_threads(self.threads)
        self._index.nprobe = self.nprobe
        D, I = self._index.search(xq, k)
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        if D.shape != (nq, k) or I.shape != (nq, k):
            D = np.ascontiguousarray(D.reshape(nq, k), dtype=np.float32)
            I = np.ascontiguousarray(I.reshape(nq, k), dtype=np.int64)
        return D, I