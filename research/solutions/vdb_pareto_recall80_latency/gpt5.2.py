import os
from typing import Tuple, Optional

import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)

        self.nlist = int(kwargs.get("nlist", 4096))
        self.nprobe = int(kwargs.get("nprobe", 16))

        self.train_size = int(kwargs.get("train_size", 200000))
        self.kmeans_niter = int(kwargs.get("kmeans_niter", 10))
        self.kmeans_nredo = int(kwargs.get("kmeans_nredo", 1))

        self.target_list_size = int(kwargs.get("target_list_size", 128))

        self.n_threads = int(kwargs.get("n_threads", min(8, (os.cpu_count() or 1))))

        self._index = None
        self._ntotal = 0

        if faiss is not None:
            try:
                faiss.omp_set_num_threads(self.n_threads)
            except Exception:
                pass

    def _as_f32_c(self, x: np.ndarray) -> np.ndarray:
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        return x

    def _choose_nlist(self, n: int) -> int:
        if n <= 0:
            return 1
        nlist_eff = max(1, min(self.nlist, max(1, n // max(1, self.target_list_size))))
        # round to power-of-two-ish for speed/stability
        p = 1
        while p < nlist_eff:
            p <<= 1
        nlist_eff = p
        if nlist_eff > self.nlist:
            nlist_eff >>= 1
        nlist_eff = max(1, min(self.nlist, nlist_eff))
        if nlist_eff > n:
            nlist_eff = max(1, min(self.nlist, n))
        return nlist_eff

    def _train_sample(self, xb: np.ndarray, nlist_eff: int) -> np.ndarray:
        n = xb.shape[0]
        ts = max(self.train_size, nlist_eff * 40)
        ts = min(n, ts)
        if ts <= 0:
            return xb[:0].copy()
        step = max(1, n // ts)
        x = xb[::step][:ts]
        return np.ascontiguousarray(x, dtype=np.float32)

    def _build(self, xb: np.ndarray) -> None:
        if faiss is None:
            raise RuntimeError("faiss is required but not available")

        n = int(xb.shape[0])
        nlist_eff = self._choose_nlist(n)
        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, nlist_eff, faiss.METRIC_L2)

        if hasattr(index, "cp"):
            try:
                index.cp.niter = self.kmeans_niter
            except Exception:
                pass
            try:
                index.cp.nredo = self.kmeans_nredo
            except Exception:
                pass
            try:
                index.cp.verbose = False
            except Exception:
                pass

        xtrain = self._train_sample(xb, nlist_eff)
        if xtrain.shape[0] == 0:
            # Degenerate case; still mark trained so add() won't crash
            index.is_trained = True
        else:
            index.train(xtrain)

        index.nprobe = max(1, min(int(self.nprobe), int(index.nlist)))
        self._index = index

    def add(self, xb: np.ndarray) -> None:
        xb = self._as_f32_c(xb)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim})")

        if self._index is None:
            self._build(xb)

        self._index.add(xb)
        self._ntotal += int(xb.shape[0])

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None or self._ntotal == 0:
            nq = int(xq.shape[0]) if isinstance(xq, np.ndarray) and xq.ndim >= 1 else 0
            kk = int(k)
            D = np.full((nq, kk), np.inf, dtype=np.float32)
            I = np.full((nq, kk), -1, dtype=np.int64)
            return D, I

        xq = self._as_f32_c(xq)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim})")

        k = int(k)
        if k <= 0:
            nq = xq.shape[0]
            return np.empty((nq, 0), dtype=np.float32), np.empty((nq, 0), dtype=np.int64)

        if faiss is not None:
            try:
                faiss.omp_set_num_threads(self.n_threads)
            except Exception:
                pass

        D, I = self._index.search(xq, k)

        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)

        # Ensure exact k outputs even for tiny databases
        nq = xq.shape[0]
        if I.shape != (nq, k):
            D2 = np.full((nq, k), np.inf, dtype=np.float32)
            I2 = np.full((nq, k), -1, dtype=np.int64)
            kk = min(k, I.shape[1])
            D2[:, :kk] = D[:, :kk]
            I2[:, :kk] = I[:, :kk]
            return D2, I2

        return D, I