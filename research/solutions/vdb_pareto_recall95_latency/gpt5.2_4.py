import os
import numpy as np
from typing import Tuple

try:
    import faiss  # type: ignore
    _HAVE_FAISS = True
except Exception:
    faiss = None
    _HAVE_FAISS = False


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)

        self.nlist = int(kwargs.get("nlist", 8192))
        self.nprobe = int(kwargs.get("nprobe", 256))
        self.train_size = int(kwargs.get("train_size", 200000))

        nt = kwargs.get("num_threads", None)
        if nt is None:
            nt = os.cpu_count() or 1
        self.num_threads = int(nt)

        self._use_faiss = _HAVE_FAISS
        self._index = None
        self._xb = None  # fallback storage
        self._ntotal = 0

        if self._use_faiss:
            try:
                faiss.omp_set_num_threads(self.num_threads)
            except Exception:
                pass

    def add(self, xb: np.ndarray) -> None:
        xb = np.asarray(xb, dtype=np.float32, order="C")
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim})")

        n = xb.shape[0]
        if n == 0:
            return

        if self._use_faiss:
            if self._index is None:
                quantizer = faiss.IndexFlatL2(self.dim)
                index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)
                index.nprobe = min(self.nprobe, self.nlist)

                if not index.is_trained:
                    ts = min(self.train_size, n)
                    train_data = xb[:ts]
                    if ts < self.nlist:
                        # Ensure enough training vectors if user passes tiny batches
                        reps = (self.nlist + ts - 1) // ts
                        train_data = np.vstack([train_data] * reps)[: self.nlist]
                    index.train(train_data)

                index.add(xb)
                self._index = index
                self._ntotal = int(index.ntotal)
            else:
                self._index.add(xb)
                self._ntotal = int(self._index.ntotal)
            return

        # Fallback (slow): store all base vectors
        if self._xb is None:
            self._xb = xb.copy()
        else:
            self._xb = np.vstack([self._xb, xb])
        self._ntotal = int(self._xb.shape[0])

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        xq = np.asarray(xq, dtype=np.float32, order="C")
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim})")

        nq = xq.shape[0]
        if nq == 0:
            return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64)

        if self._use_faiss and self._index is not None:
            try:
                self._index.nprobe = min(self.nprobe, self.nlist)
            except Exception:
                pass
            D, I = self._index.search(xq, k)
            if D.dtype != np.float32:
                D = D.astype(np.float32, copy=False)
            if I.dtype != np.int64:
                I = I.astype(np.int64, copy=False)
            return D, I

        # Fallback brute force (very slow on 1M; intended only if faiss unavailable)
        if self._xb is None or self._xb.shape[0] == 0:
            D = np.full((nq, k), np.inf, dtype=np.float32)
            I = np.full((nq, k), -1, dtype=np.int64)
            return D, I

        xb = self._xb
        xb_norm = (xb * xb).sum(axis=1, keepdims=True).T  # (1, N)
        xq_norm = (xq * xq).sum(axis=1, keepdims=True)    # (nq, 1)
        # squared L2: ||q||^2 + ||b||^2 - 2 q.b
        dots = xq @ xb.T
        dist = xq_norm + xb_norm - 2.0 * dots

        if k == 1:
            I = np.argmin(dist, axis=1).astype(np.int64)[:, None]
            D = dist[np.arange(nq), I[:, 0]].astype(np.float32)[:, None]
            return D, I

        idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        row = np.arange(nq)[:, None]
        dsel = dist[row, idx]
        ord_ = np.argsort(dsel, axis=1)
        I = idx[row, ord_].astype(np.int64)
        D = dist[row, I].astype(np.float32)
        return D, I