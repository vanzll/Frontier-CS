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

        self.nlist = int(kwargs.get("nlist", 4096))
        self.nprobe = int(kwargs.get("nprobe", 16))

        self.hnsw_m = int(kwargs.get("hnsw_m", 32))
        self.ef_construction = int(kwargs.get("ef_construction", 80))
        self.ef_search = int(kwargs.get("ef_search", max(64, self.nprobe * 8)))

        self.train_size = int(kwargs.get("train_size", 200000))
        self.niter = int(kwargs.get("niter", 10))

        self.omp_threads = int(kwargs.get("omp_threads", int(os.environ.get("OMP_NUM_THREADS", "8"))))

        self._index = None
        self._ntotal = 0

        if faiss is not None:
            try:
                faiss.omp_set_num_threads(self.omp_threads)
            except Exception:
                pass

    def _ensure_index(self):
        if self._index is not None:
            return
        if faiss is None:
            raise RuntimeError("faiss is required for this solution")

        quantizer = faiss.IndexHNSWFlat(self.dim, self.hnsw_m)
        try:
            quantizer.hnsw.efConstruction = self.ef_construction
            quantizer.hnsw.efSearch = self.ef_search
        except Exception:
            pass

        index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)

        try:
            index.cp.niter = self.niter
            index.cp.verbose = False
        except Exception:
            pass

        try:
            index.parallel_mode = 1
        except Exception:
            pass

        index.nprobe = self.nprobe
        self._index = index

    def add(self, xb: np.ndarray) -> None:
        self._ensure_index()

        xb = np.ascontiguousarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim})")

        if not self._index.is_trained:
            n = xb.shape[0]
            ts = min(self.train_size, n)
            if ts < self.nlist:
                ts = min(n, self.nlist * 2)

            if ts == n:
                xt = xb
            else:
                rng = np.random.default_rng(12345)
                idx = rng.choice(n, size=ts, replace=False)
                xt = xb[idx]

            self._index.train(xt)

        self._index.add(xb)
        self._ntotal += xb.shape[0]

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None or self._ntotal == 0:
            nq = int(xq.shape[0])
            D = np.full((nq, k), np.inf, dtype=np.float32)
            I = np.full((nq, k), -1, dtype=np.int64)
            return D, I

        xq = np.ascontiguousarray(xq, dtype=np.float32)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim})")

        self._index.nprobe = self.nprobe
        try:
            q = self._index.quantizer
            if hasattr(q, "hnsw"):
                q.hnsw.efSearch = self.ef_search
        except Exception:
            pass

        D, I = self._index.search(xq, int(k))
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I