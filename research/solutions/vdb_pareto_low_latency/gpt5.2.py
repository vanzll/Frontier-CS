import os
from typing import Tuple, Optional

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)

        self.nlist = int(kwargs.get("nlist", 16384))
        self.nprobe = int(kwargs.get("nprobe", 16))

        self.threads = int(kwargs.get("threads", min(8, (os.cpu_count() or 1))))
        if self.threads < 1:
            self.threads = 1

        self.use_hnsw_quantizer = bool(kwargs.get("use_hnsw_quantizer", True))
        self.quantizer_hnsw_m = int(kwargs.get("quantizer_hnsw_m", 32))
        self.quantizer_ef_search = int(kwargs.get("quantizer_ef_search", 64))
        self.quantizer_ef_construction = int(kwargs.get("quantizer_ef_construction", 80))

        self.kmeans_niter = int(kwargs.get("kmeans_niter", 10))
        self.kmeans_max_points_per_centroid = int(kwargs.get("kmeans_max_points_per_centroid", 32))
        self.kmeans_seed = int(kwargs.get("kmeans_seed", 123))
        self.train_size = kwargs.get("train_size", None)
        self.min_train_points = int(kwargs.get("min_train_points", max(self.nlist, 20000)))

        self._index = None
        self._pending = []

        if faiss is not None:
            faiss.omp_set_num_threads(self.threads)

    def _ensure_index(self):
        if self._index is not None:
            return
        if faiss is None:
            raise RuntimeError("faiss is required but not available")

        if self.use_hnsw_quantizer:
            quantizer = faiss.IndexHNSWFlat(self.dim, self.quantizer_hnsw_m)
            quantizer.hnsw.efSearch = self.quantizer_ef_search
            quantizer.hnsw.efConstruction = self.quantizer_ef_construction
        else:
            quantizer = faiss.IndexFlatL2(self.dim)

        index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)
        index.nprobe = self.nprobe
        try:
            index.parallel_mode = 1
        except Exception:
            pass

        try:
            cp = index.cp
            cp.niter = self.kmeans_niter
            cp.max_points_per_centroid = self.kmeans_max_points_per_centroid
            cp.seed = self.kmeans_seed
        except Exception:
            pass

        self._index = index

    def _maybe_train(self):
        if self._index is None:
            return
        if self._index.is_trained:
            return

        total = sum(int(x.shape[0]) for x in self._pending)
        if total < self.min_train_points:
            return

        first = self._pending[0]
        if len(self._pending) == 1:
            xb_all = first
        else:
            xb_all = None

        if self.train_size is None:
            ts = max(200000, self.nlist * 20)
            ts = min(ts, 400000)
            ts = min(ts, total)
        else:
            ts = int(self.train_size)
            ts = max(ts, self.nlist)
            ts = min(ts, total)

        if xb_all is None:
            acc = []
            need = ts
            for arr in self._pending:
                if need <= 0:
                    break
                take = min(need, arr.shape[0])
                if take > 0:
                    acc.append(arr[:take])
                    need -= take
            train_x = np.ascontiguousarray(np.vstack(acc).astype(np.float32, copy=False))
        else:
            train_x = np.ascontiguousarray(xb_all[:ts].astype(np.float32, copy=False))

        self._index.train(train_x)

        for arr in self._pending:
            self._index.add(np.ascontiguousarray(arr.astype(np.float32, copy=False)))
        self._pending.clear()

    def add(self, xb: np.ndarray) -> None:
        self._ensure_index()
        if xb is None:
            return
        xb = np.ascontiguousarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != self.dim or xb.shape[0] == 0:
            return

        if self._index.is_trained:
            self._index.add(xb)
            return

        self._pending.append(xb)
        self._maybe_train()

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = int(k)
        if k <= 0:
            nq = int(xq.shape[0]) if isinstance(xq, np.ndarray) and xq.ndim >= 1 else 0
            return np.empty((nq, 0), dtype=np.float32), np.empty((nq, 0), dtype=np.int64)

        if self._index is None:
            nq = int(xq.shape[0]) if isinstance(xq, np.ndarray) and xq.ndim >= 1 else 0
            D = np.full((nq, k), np.inf, dtype=np.float32)
            I = np.full((nq, k), -1, dtype=np.int64)
            return D, I

        if not self._index.is_trained:
            self._maybe_train()
            if not self._index.is_trained:
                nq = int(xq.shape[0]) if isinstance(xq, np.ndarray) and xq.ndim >= 1 else 0
                D = np.full((nq, k), np.inf, dtype=np.float32)
                I = np.full((nq, k), -1, dtype=np.int64)
                return D, I

        xq = np.ascontiguousarray(xq, dtype=np.float32)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError("xq must have shape (nq, dim)")

        self._index.nprobe = self.nprobe
        D, I = self._index.search(xq, k)
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I