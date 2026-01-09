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
        self._faiss_available = faiss is not None

        self.num_threads = int(kwargs.get("num_threads", min(8, os.cpu_count() or 1)))

        self.index_type = str(kwargs.get("index_type", "hnsw")).lower()
        self.metric = kwargs.get("metric", "l2")
        if self.metric not in ("l2", "euclidean"):
            raise ValueError("Only L2/Euclidean metric is supported")

        self.M = int(kwargs.get("M", 32))
        self.ef_construction = int(kwargs.get("ef_construction", 200))
        self.ef_search = int(kwargs.get("ef_search", 512))

        self.nlist = int(kwargs.get("nlist", 8192))
        self.nprobe = int(kwargs.get("nprobe", 256))
        self.train_size = int(kwargs.get("train_size", 200000))
        self.ivf_use_hnsw_quantizer = bool(kwargs.get("ivf_use_hnsw_quantizer", False))
        self.quantizer_M = int(kwargs.get("quantizer_M", 32))
        self.quantizer_ef_search = int(kwargs.get("quantizer_ef_search", 64))

        self._ntotal = 0
        self._xb_fallback = None

        if self._faiss_available:
            faiss.omp_set_num_threads(self.num_threads)

            if self.index_type == "ivf":
                if self.ivf_use_hnsw_quantizer:
                    quantizer = faiss.IndexHNSWFlat(self.dim, self.quantizer_M, faiss.METRIC_L2)
                    quantizer.hnsw.efConstruction = max(40, min(200, self.ef_construction))
                    quantizer.hnsw.efSearch = self.quantizer_ef_search
                else:
                    quantizer = faiss.IndexFlatL2(self.dim)

                index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)
                try:
                    index.cp.niter = int(kwargs.get("niter", 20))
                    index.cp.max_points_per_centroid = int(kwargs.get("max_points_per_centroid", 256))
                    index.cp.min_points_per_centroid = int(kwargs.get("min_points_per_centroid", 5))
                except Exception:
                    pass
                index.nprobe = self.nprobe
                self.index = index
            else:
                index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
                index.hnsw.efConstruction = self.ef_construction
                index.hnsw.efSearch = self.ef_search
                try:
                    index.hnsw.search_bounded_queue = int(kwargs.get("search_bounded_queue", 1))
                except Exception:
                    pass
                self.index = index
        else:
            self.index = None

    def add(self, xb: np.ndarray) -> None:
        xb = np.asarray(xb, dtype=np.float32, order="C")
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim})")

        if not self._faiss_available:
            if self._xb_fallback is None:
                self._xb_fallback = xb.copy()
            else:
                self._xb_fallback = np.vstack([self._xb_fallback, xb])
            self._ntotal = int(self._xb_fallback.shape[0])
            return

        faiss.omp_set_num_threads(self.num_threads)

        if self.index_type == "ivf" and not self.index.is_trained:
            n = xb.shape[0]
            ts = min(self.train_size, n)
            rs = np.random.RandomState(123)
            if ts == n:
                xt = xb
            else:
                idx = rs.choice(n, size=ts, replace=False)
                xt = xb[idx]
            self.index.train(np.ascontiguousarray(xt, dtype=np.float32))

        self.index.add(xb)
        self._ntotal = int(self.index.ntotal)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        xq = np.asarray(xq, dtype=np.float32, order="C")
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim})")

        nq = xq.shape[0]
        if self._ntotal <= 0:
            D = np.full((nq, k), np.inf, dtype=np.float32)
            I = np.full((nq, k), -1, dtype=np.int64)
            return D, I

        if not self._faiss_available:
            xb = self._xb_fallback
            xb2 = (xb * xb).sum(axis=1, keepdims=True).T
            xq2 = (xq * xq).sum(axis=1, keepdims=True)
            dists = xq2 + xb2 - 2.0 * (xq @ xb.T)
            dists = np.maximum(dists, 0.0).astype(np.float32, copy=False)
            idx = np.argpartition(dists, kth=min(k - 1, dists.shape[1] - 1), axis=1)[:, :k]
            row = np.arange(nq)[:, None]
            dd = dists[row, idx]
            ord_ = np.argsort(dd, axis=1)
            I = idx[row, ord_].astype(np.int64, copy=False)
            D = dists[row, I].astype(np.float32, copy=False)
            if I.shape[1] < k:
                pad = k - I.shape[1]
                I = np.hstack([I, np.full((nq, pad), -1, dtype=np.int64)])
                D = np.hstack([D, np.full((nq, pad), np.inf, dtype=np.float32)])
            return D, I

        faiss.omp_set_num_threads(self.num_threads)

        if self.index_type == "ivf":
            self.index.nprobe = min(self.nprobe, self.index.nlist)
        else:
            self.index.hnsw.efSearch = max(self.ef_search, k)

        D, I = self.index.search(xq, k)
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I