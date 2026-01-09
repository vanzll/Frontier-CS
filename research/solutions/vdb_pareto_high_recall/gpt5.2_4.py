import os
import time
from typing import Tuple, Optional

import numpy as np

try:
    import faiss  # type: ignore
    _FAISS_OK = True
except Exception:
    faiss = None
    _FAISS_OK = False


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)

        self.n_threads = int(kwargs.get("n_threads", min(8, (os.cpu_count() or 1))))
        self.M = int(kwargs.get("M", 48))
        self.ef_construction = int(kwargs.get("ef_construction", 200))
        self.ef_search = int(kwargs.get("ef_search", 4096))
        self.max_ef_search = int(kwargs.get("max_ef_search", max(self.ef_search, 4096)))
        self.min_ef_search = int(kwargs.get("min_ef_search", 64))

        self.calibrate = bool(kwargs.get("calibrate", True))
        self.calib_nq = int(kwargs.get("calib_nq", 1000))
        self.calib_target_ms = float(kwargs.get("calib_target_ms", 6.9))
        self.calib_max_iters = int(kwargs.get("calib_max_iters", 10))
        self.calib_seed = int(kwargs.get("calib_seed", 12345))

        self._index = None
        self._sample_queries: Optional[np.ndarray] = None
        self._calibrated = False

        if _FAISS_OK:
            try:
                faiss.omp_set_num_threads(self.n_threads)
            except Exception:
                pass

            try:
                self._index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
            except TypeError:
                self._index = faiss.IndexHNSWFlat(self.dim, self.M)

            self._index.hnsw.efConstruction = self.ef_construction
            self._index.hnsw.efSearch = min(self.max_ef_search, max(self.min_ef_search, self.ef_search))

            try:
                self._index.hnsw.search_bounded_queue = 0
            except Exception:
                pass
        else:
            self._xb = None

    def _ensure_sample_queries(self, xb: np.ndarray) -> None:
        if self._sample_queries is not None:
            return
        nq = min(self.calib_nq, xb.shape[0])
        if nq <= 0:
            return
        rng = np.random.default_rng(self.calib_seed)
        if xb.shape[0] <= nq:
            idx = np.arange(xb.shape[0], dtype=np.int64)
        else:
            idx = rng.choice(xb.shape[0], size=nq, replace=False).astype(np.int64, copy=False)
        self._sample_queries = np.ascontiguousarray(xb[idx], dtype=np.float32)

    def _bench_ms_per_query(self, xq: np.ndarray, k: int = 1) -> float:
        if xq is None or xq.shape[0] == 0:
            return 0.0
        try:
            faiss.omp_set_num_threads(self.n_threads)
        except Exception:
            pass
        _ = self._index.search(xq[: min(16, xq.shape[0])], k)
        t0 = time.perf_counter()
        _ = self._index.search(xq, k)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0 / float(xq.shape[0])

    def _calibrate_ef_search(self) -> None:
        if not _FAISS_OK or self._index is None or self._calibrated or not self.calibrate:
            return
        if self._sample_queries is None or self._sample_queries.shape[0] == 0:
            self._calibrated = True
            return

        xq = self._sample_queries
        max_ef = int(min(self.max_ef_search, max(self.min_ef_search, self.max_ef_search)))
        min_ef = int(max(1, self.min_ef_search))
        target = float(self.calib_target_ms)

        ef_hi = max_ef
        self._index.hnsw.efSearch = ef_hi
        ms_hi = self._bench_ms_per_query(xq, k=1)

        if ms_hi <= target:
            self._calibrated = True
            return

        ef_lo = min_ef
        self._index.hnsw.efSearch = ef_lo
        ms_lo = self._bench_ms_per_query(xq, k=1)

        if ms_lo > target:
            self._index.hnsw.efSearch = ef_lo
            self._calibrated = True
            return

        best_ef = ef_lo
        lo, hi = ef_lo, ef_hi
        iters = 0
        while lo <= hi and iters < self.calib_max_iters:
            iters += 1
            mid = (lo + hi) // 2
            self._index.hnsw.efSearch = int(mid)
            ms_mid = self._bench_ms_per_query(xq, k=1)
            if ms_mid <= target:
                best_ef = mid
                lo = mid + 1
            else:
                hi = mid - 1

        self._index.hnsw.efSearch = int(best_ef)
        self._calibrated = True

    def add(self, xb: np.ndarray) -> None:
        if xb is None or xb.size == 0:
            return
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32, copy=False)
        xb = np.ascontiguousarray(xb)

        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim})")

        if _FAISS_OK and self._index is not None:
            self._ensure_sample_queries(xb)
            try:
                faiss.omp_set_num_threads(self.n_threads)
            except Exception:
                pass

            chunk_size = 200000
            n = xb.shape[0]
            if n <= chunk_size:
                self._index.add(xb)
            else:
                for i in range(0, n, chunk_size):
                    self._index.add(xb[i : i + chunk_size])

            if not self._calibrated:
                self._calibrate_ef_search()
        else:
            if self._xb is None:
                self._xb = xb.copy()
            else:
                self._xb = np.vstack([self._xb, xb])

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if k <= 0:
            nq = 0 if xq is None else int(xq.shape[0])
            return np.empty((nq, 0), dtype=np.float32), np.empty((nq, 0), dtype=np.int64)

        if xq is None or xq.size == 0:
            return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64)

        if xq.dtype != np.float32:
            xq = xq.astype(np.float32, copy=False)
        xq = np.ascontiguousarray(xq)

        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim})")

        if _FAISS_OK and self._index is not None:
            try:
                faiss.omp_set_num_threads(self.n_threads)
            except Exception:
                pass
            D, I = self._index.search(xq, int(k))
            if D.dtype != np.float32:
                D = D.astype(np.float32, copy=False)
            if I.dtype != np.int64:
                I = I.astype(np.int64, copy=False)
            return D, I

        xb = self._xb
        if xb is None or xb.shape[0] == 0:
            nq = xq.shape[0]
            return np.full((nq, k), np.inf, dtype=np.float32), np.full((nq, k), -1, dtype=np.int64)

        N = xb.shape[0]
        if N <= 200000:
            xq_norm = (xq * xq).sum(axis=1, keepdims=True)
            xb_norm = (xb * xb).sum(axis=1)[None, :]
            distances = xq_norm + xb_norm - 2.0 * (xq @ xb.T)
            distances = distances.astype(np.float32, copy=False)

            kk = min(int(k), N)
            idx = np.argpartition(distances, kk - 1, axis=1)[:, :kk]
            row = np.arange(xq.shape[0])[:, None]
            dsel = distances[row, idx]
            ord_ = np.argsort(dsel, axis=1)
            I = idx[row, ord_].astype(np.int64, copy=False)
            D = distances[row, I].astype(np.float32, copy=False)

            if kk < k:
                padI = np.full((xq.shape[0], k - kk), -1, dtype=np.int64)
                padD = np.full((xq.shape[0], k - kk), np.inf, dtype=np.float32)
                I = np.hstack([I, padI])
                D = np.hstack([D, padD])
            return D, I

        raise RuntimeError("FAISS not available and brute-force fallback is too slow for large datasets.")