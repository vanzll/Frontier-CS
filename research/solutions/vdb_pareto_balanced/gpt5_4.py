import os
import numpy as np
from typing import Tuple

try:
    import faiss
except Exception as e:
    faiss = None


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = int(dim)
        # Parameters (tuned for SIFT1M, recall priority within latency constraint)
        self.nlist = int(kwargs.get("nlist", 16384))
        self.nprobe = int(kwargs.get("nprobe", 128))
        self.use_pq = bool(kwargs.get("use_pq", False))  # default to IVF-Flat; set True to use IVFPQ
        self.pq_m = int(kwargs.get("pq_m", 16))          # only used if use_pq=True
        self.pq_bits = int(kwargs.get("pq_bits", 8))     # only used if use_pq=True

        # HNSW coarse quantizer settings
        self.use_hnsw_coarse = bool(kwargs.get("use_hnsw_coarse", True))
        self.hnsw_m = int(kwargs.get("hnsw_m", 32))
        self.hnsw_ef_search = int(kwargs.get("hnsw_ef_search", max(64, 2 * self.nprobe)))
        self.hnsw_ef_construction = int(kwargs.get("hnsw_ef_construction", 200))

        # Training
        self.train_samples = int(kwargs.get("train_samples", 200000))
        self.random_seed = int(kwargs.get("random_seed", 123))

        # Threads
        self.num_threads = kwargs.get("num_threads", None)
        try:
            if faiss is not None and self.num_threads is not None:
                faiss.omp_set_num_threads(int(self.num_threads))
        except Exception:
            pass

        self._index = None
        self._is_trained = False
        self._ntotal = 0

        # Buffer if add() is called multiple times before training/index creation
        self._buffer = []

    def _build_index(self, xb: np.ndarray):
        assert faiss is not None, "faiss library is required"
        d = self.dim

        if self.use_hnsw_coarse:
            quantizer = faiss.IndexHNSWFlat(d, self.hnsw_m)
            quantizer.hnsw.efConstruction = self.hnsw_ef_construction
            quantizer.hnsw.efSearch = self.hnsw_ef_search
        else:
            quantizer = faiss.IndexFlatL2(d)

        metric = faiss.METRIC_L2
        if self.use_pq:
            # IVFPQ with PQ m x 8 bits
            self._index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.pq_m, self.pq_bits, metric)
        else:
            # IVF-Flat (exact L2 within probed lists)
            self._index = faiss.IndexIVFFlat(quantizer, d, self.nlist, metric)

        # Training
        rs = np.random.RandomState(self.random_seed)
        N = xb.shape[0]
        train_sz = min(self.train_samples, N)
        if train_sz < self.nlist:
            train_sz = min(max(self.nlist * 2, 10000), N)
        perm = rs.permutation(N)[:train_sz]
        xt = np.ascontiguousarray(xb[perm], dtype=np.float32)
        self._index.train(xt)
        self._is_trained = True

        # Set nprobe and quantizer search parameters
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = int(self.nprobe)
        try:
            if self.use_hnsw_coarse:
                # after training, quantizer contains centroids; set efSearch again for safety
                self._index.quantizer.hnsw.efSearch = int(self.hnsw_ef_search)
        except Exception:
            pass

    def add(self, xb: np.ndarray) -> None:
        if xb is None or len(xb) == 0:
            return
        xb = np.asarray(xb, dtype=np.float32, order="C")
        if xb.shape[1] != self.dim:
            raise ValueError("Input vectors must have shape (N, dim) with dim=%d" % self.dim)

        if self._index is None:
            # Build index on first add using this batch's data to train
            self._build_index(xb)

        if not self._is_trained:
            # Should not happen; ensure trained
            self._build_index(xb)

        # Add in chunks to control memory usage
        add_bs = 65536
        n = xb.shape[0]
        for i0 in range(0, n, add_bs):
            i1 = min(i0 + add_bs, n)
            self._index.add(xb[i0:i1])
            self._ntotal += (i1 - i0)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None or not self._is_trained or self._ntotal == 0:
            raise RuntimeError("Index is empty or not trained. Call add() before search().")
        xq = np.asarray(xq, dtype=np.float32, order="C")
        if xq.shape[1] != self.dim:
            raise ValueError("Query vectors must have shape (nq, dim) with dim=%d" % self.dim)
        if k <= 0:
            raise ValueError("k must be >= 1")

        # ensure nprobe and HNSW efSearch are set for search time
        try:
            if hasattr(self._index, "nprobe"):
                self._index.nprobe = int(self.nprobe)
        except Exception:
            pass
        try:
            if self.use_hnsw_coarse:
                self._index.quantizer.hnsw.efSearch = int(self.hnsw_ef_search)
        except Exception:
            pass

        D, I = self._index.search(xq, k)
        # Ensure correct dtypes
        if D is None or I is None:
            # If FAISS returns None (shouldn't), return empty arrays
            nq = xq.shape[0]
            D = np.full((nq, k), np.inf, dtype=np.float32)
            I = np.full((nq, k), -1, dtype=np.int64)
        else:
            if D.dtype != np.float32:
                D = D.astype(np.float32, copy=False)
            if I.dtype != np.int64:
                I = I.astype(np.int64, copy=False)
        return D, I