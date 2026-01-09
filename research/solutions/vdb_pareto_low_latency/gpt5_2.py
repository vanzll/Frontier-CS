import numpy as np
from typing import Tuple
import faiss
import os
import threading

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim

        # Parameters
        self.nlist = int(kwargs.get("nlist", 16384))            # number of IVF lists
        self.pq_m = int(kwargs.get("pq_m", 16))                 # number of PQ subquantizers
        self.pq_nbits = int(kwargs.get("pq_nbits", 8))          # bits per subquantizer
        self.nprobe = int(kwargs.get("nprobe", 3))              # number of probed lists at search
        self.use_opq = bool(kwargs.get("use_opq", True))        # apply OPQ
        self.opq_m = int(kwargs.get("opq_m", self.pq_m))        # OPQ rotation size
        self.train_size = int(kwargs.get("train_size", 250000)) # number of vectors for training

        # HNSW coarse quantizer params
        self.hnsw_m = int(kwargs.get("hnsw_m", 32))
        self.hnsw_ef_search = int(kwargs.get("hnsw_ef_search", max(64, self.nprobe * 40)))
        self.hnsw_ef_construction = int(kwargs.get("hnsw_ef_construction", 200))

        # Threads
        try:
            nt = int(kwargs.get("num_threads", os.cpu_count() or 8))
            faiss.omp_set_num_threads(max(1, nt))
        except Exception:
            pass

        self.index = None
        self.ivf = None
        self.quantizer = None
        self._is_trained = False
        self._xb_count = 0
        self._lock = threading.Lock()

    def _build_index(self):
        # Build HNSW quantizer for IVF
        quantizer = faiss.IndexHNSWFlat(self.dim, self.hnsw_m)
        quantizer.hnsw.efSearch = self.hnsw_ef_search
        quantizer.hnsw.efConstruction = self.hnsw_ef_construction

        # Build IVFPQ
        ivfpq = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.pq_m, self.pq_nbits, faiss.METRIC_L2)
        ivfpq.nprobe = self.nprobe

        # Prefer using precomputed tables for faster ADC if available
        try:
            ivfpq.use_precomputed_table = 1  # may not exist in all versions
        except Exception:
            pass

        # Wrap with OPQ transform if enabled
        if self.use_opq:
            opq = faiss.OPQMatrix(self.dim, self.opq_m)
            # Speed up training a bit while maintaining quality
            opq.niter = 20
            opq.nb_extra_pass = 1
            index = faiss.IndexPreTransform(opq, ivfpq)
        else:
            index = ivfpq

        self.index = index
        self.ivf = ivfpq
        self.quantizer = quantizer

    def _train_if_needed(self, xb: np.ndarray):
        if self.index is None:
            self._build_index()

        if not self.index.is_trained:
            n_train = min(self.train_size, xb.shape[0])
            if n_train < max(1000, self.nlist):
                n_train = min(xb.shape[0], max(1000, self.nlist))
            # Randomly sample training set without replacement
            if xb.shape[0] == n_train:
                train_data = xb
            else:
                rng = np.random.default_rng(12345)
                idx = rng.choice(xb.shape[0], size=n_train, replace=False)
                train_data = xb[idx]
            self.index.train(train_data)

            # Ensure HNSW quantizer efSearch is set for query-time
            try:
                self.quantizer.hnsw.efSearch = self.hnsw_ef_search
            except Exception:
                pass

            # Set IVF nprobe
            try:
                self.ivf.nprobe = self.nprobe
            except Exception:
                pass

            self._is_trained = True

    def add(self, xb: np.ndarray) -> None:
        xb = np.ascontiguousarray(xb.astype(np.float32))
        with self._lock:
            if self.index is None or not self._is_trained:
                self._train_if_needed(xb)
            self.index.add(xb)
            self._xb_count += xb.shape[0]

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None or self._xb_count == 0:
            nq = xq.shape[0]
            return np.full((nq, k), np.inf, dtype=np.float32), np.full((nq, k), -1, dtype=np.int64)

        xq = np.ascontiguousarray(xq.astype(np.float32))

        # Ensure search params are set
        try:
            self.ivf.nprobe = self.nprobe
        except Exception:
            pass
        try:
            self.quantizer.hnsw.efSearch = self.hnsw_ef_search
        except Exception:
            pass

        D, I = self.index.search(xq, k)
        return D, I