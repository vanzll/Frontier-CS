import os
import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        # Hyperparameters with sensible defaults targeting recall>=0.80 and latency<0.6ms
        self.nlist = int(kwargs.get("nlist", 32768))
        self.nprobe = int(kwargs.get("nprobe", 10))
        self.use_hnsw_quantizer = bool(kwargs.get("use_hnsw_quantizer", True))
        self.hnsw_m = int(kwargs.get("hnsw_m", 32))
        self.hnsw_efConstruction = int(kwargs.get("hnsw_efConstruction", 200))
        # Let efSearch be a bit larger than nprobe to maintain high-quality coarse selection
        self.hnsw_efSearch = int(kwargs.get("hnsw_efSearch", max(64, self.nprobe * 4)))
        # Training sample size should be sufficiently larger than nlist for good centroids
        self.train_size = int(kwargs.get("train_size", min(250000, self.nlist * 8)))
        self.random_seed = int(kwargs.get("seed", 123))
        self.num_threads = int(kwargs.get("num_threads", min(8, os.cpu_count() or 8)))
        # Internal FAISS index
        self.index = None
        # Set FAISS threads
        try:
            faiss.omp_set_num_threads(self.num_threads)
        except Exception:
            pass

    def _create_quantizer(self):
        if self.use_hnsw_quantizer:
            quantizer = faiss.IndexHNSWFlat(self.dim, self.hnsw_m)
            quantizer.hnsw.efConstruction = self.hnsw_efConstruction
            quantizer.hnsw.efSearch = self.hnsw_efSearch
        else:
            quantizer = faiss.IndexFlatL2(self.dim)
        return quantizer

    def _build_index(self, xb: np.ndarray):
        quantizer = self._create_quantizer()
        index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)
        # nprobe for search
        index.nprobe = self.nprobe
        # Training
        N = xb.shape[0]
        if not index.is_trained:
            rs = np.random.RandomState(self.random_seed)
            if N > self.train_size:
                idx = rs.choice(N, size=self.train_size, replace=False)
                xtrain = xb[idx].astype(np.float32, copy=False)
            else:
                xtrain = xb.astype(np.float32, copy=False)
            index.train(xtrain)
        self.index = index

    def add(self, xb: np.ndarray) -> None:
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
        if self.index is None:
            self._build_index(xb)
        # Ensure HNSW quantizer has efSearch set (training may reset params internally)
        if self.use_hnsw_quantizer and hasattr(self.index, "quantizer"):
            try:
                self.index.quantizer.hnsw.efSearch = self.hnsw_efSearch
            except Exception:
                pass
        # Set nprobe
        self.index.nprobe = self.nprobe
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)
        if self.index is None or self.index.ntotal == 0:
            # No data added; return empty results
            nq = xq.shape[0]
            return np.full((nq, k), np.inf, dtype=np.float32), -np.ones((nq, k), dtype=np.int64)
        # Ensure HNSW quantizer search parameter is set for coarse routing
        if self.use_hnsw_quantizer and hasattr(self.index, "quantizer"):
            try:
                self.index.quantizer.hnsw.efSearch = self.hnsw_efSearch
            except Exception:
                pass
        # Ensure nprobe is set
        self.index.nprobe = self.nprobe
        D, I = self.index.search(xq, k)
        # Ensure correct dtypes
        if D.dtype != np.float32:
            D = D.astype(np.float32)
        if I.dtype != np.int64:
            I = I.astype(np.int64)
        return D, I