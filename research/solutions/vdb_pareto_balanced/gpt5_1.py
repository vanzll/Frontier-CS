import numpy as np
from typing import Tuple, Optional
import faiss
import os
import threading

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.nlist = int(kwargs.get("nlist", 8192))
        self.nprobe = int(kwargs.get("nprobe", 96))
        self.train_size = int(kwargs.get("train_size", 300000))
        self.use_hnsw_quantizer = bool(kwargs.get("use_hnsw_quantizer", False))
        self.hnsw_m = int(kwargs.get("hnsw_m", 32))
        self.metric = faiss.METRIC_L2
        self.index: Optional[faiss.Index] = None
        self._is_trained = False
        self._lock = threading.Lock()
        # Respect existing threading settings; if none, set to available CPUs
        try:
            max_threads = faiss.omp_get_max_threads()
            if max_threads <= 0:
                faiss.omp_set_num_threads(max(1, os.cpu_count() or 8))
        except Exception:
            pass

    def _build_index(self):
        if self.use_hnsw_quantizer:
            quantizer = faiss.IndexHNSWFlat(self.dim, self.hnsw_m)
            quantizer.hnsw.efSearch = max(self.hnsw_m * 2, 64)
            quantizer.hnsw.efConstruction = max(self.hnsw_m * 4, 200)
        else:
            quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, self.metric)
        index.nprobe = self.nprobe
        self.index = index

    def add(self, xb: np.ndarray) -> None:
        if not isinstance(xb, np.ndarray):
            xb = np.asarray(xb, dtype=np.float32)
        xb = xb.astype(np.float32, copy=False)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError("xb must have shape (N, dim) with dim matching initialization")
        with self._lock:
            if self.index is None:
                self._build_index()
            if not self._is_trained:
                N = xb.shape[0]
                if N <= self.train_size:
                    train_data = xb
                else:
                    rs = np.random.RandomState(12345)
                    idx = rs.choice(N, size=self.train_size, replace=False)
                    train_data = xb[idx]
                self.index.train(train_data)
                self._is_trained = True
            self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(xq, np.ndarray):
            xq = np.asarray(xq, dtype=np.float32)
        xq = xq.astype(np.float32, copy=False)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError("xq must have shape (nq, dim) with dim matching initialization")
        if self.index is None or not self._is_trained:
            raise RuntimeError("Index not built or trained. Call add() with data first.")
        self.index.nprobe = self.nprobe
        D, I = self.index.search(xq, k)
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)
        return D, I