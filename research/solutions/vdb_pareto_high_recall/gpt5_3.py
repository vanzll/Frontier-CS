import numpy as np
from typing import Tuple
import faiss

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.M = int(kwargs.get("M", 32))
        self.ef_construction = int(kwargs.get("ef_construction", 200))
        self.ef_search = int(kwargs.get("ef_search", 900))
        num_threads = kwargs.get("num_threads", None)
        if num_threads is not None:
            try:
                faiss.omp_set_num_threads(int(num_threads))
            except Exception:
                pass
        else:
            try:
                max_threads = faiss.omp_get_max_threads()
                if max_threads is not None and max_threads > 0:
                    faiss.omp_set_num_threads(max_threads)
            except Exception:
                pass

        self.index = faiss.IndexHNSWFlat(self.dim, self.M)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search

    def add(self, xb: np.ndarray) -> None:
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32, copy=False)
        xb = np.ascontiguousarray(xb)
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32, copy=False)
        xq = np.ascontiguousarray(xq)
        # Ensure efSearch is set (allow dynamic adjustment if ef_search changed)
        self.index.hnsw.efSearch = self.ef_search
        D, I = self.index.search(xq, int(k))
        return D, I