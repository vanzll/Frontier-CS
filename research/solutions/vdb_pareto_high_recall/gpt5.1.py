import numpy as np
from typing import Tuple
import faiss


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize an HNSW index optimized for high recall within a relaxed latency budget.
        """
        self.dim = dim

        # HNSW parameters with high-recall oriented defaults
        M = int(kwargs.get("M", 32))  # number of neighbors per node
        ef_construction = int(kwargs.get("ef_construction", 200))
        ef_search = int(kwargs.get("ef_search", 1024))

        # Optionally control number of threads used by Faiss
        n_threads = kwargs.get("n_threads", None)
        if n_threads is not None:
            try:
                faiss.omp_set_num_threads(int(n_threads))
            except AttributeError:
                # Older Faiss versions may not expose omp controls; safe to ignore
                pass

        # Build HNSW-Flat index (L2 metric by default)
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

        self.ef_search = ef_search

    def add(self, xb: np.ndarray) -> None:
        """
        Add base vectors to the HNSW index.
        """
        if xb.dtype != np.float32 or not xb.flags['C_CONTIGUOUS']:
            xb = np.ascontiguousarray(xb, dtype=np.float32)
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search k nearest neighbors using the HNSW index.
        """
        if xq.dtype != np.float32 or not xq.flags['C_CONTIGUOUS']:
            xq = np.ascontiguousarray(xq, dtype=np.float32)

        k = int(k)
        # Ensure efSearch is at least k for stable behavior at larger k
        if k > self.index.hnsw.efSearch:
            new_ef = max(self.ef_search, k)
            self.index.hnsw.efSearch = new_ef
            self.ef_search = new_ef

        D, I = self.index.search(xq, k)

        # Ensure correct dtypes
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)

        return D, I