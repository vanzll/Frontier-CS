import numpy as np
import faiss
from typing import Tuple


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the HNSW index for vectors of dimension `dim`.

        Optional kwargs:
            - M: HNSW connectivity (default: 32)
            - ef_construction: HNSW construction parameter (default: 100)
            - ef_search: HNSW search parameter (default: 128)
            - num_threads: number of threads for Faiss (default: Faiss default)
        """
        self.dim = dim

        self.M = int(kwargs.get("M", 32))
        self.ef_construction = int(kwargs.get("ef_construction", 100))
        self.ef_search = int(kwargs.get("ef_search", 128))

        num_threads = kwargs.get("num_threads", None)
        if num_threads is not None:
            try:
                faiss.omp_set_num_threads(int(num_threads))
            except AttributeError:
                pass  # In case Faiss build lacks omp control APIs

        # HNSW with flat (exact) base storage, L2 metric
        self.index = faiss.IndexHNSWFlat(self.dim, self.M)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search

    def add(self, xb: np.ndarray) -> None:
        """
        Add base vectors to the index.

        Args:
            xb: np.ndarray of shape (N, dim), dtype float32
        """
        if xb is None:
            return

        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim}), got {xb.shape}")

        if xb.dtype != np.float32 or not xb.flags["C_CONTIGUOUS"]:
            xb = np.ascontiguousarray(xb, dtype=np.float32)

        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search k nearest neighbors for query vectors.

        Args:
            xq: np.ndarray of shape (nq, dim), dtype float32
            k: number of neighbors to retrieve

        Returns:
            distances: (nq, k) float32
            indices: (nq, k) int64
        """
        if xq is None:
            raise ValueError("xq must not be None")

        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim}), got {xq.shape}")

        if k <= 0:
            raise ValueError("k must be positive")

        nq = xq.shape[0]

        if self.index is None or self.index.ntotal == 0:
            # No data added yet; return empty results
            distances = np.full((nq, k), np.inf, dtype=np.float32)
            indices = np.full((nq, k), -1, dtype=np.int64)
            return distances, indices

        if xq.dtype != np.float32 or not xq.flags["C_CONTIGUOUS"]:
            xq = np.ascontiguousarray(xq, dtype=np.float32)

        # Ensure efSearch is set (in case user modified it externally)
        self.index.hnsw.efSearch = self.ef_search

        D, I = self.index.search(xq, k)
        return D, I