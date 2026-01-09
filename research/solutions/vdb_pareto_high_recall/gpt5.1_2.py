import numpy as np
from typing import Tuple
import faiss


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        High-recall index based on FAISS HNSW with Flat (exact) vectors.

        Args:
            dim: dimensionality of vectors
            kwargs:
                - M: HNSW connectivity parameter (default: 32)
                - ef_search: HNSW search parameter (default: 1024)
                - ef_construction: HNSW construction parameter (default: 200)
                - n_threads: number of OpenMP threads for FAISS (default: faiss.omp_get_max_threads())
        """
        self.dim = int(dim)

        self.M = int(kwargs.get("M", 32))
        self.ef_search = int(kwargs.get("ef_search", 1024))
        self.ef_construction = int(kwargs.get("ef_construction", 200))

        # Configure FAISS threading
        if hasattr(faiss, "omp_get_max_threads"):
            default_threads = faiss.omp_get_max_threads()
        else:
            default_threads = 0
        n_threads = int(kwargs.get("n_threads", default_threads or 8))
        if hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(n_threads)

        # Initialize HNSW index with L2 metric (squared L2 distances)
        self.index = faiss.IndexHNSWFlat(self.dim, self.M)
        # Set construction parameter before any additions
        self.index.hnsw.efConstruction = self.ef_construction
        # Default search ef
        self.index.hnsw.efSearch = self.ef_search

    def add(self, xb: np.ndarray) -> None:
        """
        Add base vectors to the index.
        """
        if xb is None:
            return

        xb = np.asarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim}), got {xb.shape}")

        xb = np.ascontiguousarray(xb)
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for k nearest neighbors of query vectors.
        """
        xq = np.asarray(xq, dtype=np.float32)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim}), got {xq.shape}")

        if k <= 0:
            raise ValueError("k must be positive")

        xq = np.ascontiguousarray(xq)

        ntotal = self.index.ntotal
        if ntotal == 0:
            # No data added yet; return empty results
            nq = xq.shape[0]
            D = np.full((nq, k), np.inf, dtype=np.float32)
            I = -np.ones((nq, k), dtype=np.int64)
            return D, I

        if k > ntotal:
            k = ntotal

        # Ensure efSearch is set
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = self.ef_search

        D, I = self.index.search(xq, k)

        # Ensure correct dtypes
        if D.dtype != np.float32:
            D = D.astype(np.float32, copy=False)
        if I.dtype != np.int64:
            I = I.astype(np.int64, copy=False)

        return D, I