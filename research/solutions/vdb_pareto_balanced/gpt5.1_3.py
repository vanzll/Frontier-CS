import numpy as np
import faiss
import multiprocessing
from typing import Tuple


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the HNSW index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality.
            **kwargs:
                - M: HNSW graph neighbor parameter (default: 32)
                - ef_construction: Construction exploration factor (default: 200)
                - ef_search: Search exploration factor (default: 400)
                - num_threads: Number of threads for FAISS (default: all available)
        """
        self.dim = dim

        # Set FAISS threading
        n_threads = kwargs.get("num_threads", None)
        if n_threads is None:
            try:
                n_threads = multiprocessing.cpu_count()
            except Exception:
                n_threads = 1
        try:
            faiss.omp_set_num_threads(int(n_threads))
        except Exception:
            # If OpenMP is not available, ignore
            pass

        M = int(kwargs.get("M", 32))
        ef_construction = int(kwargs.get("ef_construction", 200))
        ef_search = int(kwargs.get("ef_search", 400))

        # Create HNSW index with L2 metric
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32
        """
        if xb.dtype != np.float32 or not xb.flags["C_CONTIGUOUS"]:
            xb = np.ascontiguousarray(xb.astype(np.float32))
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.

        Args:
            xq: Query vectors, shape (nq, dim), dtype float32
            k: Number of nearest neighbors to return

        Returns:
            (distances, indices):
                - distances: shape (nq, k), dtype float32, L2-squared distances
                - indices: shape (nq, k), dtype int64, indices into base vectors
        """
        if xq.dtype != np.float32 or not xq.flags["C_CONTIGUOUS"]:
            xq = np.ascontiguousarray(xq.astype(np.float32))
        distances, indices = self.index.search(xq, k)
        return distances, indices