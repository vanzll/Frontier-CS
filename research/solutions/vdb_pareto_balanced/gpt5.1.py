import numpy as np
import faiss
from typing import Tuple


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize an HNSW index for vectors of dimension `dim`.
        Uses FAISS IndexHNSWFlat with L2 metric.
        """
        self.dim = dim

        # Hyperparameters with sensible defaults for high recall under latency constraint
        self.M = int(kwargs.get("M", 32))  # number of neighbors in HNSW graph
        self.ef_construction = int(kwargs.get("ef_construction", 200))
        self.ef_search = int(kwargs.get("ef_search", 256))

        # Configure FAISS threading
        try:
            max_threads = faiss.omp_get_max_threads()
            if max_threads >= 1:
                faiss.omp_set_num_threads(max_threads)
        except Exception:
            pass  # If threading configuration fails, continue with FAISS defaults

        # Build HNSW index (L2 metric by default)
        self.index = faiss.IndexHNSWFlat(self.dim, self.M)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search

    def add(self, xb: np.ndarray) -> None:
        """
        Add base vectors to the index.
        """
        if not isinstance(xb, np.ndarray):
            xb = np.array(xb, dtype="float32")
        if xb.dtype != np.float32:
            xb = xb.astype("float32", copy=False)

        xb = np.ascontiguousarray(xb)
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim}), got {xb.shape}")

        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.
        """
        if not isinstance(xq, np.ndarray):
            xq = np.array(xq, dtype="float32")
        if xq.dtype != np.float32:
            xq = xq.astype("float32", copy=False)

        xq = np.ascontiguousarray(xq)
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim}), got {xq.shape}")

        ntotal = self.index.ntotal
        if ntotal == 0:
            raise RuntimeError("Index is empty. Call add() before search().")

        if k <= 0:
            raise ValueError("k must be positive.")
        if k > ntotal:
            k = ntotal  # Clamp k to number of database vectors

        # Ensure search parameter is set (in case user modified it externally)
        self.index.hnsw.efSearch = self.ef_search

        D, I = self.index.search(xq, k)
        return D, I