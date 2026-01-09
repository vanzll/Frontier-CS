import numpy as np
import faiss
from typing import Tuple


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the HNSW index for vectors of dimension `dim`.
        """
        self.dim = int(dim)
        self.M = int(kwargs.get("M", 32))
        self.ef_construction = int(kwargs.get("ef_construction", 100))
        self.ef_search = int(kwargs.get("ef_search", 256))

        self._index = faiss.IndexHNSWFlat(self.dim, self.M)
        self._index.metric_type = faiss.METRIC_L2

        hnsw = self._index.hnsw
        hnsw.efConstruction = self.ef_construction
        hnsw.efSearch = self.ef_search

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32
        """
        if xb.ndim != 2 or xb.shape[1] != self.dim:
            raise ValueError(f"xb must have shape (N, {self.dim}), got {xb.shape}")
        xb = np.ascontiguousarray(xb, dtype=np.float32)
        self._index.add(xb)

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
        if xq.ndim != 2 or xq.shape[1] != self.dim:
            raise ValueError(f"xq must have shape (nq, {self.dim}), got {xq.shape}")
        if k <= 0:
            raise ValueError("k must be positive")

        xq = np.ascontiguousarray(xq, dtype=np.float32)
        self._index.hnsw.efSearch = self.ef_search

        distances, indices = self._index.search(xq, k)
        return distances, indices