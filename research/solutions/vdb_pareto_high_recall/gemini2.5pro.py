import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality (e.g., 128 for SIFT1M)
            **kwargs: Optional parameters for HNSW index construction.
                      - M: Number of connections per node (default: 64)
                      - ef_construction: Build-time quality parameter (default: 400)
                      - ef_search: Search-time quality parameter (default: 800)
                      - num_threads: Number of CPU threads to use (default: 8)
        """
        self.dim = dim
        
        num_threads = kwargs.get("num_threads", 8)
        faiss.omp_set_num_threads(num_threads)

        # Parameters tuned for high recall, leveraging the generous latency budget.
        # A high M and ef_construction create a high-quality graph.
        self.m = kwargs.get("M", 64)
        self.ef_construction = kwargs.get("ef_construction", 400)
        # A very high ef_search performs a more thorough search for higher recall.
        self.ef_search = kwargs.get("ef_search", 800)

        # Use IndexHNSWFlat for maximum accuracy (no compression).
        # Memory is not a constraint for SIFT1M (1M * 128 * 4B ≈ 512MB).
        self.index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_L2)
        
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efConstruction = self.ef_construction
        
        self.is_built = False

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index. For HNSW, this builds the graph.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32
        """
        # Faiss requires float32.
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
        
        self.index.add(xb)
        self.is_built = True

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.

        Args:
            xq: Query vectors, shape (nq, dim), dtype float32
            k: Number of nearest neighbors to return

        Returns:
            (distances, indices):
                - distances: shape (nq, k), dtype float32, L2 distances
                - indices: shape (nq, k), dtype int64, indices into base vectors
        """
        if not self.is_built:
            raise RuntimeError("Index has not been built. Call add() first.")
        
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = self.ef_search

        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)

        distances, indices = self.index.search(xq, k)
        
        return distances, indices