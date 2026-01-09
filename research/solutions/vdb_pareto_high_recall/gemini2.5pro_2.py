import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        This implementation uses FAISS's HNSW (Hierarchical Navigable Small Worlds)
        index, which is well-suited for high-recall approximate nearest neighbor search
        on CPU. The parameters are tuned to maximize recall under the given relaxed
        latency constraint of 7.7ms.

        Args:
            dim: Vector dimensionality.
            **kwargs: Optional parameters to override HNSW defaults.
                - M: Number of neighbors per node in the graph (default: 64).
                - ef_construction: Build-time search depth (default: 500).
                - ef_search: Query-time search depth (default: 400).
        """
        self.dim = dim
        
        # Parameters are chosen to build a high-quality graph and perform a
        # thorough search, capitalizing on the generous latency budget.
        # A higher 'M' creates a denser graph.
        m_val = kwargs.get('M', 64)
        # A higher 'ef_construction' leads to a better index at the cost of build time.
        ef_construction = kwargs.get('ef_construction', 500)
        # 'ef_search' is the most critical parameter for the speed/recall tradeoff.
        # This value is set aggressively to achieve high recall, assuming the
        # 2x latency budget allows for it.
        self.ef_search = kwargs.get('ef_search', 400)

        # Initialize the HNSW index with L2 distance metric.
        # IndexHNSWFlat stores the full vectors, ensuring no loss of precision from quantization.
        self.index = faiss.IndexHNSWFlat(self.dim, m_val, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = ef_construction

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32.
        """
        # FAISS requires float32 data.
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
        
        # For HNSW, the index is built as vectors are added.
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.

        Args:
            xq: Query vectors, shape (nq, dim), dtype float32.
            k: Number of nearest neighbors to return.

        Returns:
            (distances, indices): A tuple containing:
                - distances: shape (nq, k), L2 squared distances.
                - indices: shape (nq, k), indices of the nearest neighbors.
        """
        # Set the search-time parameter before querying.
        self.index.hnsw.efSearch = self.ef_search
        
        # Ensure query vectors are float32.
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)
            
        distances, indices = self.index.search(xq, k)
        
        return distances, indices