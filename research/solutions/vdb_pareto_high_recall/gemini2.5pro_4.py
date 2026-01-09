import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    """
    An index optimized for high recall using Faiss's HNSW implementation.

    This implementation targets the high-recall tier by leveraging the relaxed
    latency budget (7.7ms) to perform a more thorough search.

    Key design choices:
    - Algorithm: HNSW (Hierarchical Navigable Small Worlds) is chosen for its
      state-of-the-art performance in high-recall ANN search on CPUs.
    - Index Type: `faiss.IndexHNSWFlat` is used. This avoids any form of vector
      compression (like PQ), which would introduce quantization errors and
      limit the maximum achievable recall. The SIFT1M dataset (1M * 128 * 4B
      ~= 512MB) fits comfortably in memory.
    - Metric: L2 (Euclidean distance) is specified as `faiss.METRIC_L2`.
    - Parameters:
      - M=64: A high number of neighbors per node in the HNSW graph. This
        creates a dense, high-quality graph structure, which is crucial for
        navigating to the true nearest neighbors. It increases memory usage and
        build time, which are acceptable trade-offs for higher recall.
      - ef_construction=400: A large search beam width during index construction.
        This ensures that when vectors are inserted, the best possible
        connections are made, improving the overall quality and searchability
        of the graph.
      - ef_search=640: A high search beam width at query time. This is the
        primary parameter for tuning the recall/latency trade-off. This
        aggressive value instructs the search algorithm to explore a large
        number of nodes, significantly increasing the probability of finding
        the true nearest neighbor, while aiming to stay within the 7.7ms
        latency constraint on the 8-core CPU environment.
    """
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality (e.g., 128 for SIFT1M)
            **kwargs: Optional parameters for HNSW.
                      - M: Number of connections per layer (default: 64)
                      - ef_construction: Build-time search depth (default: 400)
                      - ef_search: Query-time search depth (default: 640)
        """
        self.dim = dim
        self.m = kwargs.get("M", 64)
        self.ef_construction = kwargs.get("ef_construction", 400)
        self.ef_search = kwargs.get("ef_search", 640)

        self.index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction
        # Faiss will use all available cores by default, which is optimal.

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32

        Notes:
            - Can be called multiple times (cumulative)
            - Must handle large N (e.g., 1,000,000 vectors)
        """
        # Faiss HNSW requires float32 input.
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)

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
        # Set the search-time parameter for the recall/latency trade-off.
        self.index.hnsw.efSearch = self.ef_search

        # Ensure query vectors are float32.
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)

        # Perform the search and return results.
        distances, indices = self.index.search(xq, k)

        return distances, indices