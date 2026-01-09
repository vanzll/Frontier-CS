import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        This implementation uses Faiss's Hierarchical Navigable Small World (HNSW)
        graph-based index. HNSW is chosen for its excellent performance in high-recall
        scenarios on CPU-only environments.

        The parameters are tuned to prioritize meeting the 95% recall gate first,
        and then minimizing latency. A high-quality graph is built by using
        generous `M` and `efConstruction` values. This allows for a more moderate
        `efSearch` during query time, which reduces latency while still leveraging
        the well-structured graph to achieve high recall.

        Args:
            dim: Vector dimensionality (e.g., 128 for SIFT1M)
            **kwargs: Optional parameters for HNSW index configuration.
                - M: Number of connections per node (default: 64).
                - efConstruction: Build-time quality parameter (default: 400).
                - efSearch: Search-time quality/speed trade-off parameter (default: 128).
        """
        self.dim = dim
        self.is_built = False

        # HNSW parameters. These defaults aim for a balance that should exceed the
        # 95% recall threshold while maintaining a competitive latency.
        self.M = kwargs.get('M', 64)
        self.efConstruction = kwargs.get('efConstruction', 400)
        self.efSearch = kwargs.get('efSearch', 128)

        # Initialize Faiss HNSW index. IndexHNSWFlat stores full vectors, which is
        # ideal for accuracy when memory is not a major constraint (SIFT1M fits easily).
        # The metric is L2 as specified in the problem.
        self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
        
        # Set the construction-time parameter. A higher value results in a better
        # index graph at the cost of longer build time.
        self.index.hnsw.efConstruction = self.efConstruction
        
        # Set the search-time parameter. This is the primary lever for tuning the
        # recall vs. latency trade-off.
        faiss.ParameterSpace().set_index_parameter(self.index, "efSearch", self.efSearch)

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32. Faiss requires float32.
        """
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)

        self.index.add(xb)
        self.is_built = True

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.

        Args:
            xq: Query vectors, shape (nq, dim), dtype float32.
            k: Number of nearest neighbors to return.

        Returns:
            (distances, indices):
                - distances: shape (nq, k), dtype float32, L2-squared distances.
                - indices: shape (nq, k), dtype int64, indices into base vectors.
        """
        if not self.is_built:
            raise RuntimeError("Index has not been built. Please call add() first.")

        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)

        # Faiss's search method is optimized for batch queries and multithreaded,
        # which is ideal for the evaluation environment.
        distances, indices = self.index.search(xq, k)
        
        return distances, indices