import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    """
    An optimized vector index for the Balanced Tier VDB Design Problem.

    This implementation uses FAISS's HNSW (Hierarchical Navigable Small World)
    graph-based index. HNSW is chosen for its excellent performance at high recall
    regimes, which is the primary optimization goal of this problem.

    The index is specifically tuned for the SIFT1M dataset (1M vectors, 128 dims)
    under the given latency constraint (<= 5.775ms).

    Key architectural choices:
    - Index Type: `faiss.IndexHNSWFlat`. The "Flat" version stores full,
      uncompressed vectors, avoiding quantization errors to maximize recall.
      Given the dataset size (~500MB) and RAM (16GB), memory is not a bottleneck.
    - Metric: L2 (Euclidean distance). `faiss.METRIC_L2` is used, which calculates
      squared L2 distances for efficiency, fulfilling the problem's requirement.

    Parameter Tuning:
    The core of the solution lies in balancing HNSW parameters to maximize
    recall while staying under the latency limit.
    - M (neighbors per node): Set to 48. A higher M creates a denser, more robust
      graph, improving search accuracy at the cost of higher memory and build time.
      48 is a strong value for high-recall scenarios.
    - ef_construction (build-time search depth): Set to 200. This controls
      the index graph quality. A higher value leads to a better graph and
      potentially higher recall, with a longer build time. The 1-hour
      build limit is generous enough for this value.
    - ef_search (query-time search depth): Set to 256. This is the most critical
      parameter for the recall/latency trade-off. It determines how many entry
      points are explored during a search. This value is set aggressively high to
      push recall towards its maximum, leveraging the generous 5.775ms latency
      budget, which is 150% of the baseline's time. This value aims to surpass
      the 0.9914 baseline recall.
    """
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality (e.g., 128 for SIFT1M).
            **kwargs: Optional parameters for HNSW.
        """
        self.dim = dim
        self.M = kwargs.get('M', 48)
        self.ef_construction = kwargs.get('ef_construction', 200)
        self.ef_search = kwargs.get('ef_search', 256)

        # Initialize the FAISS HNSW index with L2 metric.
        self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index. FAISS HNSW builds the graph during this step.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32.
        """
        # FAISS requires C-contiguous float32 arrays for optimal performance.
        if not xb.flags['C_CONTIGUOUS']:
            xb = np.ascontiguousarray(xb, dtype=np.float32)
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)

        # Add vectors to the HNSW graph. This can be called multiple times.
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.

        Args:
            xq: Query vectors, shape (nq, dim), dtype float32.
            k: Number of nearest neighbors to return.

        Returns:
            (distances, indices):
                - distances: shape (nq, k), squared L2 distances.
                - indices: shape (nq, k), 0-based indices into base vectors.
        """
        # Set the crucial search-time parameter. This controls the trade-off
        # between search speed and accuracy.
        faiss.ParameterSpace().set_index_parameter(self.index, "efSearch", self.ef_search)

        # Ensure query vectors are in the correct format for FAISS.
        if not xq.flags['C_CONTIGUOUS']:
            xq = np.ascontiguousarray(xq, dtype=np.float32)
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)

        # Perform the search. The search over a batch of queries is
        # automatically parallelized by FAISS across available CPU cores.
        distances, indices = self.index.search(xq, k)

        return distances, indices