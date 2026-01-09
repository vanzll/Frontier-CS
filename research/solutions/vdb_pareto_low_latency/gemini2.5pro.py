import numpy as np
import faiss
from typing import Tuple

class VDBLowLatencyIndex:
    """
    A Vector Database index optimized for recall under a strict latency constraint.

    This implementation uses Faiss's HNSW (Hierarchical Navigable Small World)
    graph-based index. HNSW provides an excellent speed/recall trade-off, which
    is critical for this problem.

    The parameters are aggressively tuned to meet the t_max = 2.31ms latency
    constraint on the SIFT1M dataset benchmark.

    Key Parameter Choices:
    - M (connections per node): 48. A higher M creates a more accurate graph,
      improving recall even with a low search budget. This increases memory
      and build time, but both are within the competition limits.
    - efConstruction (build-time search depth): 200. A high value is used to
      build the highest quality graph possible, as build time is not the
      primary constraint.
    - efSearch (query-time search depth): 32. This is a very aggressive (low)
      value, chosen to prioritize speed and ensure the latency target is met.
      The high-quality graph from M and efConstruction helps to compensate
      for the limited search scope, maintaining reasonable recall.
    """

    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality.
            **kwargs: Optional parameters to override defaults.
                - M: HNSW graph connectivity (default: 48)
                - efConstruction: HNSW build-time quality (default: 200)
                - efSearch: HNSW search-time quality/speed trade-off (default: 32)
        """
        self.dim = dim
        
        # Tuned parameters for the Low Latency Tier.
        # These can be overridden by the user via kwargs.
        self.M = kwargs.get("M", 48)
        self.efConstruction = kwargs.get("efConstruction", 200)
        self.efSearch = kwargs.get("efSearch", 32)

        # Initialize the HNSW index. METRIC_L2 computes squared L2 distances,
        # which is acceptable as it preserves rank order.
        self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
        
        # Set the build-time parameter.
        self.index.hnsw.efConstruction = self.efConstruction

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index. Faiss HNSW does not require a separate
        training step.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32.
        """
        # Faiss requires float32 and C-contiguous arrays for optimal performance.
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
        if not xb.flags['C_CONTIGUOUS']:
            xb = np.ascontiguousarray(xb)
        
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
                - indices: shape (nq, k), indices of the nearest neighbors.
        """
        # Set the search-time parameter. This is the crucial knob for balancing
        # speed and accuracy.
        self.index.hnsw.efSearch = self.efSearch
        
        # Ensure query vectors are in the correct format.
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)
        if not xq.flags['C_CONTIGUOUS']:
            xq = np.ascontiguousarray(xq)

        # Perform the search. Faiss handles multi-threading internally
        # to leverage all available CPU cores, which is ideal for batch queries.
        distances, indices = self.index.search(xq, k)
        
        return distances, indices