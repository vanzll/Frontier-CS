import numpy as np
import faiss
import os
from typing import Tuple

# Configure FAISS to use all available CPU cores for multithreading.
# The evaluation environment specifies 8 vCPUs.
try:
    # This is more reliable on Linux systems with affinity settings
    num_cores = len(os.sched_getaffinity(0))
except AttributeError:
    # Fallback for other systems
    num_cores = os.cpu_count()

if num_cores:
    faiss.omp_set_num_threads(num_cores)


class YourIndexClass:
    """
    An index optimized for high recall using FAISS's HNSW implementation.

    This solution targets the high recall tier by using a high-quality HNSW graph
    and a very thorough search process. The latency budget of 7.7ms is generous,
    allowing us to crank up search parameters to levels that would be too slow
    for a balanced tier.

    Key choices:
    - IndexHNSWFlat: We use the uncompressed "Flat" version of HNSW. This avoids
      any quantization error from methods like PQ, providing the maximum possible
      recall for a given graph structure.
    - High-quality graph (M, efConstruction): A high number of neighbors per
      node (M=64) and a deep construction-time search (efConstruction=512)
      ensure the graph is robust and well-connected.
    - Exhaustive search (efSearch): This is the most critical parameter. We set
      it to a very high value (1024) to explore a large portion of the graph
      during query time, maximizing the chances of finding the true nearest
      neighbor and thus maximizing recall@1.
    """
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality.
            **kwargs: Optional parameters to override HNSW settings.
                      Supported: 'M', 'ef_construction', 'ef_search'.
        """
        self.dim = dim
        self.is_built = False

        # Parameters tuned for maximum recall within the 7.7ms latency budget.
        M = kwargs.get("M", 64)
        ef_construction = kwargs.get("ef_construction", 512)
        ef_search = kwargs.get("ef_search", 1024)

        # Use IndexHNSWFlat for maximum accuracy (no quantization error).
        # METRIC_L2 for Euclidean distance.
        self.index = faiss.IndexHNSWFlat(self.dim, M, faiss.METRIC_L2)
        
        # Set construction and search parameters on the HNSW object.
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32.
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
            A tuple (distances, indices):
                - distances: shape (nq, k), L2-squared distances.
                - indices: shape (nq, k), 0-based indices of the neighbors.
        """
        if not self.is_built:
            raise RuntimeError("Index has not been built. Call add() before search().")

        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)

        # The efSearch parameter is already set on the index object.
        # FAISS search will use the configured number of threads automatically.
        distances, indices = self.index.search(xq, k)
        
        return distances, indices