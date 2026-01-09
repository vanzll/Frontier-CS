import numpy as np
import faiss
from typing import Tuple

# Set the number of threads for Faiss to leverage the 8 vCPUs in the environment
faiss.omp_set_num_threads(8)

class YourIndexClass:
    """
    A Faiss-based HNSW index optimized for the Low Latency Tier.
    
    This index uses Hierarchical Navigable Small World (HNSW), a graph-based
    ANN algorithm that provides an excellent speed-recall trade-off. The
    parameters are specifically tuned to meet the strict latency constraint
    of 2.31ms while maximizing recall@1 on the SIFT1M dataset.

    Parameter Justification:
    - M=16: A relatively small number of neighbors per node. This creates a
      sparser graph, which is faster to traverse during search, directly
      contributing to lower latency at a small cost to recall.
    - efConstruction=100: A moderate value for index build quality. It provides
      a well-structured graph without an excessive build time. This parameter
      does not affect search latency.
    - efSearch=40: The most critical parameter for the latency/recall trade-off.
      It defines the size of the dynamic candidate list during search. This value
      is aggressively tuned to be low enough to meet the <2.31ms target, while
      being high enough to explore the graph sufficiently for high recall.
    """

    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality (e.g., 128 for SIFT1M).
            **kwargs: Optional parameters are ignored; internally tuned values are used.
        """
        self.dim = dim
        
        # Hardcoded, tuned parameters for the low-latency challenge.
        self.M = 16
        self.ef_construction = 100
        self.ef_search = 40
        
        # Initialize the Faiss HNSW index. IndexHNSWFlat stores full vectors.
        # The default metric is L2, which is required for SIFT1M.
        self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32.
        """
        # Faiss requires float32 data.
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
            
        # Add vectors to the HNSW graph. This is the build step.
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.

        Args:
            xq: Query vectors, shape (nq, dim), dtype float32.
            k: Number of nearest neighbors to return.

        Returns:
            (distances, indices): A tuple of numpy arrays.
                - distances: shape (nq, k), L2-squared distances.
                - indices: shape (nq, k), 0-based indices.
        """
        # If the index is empty, return sentinel values.
        if self.index.ntotal == 0:
            nq = xq.shape[0]
            distances = np.full((nq, k), -1.0, dtype=np.float32)
            indices = np.full((nq, k), -1, dtype=np.int64)
            return distances, indices

        # Ensure query vectors are float32.
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)
            
        # Set the search-time parameter. This is the main knob for the
        # speed vs. recall trade-off.
        self.index.hnsw.efSearch = self.ef_search
        
        # Perform the search. Faiss handles batching and multithreading internally.
        distances, indices = self.index.search(xq, k)
        
        return distances, indices