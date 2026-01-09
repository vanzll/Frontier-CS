import numpy as np
from typing import Tuple
import faiss

class YourIndexClass:
    """
    A Faiss-based HNSW index optimized for high recall under a latency constraint.

    This implementation uses the Hierarchical Navigable Small World (HNSW) graph-based
    indexing method from Faiss. The parameters (M, efConstruction, efSearch) are
    tuned specifically for the SIFT1M dataset to achieve a recall@1 > 0.9914
    with an average query latency below 5.775ms on an 8-core CPU environment.

    Strategy:
    1.  **Algorithm**: `faiss.IndexHNSWFlat` is chosen for its state-of-the-art
        performance in high-recall scenarios on CPU. It avoids quantization error
        by storing full vectors, which is critical for maximizing recall.
    2.  **Parameter Tuning**: The parameters are aggressively tuned to build a
        high-quality search graph and perform a deep search, pushing recall as
        high as possible while remaining within the latency budget.
        - `M=64`: Creates a dense, high-quality graph.
        - `efConstruction=400`: Ensures the graph is well-connected during build time.
        - `efSearch=128`: Provides a deep search to maximize accuracy, tuned to fit
          within the latency constraint.
    3.  **Parallelization**: The index is configured to use all 8 vCPUs available
        in the evaluation environment via `faiss.omp_set_num_threads(8)`. This is
        essential for minimizing batch query latency.
    4.  **Metric**: `faiss.METRIC_L2` is used, which computes L2-squared distances.
        This is faster than full L2 distance and yields the same neighbor ranking.
    """

    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality (e.g., 128 for SIFT1M)
            **kwargs: Optional parameters (not used; parameters are hard-tuned)
        """
        self.dim = dim
        
        # Set Faiss to use all 8 available CPU cores, as per evaluation environment.
        # This is critical for achieving low latency on batch queries.
        faiss.omp_set_num_threads(8)
        
        # Tuned HNSW parameters for SIFT1M to maximize recall@1 while staying
        # under the 5.775ms latency constraint.
        M = 64              # Number of neighbors per node in the graph
        efConstruction = 400 # Search depth during index construction
        efSearch = 128       # Search depth during search
        
        # Initialize the HNSW index with the L2 (squared) distance metric.
        self.index = faiss.IndexHNSWFlat(self.dim, M, faiss.METRIC_L2)
        
        # Set construction and search-time parameters.
        self.index.hnsw.efConstruction = efConstruction
        self.index.hnsw.efSearch = efSearch

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32.
        """
        # The evaluation data is expected to be float32, as required by Faiss.
        # HNSW does not require a separate training step.
        self.index.add(xb)

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
        # Faiss performs a batched search, which is highly optimized for multi-core CPUs.
        # The efSearch parameter was set during initialization.
        distances, indices = self.index.search(xq, k)
        return distances, indices