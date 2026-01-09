import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality (e.g., 128 for SIFT1M)
            **kwargs: Optional parameters for HNSW.
                      M: Number of connections per node (default: 48)
                      ef_construction: Search budget during build (default: 128)
                      ef_search: Search budget during query (default: 48)
        """
        self.dim = dim
        
        # Parameters selected for low-latency, high-recall performance
        # on a CPU-only environment.
        # M: Controls graph density. Higher M allows for lower efSearch for a given recall.
        # ef_construction: Build-time quality. Higher is better but slower to build.
        # ef_search: Query-time quality/speed trade-off. This is the most critical
        #            parameter for meeting the strict latency constraint.
        self.M = kwargs.get('M', 48)
        self.ef_construction = kwargs.get('ef_construction', 128)
        self.ef_search = kwargs.get('ef_search', 48)

        # faiss.IndexHNSWFlat is a graph-based index that offers excellent
        # performance for high-recall ANN search on CPUs.
        # It stores full vectors, providing the best accuracy.
        # METRIC_L2 computes squared L2 distances, which is fast and valid
        # for nearest neighbor search.
        self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction
        
        self.is_built = False

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32
        """
        if xb.shape[0] == 0:
            return
        
        # Faiss C++ backend requires C-contiguous arrays for performance.
        # The evaluation data is specified as float32, so no type conversion is needed.
        xb_contiguous = np.ascontiguousarray(xb)
        self.index.add(xb_contiguous)
        self.is_built = True

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
        if not self.is_built or xq.shape[0] == 0:
            # Return empty/sentinel values if the index is not built or query is empty
            return (
                np.full((xq.shape[0], k), float('inf'), dtype=np.float32),
                np.full((xq.shape[0], k), -1, dtype=np.int64),
            )
        
        # Set the search-time parameter. This is the crucial knob for balancing
        # speed and recall. A lower value is faster but less accurate.
        # The value is chosen to meet the strict latency constraint.
        self.index.hnsw.efSearch = self.ef_search

        xq_contiguous = np.ascontiguousarray(xq)
        
        # The search is performed in a batch, which is highly efficient on
        # multi-core CPUs as Faiss parallelizes the operation.
        distances, indices = self.index.search(xq_contiguous, k)

        return distances, indices