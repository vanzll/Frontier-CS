import numpy as np
import faiss
from typing import Tuple

class Recall80LatencyIndex:
    """
    A FAISS-based index optimized for the Recall80 Latency Tier.

    This implementation uses the HNSW (Hierarchical Navigable Small Worlds)
    algorithm from FAISS. HNSW is a graph-based indexing method that provides
    an excellent trade-off between search speed and accuracy, making it
    well-suited for latency-critical scenarios with a high recall requirement.

    The key parameters, M (number of neighbors in the graph) and efSearch
    (search-time beam width), are tuned based on established benchmarks for
    the SIFT1M dataset. The chosen values aim to comfortably exceed the 80%
    recall@1 threshold while keeping the average query latency well below the
    0.6ms target in a batch processing environment.
    """
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality (e.g., 128 for SIFT1M)
            **kwargs: Optional parameters for HNSW configuration.
                      Supported keys:
                      - 'M': int, number of connections per node (default: 32)
                      - 'ef_construction': int, build-time quality parameter (default: 40)
                      - 'ef_search': int, search-time quality parameter (default: 10)
        """
        self.dim = dim
        
        # Hyperparameters for HNSW, chosen for a strong speed/recall balance.
        # M: Number of bi-directional links for each element in the graph.
        # efConstruction: Size of the dynamic list for nearest neighbors during construction.
        # efSearch: Size of the dynamic list for nearest neighbors during search.
        self.M = kwargs.get('M', 32)
        self.ef_construction = kwargs.get('ef_construction', 40)
        self.ef_search = kwargs.get('ef_search', 10)

        # Initialize the HNSW index. We use METRIC_L2 for Euclidean distance.
        # IndexHNSWFlat stores the full vectors, which is acceptable for SIFT1M's memory footprint.
        self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
        
        # Set the construction and search parameters on the HNSW object.
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        
        # FAISS will automatically use multiple threads if available (OMP_NUM_THREADS).
        # We rely on the evaluation environment's default configuration for parallel execution.

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32
        """
        # With HNSW, adding vectors involves inserting them into the graph.
        # There is no separate training step required as with IVF-based methods.
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
        # The search performance is controlled by `efSearch`, which was set
        # during initialization. FAISS performs the search efficiently in batch.
        # The method returns squared L2 distances, which is acceptable per the
        # problem specification and faster than computing the true L2 distance.
        distances, indices = self.index.search(xq, k)
        return distances, indices