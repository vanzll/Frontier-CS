import numpy as np
from typing import Tuple
import faiss

class YourIndexClass:
    """
    An implementation of a vector database index using FAISS HNSW,
    optimized for the Recall95 Latency Tier.

    This index is designed to meet a high recall@1 requirement (>= 0.95)
    while minimizing query latency. It uses the Hierarchical Navigable Small Worlds (HNSW)
    graph-based algorithm, which provides an excellent trade-off between search speed
    and accuracy, particularly in high-recall scenarios.

    The key parameters are chosen based on standard benchmarks for the SIFT1M dataset
    to provide a recall comfortably above the 95% gate with very low latency.
    - M=32: Controls graph connectivity. A moderate value balances index size and search performance.
    - efConstruction=400: Ensures a high-quality graph is built, which is crucial for recall.
    - efSearch=32: A carefully chosen search depth to exceed 95% recall while keeping latency minimal.

    The implementation leverages FAISS's multi-threading capabilities to take full
    advantage of the multi-core CPU environment.
    """

    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality (e.g., 128 for SIFT1M)
            **kwargs: Optional parameters for the HNSW index.
                      - M: Number of neighbors for HNSW graph (default: 32)
                      - efConstruction: efConstruction for HNSW build (default: 400)
                      - efSearch: efSearch for HNSW search (default: 32)
                      - num_threads: Number of threads for FAISS to use (default: 8)
        """
        self.dim = dim
        
        # Hyperparameters for HNSW. These defaults are chosen to balance
        # the high recall requirement (>= 0.95) with low latency.
        self.M = kwargs.get("M", 32)
        self.efConstruction = kwargs.get("efConstruction", 400)
        self.efSearch = kwargs.get("efSearch", 32)
        num_threads = kwargs.get("num_threads", 8)

        # Explicitly set the number of threads FAISS uses for parallel execution.
        faiss.omp_set_num_threads(num_threads)

        # Initialize the Faiss Index using HNSWFlat.
        # HNSW is state-of-the-art for high-recall ANN search on CPUs.
        # 'Flat' storage avoids quantization error, helping to meet the recall gate.
        self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
        
        self.index.hnsw.efConstruction = self.efConstruction
        self.index.hnsw.efSearch = self.efSearch

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32
        """
        # Faiss requires the input data to be of type float32.
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
                - distances: shape (nq, k), dtype float32, L2 distances squared
                - indices: shape (nq, k), dtype int64, indices into base vectors
        """
        # Ensure query vectors are float32.
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)

        # Perform the search.
        distances, indices = self.index.search(xq, k)

        # Note: faiss.IndexHNSWFlat returns squared L2 distances by default.
        # This is acceptable per the problem description ("L2 or L2-squared")
        # and is computationally more efficient.
        return distances, indices