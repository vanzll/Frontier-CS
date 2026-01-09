import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    """
    A FAISS-based vector index optimized for high recall using HNSW.

    This index is specifically tuned for scenarios where achieving the highest
    possible recall is the primary objective, under a relaxed latency constraint.
    It utilizes the Hierarchical Navigable Small World (HNSW) algorithm, which
    is a state-of-the-art graph-based approach for approximate nearest neighbor
    search, known for its excellent recall-latency trade-off.

    Key design choices for maximizing recall:
    1.  `faiss.IndexHNSWFlat`: This HNSW implementation stores the full,
        uncompressed vectors. During search, it uses the graph to find a set of
        promising candidates and then computes exact L2 distances for this set.
        This avoids the recall ceiling imposed by quantization-based methods (like PQ),
        making it ideal for targeting near-perfect recall.

    2.  Aggressive HNSW Parameters:
        -   `M=64`: A high number of bidirectional links created for each new
            element added to the graph. This creates a denser and more robust
            graph, which significantly improves the chances of finding the true
            nearest neighbors during search. The default is 32.
        -   `efConstruction=400`: A high value for the size of the dynamic list of
            candidates during index construction. This makes the graph building
            process more thorough, resulting in a higher quality index that
            supports more accurate searches. The default is 40.
        -   `efSearch=800`: A very high value for the size of the dynamic list of
            candidates during search. This is the most critical parameter for
            trading latency for recall. By exploring a larger portion of the graph
            for each query, we substantially increase recall. This value is chosen
            to be aggressive, aiming to utilize the provided latency budget fully.

    The class adheres to the specified API and is optimized for batch operations,
    which is crucial for performance in the evaluation environment.
    """

    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality (e.g., 128 for SIFT1M)
            **kwargs: Optional parameters to override HNSW settings:
                      - M: Number of neighbors for HNSW nodes (default: 64).
                      - ef_construction: HNSW construction search depth (default: 400).
                      - ef_search: HNSW query search depth (default: 800).
        """
        self.dim = dim
        self.is_built = False

        # Set HNSW parameters for high recall, allowing overrides via kwargs.
        self.m = int(kwargs.get('M', 64))
        self.ef_construction = int(kwargs.get('ef_construction', 400))
        self.ef_search = int(kwargs.get('ef_search', 800))

        # Initialize FAISS HNSW index using L2 distance.
        self.index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_L2)
        
        # Set the construction-time parameter.
        self.index.hnsw.efConstruction = self.ef_construction

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32
        """
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)

        # For HNSW, adding vectors is what builds the index graph structure.
        self.index.add(xb)
        self.is_built = True

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.

        Args:
            xq: Query vectors, shape (nq, dim), dtype float32
            k: Number of nearest neighbors to return

        Returns:
            (distances, indices):
                - distances: shape (nq, k), dtype float32, L2 distances
                - indices: shape (nq, k), dtype int64, indices into base vectors
        """
        if not self.is_built:
            raise RuntimeError("Index must be built using add() before searching.")

        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)

        # Set the crucial search-time parameter. A high efSearch value is key to
        # achieving high recall, at the cost of higher latency.
        self.index.hnsw.efSearch = self.ef_search

        # FAISS's search method is highly optimized for batch queries.
        distances, indices = self.index.search(xq, k)
        
        return distances, indices