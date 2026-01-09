import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    """
    An index for vector similarity search, optimized for high recall under a latency constraint.
    
    This implementation uses the HNSW (Hierarchical Navigable Small World) algorithm 
    from the Faiss library. HNSW is a graph-based approach that provides an excellent
    trade-off between search speed and accuracy (recall), making it suitable for this
    challenge.

    The key design choices are:
    1.  **Algorithm**: `IndexHNSWFlat` is chosen to avoid the recall loss associated with
        vector quantization (like in IVF_PQ). Since the primary goal is maximizing
        recall, storing full vectors is preferred.
    2.  **Parameter Tuning**: The HNSW parameters (`M`, `ef_construction`, `ef_search`)
        are set to aggressive values.
        - `M=64` and `ef_construction=512`: These build-time parameters create a
          high-quality, dense graph, which is the foundation for achieving high recall.
          Build time is a one-off cost and not part of the scored latency.
        - `ef_search=384`: This crucial search-time parameter is tuned to be as high
          as possible to maximize recall, while aiming to stay just under the latency
          limit of 5.775ms. The value is estimated based on public benchmarks and
          the performance characteristics of HNSW, where latency scales roughly with `ef_search`.
    3.  **Hardware Utilization**: The implementation explicitly sets Faiss to use 8
        threads to take full advantage of the 8 vCPUs in the evaluation environment,
        which is critical for meeting the latency target with a high `ef_search` value.
    """
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality.
            **kwargs: Optional parameters to override HNSW defaults.
        """
        # The evaluation environment has 8 vCPUs. We set Faiss to use them all.
        faiss.omp_set_num_threads(8)

        self.dim = dim
        
        # Build-time parameters for HNSW. Higher values create a better index
        # at the cost of build time and memory.
        self.m = kwargs.get('M', 64)
        self.ef_construction = kwargs.get('ef_construction', 512)
        
        # Search-time parameter. This is the main knob for the speed/recall trade-off.
        # This value is chosen to maximize recall while staying within the latency budget.
        self.ef_search = kwargs.get('ef_search', 384)

        # IndexHNSWFlat stores full vectors, ideal for maximizing recall.
        # METRIC_L2 uses squared Euclidean distance, which is monotonic with L2
        # and computationally cheaper.
        self.index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction
        
        self.is_built = False

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index. Can be called multiple times.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32.
        """
        # Faiss requires C-contiguous float32 arrays for optimal performance.
        if not xb.flags['C_CONTIGUOUS']:
            xb = np.ascontiguousarray(xb, dtype=np.float32)
        elif xb.dtype != np.float32:
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
                - distances: shape (nq, k), dtype float32, squared L2 distances.
                - indices: shape (nq, k), dtype int64, indices into base vectors.
        """
        if not self.is_built:
            nq = xq.shape[0]
            # Return empty/sentinel values if the index is not yet built.
            return np.full((nq, k), -1.0, dtype=np.float32), \
                   np.full((nq, k), -1, dtype=np.int64)

        # Ensure query vectors are in the correct format for Faiss.
        if not xq.flags['C_CONTIGUOUS']:
            xq = np.ascontiguousarray(xq, dtype=np.float32)
        elif xq.dtype != np.float32:
            xq = xq.astype(np.float32)

        # Set the search-time `efSearch` parameter. This can be done directly on the
        # index object or via the ParameterSpace API (the modern approach).
        # self.index.hnsw.efSearch = self.ef_search
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(self.index, "efSearch", self.ef_search)

        distances, indices = self.index.search(xq, k)

        return distances, indices