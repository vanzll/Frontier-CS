import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    """
    An optimized FAISS HNSW index designed to maximize recall@1 for the SIFT1M
    dataset under a strict latency constraint (5.775ms).

    The strategy is to use `faiss.IndexHNSWFlat`, which avoids any form of
    quantization or compression, thereby preserving the original vector data to
    achieve the highest possible recall. The memory footprint of SIFT1M
    (1M * 128 * float32 = ~512MB) is well within the 16GB RAM limit of the
    evaluation environment.

    Parameters are aggressively tuned based on public benchmarks and an
    understanding of the hardware (8 vCPUs):
    - M=48: Creates a dense, high-quality graph structure. A larger M generally
            improves recall for a given search cost.
    - efConstruction=500: Invests heavily in build-time quality. Since the build
                        process has a generous 1-hour timeout, a high
                        efConstruction ensures a better graph, which translates
                        to better search performance (recall and speed).
    - efSearch=1000: This is the critical search-time parameter. It is set to a
                   high value to explore a large part of the graph, maximizing
                   the probability of finding the true nearest neighbor. This
                   value is chosen to push latency close to the 5.775ms limit,
                   as any saved time does not improve the score, while higher
                   recall does. FAISS's multi-threading capabilities will
                   efficiently use all 8 vCPUs for the batch query.
    """

    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality.
            **kwargs: Optional parameters to override tuned defaults for local testing.
        """
        self.dim = dim
        self.is_built = False

        # Tuned parameters
        m_val = kwargs.get('M', 48)
        ef_construction_val = kwargs.get('efConstruction', 500)
        ef_search_val = kwargs.get('efSearch', 1000)

        # Initialize the FAISS index
        self.index = faiss.IndexHNSWFlat(self.dim, m_val, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = ef_construction_val
        self.index.hnsw.efSearch = ef_search_val

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index. The evaluation environment guarantees xb is
        a float32, C-contiguous numpy array.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32.
        """
        self.index.add(xb)
        self.is_built = True

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors. The evaluation
        environment guarantees xq is a float32, C-contiguous numpy array.

        Args:
            xq: Query vectors, shape (nq, dim), dtype float32.
            k: Number of nearest neighbors to return.

        Returns:
            (distances, indices): A tuple containing L2-squared distances and
                                  the indices of the nearest neighbors.
        """
        if not self.is_built:
            raise RuntimeError("Index has not been built. Call add() before searching.")

        distances, indices = self.index.search(xq, k)
        return distances, indices