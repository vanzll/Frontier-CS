import numpy as np
import faiss
from typing import Tuple

# The evaluation environment has 8 vCPUs. We configure Faiss to use them all
# for both index construction and searching to maximize performance.
faiss.omp_set_num_threads(8)

class YourIndexClass:
    """
    A Faiss-based IVFPQ index specifically tuned for the Recall80 Latency Tier.

    The primary goal is to minimize query latency while ensuring recall@1 is at
    least 80%. The latency target is extremely aggressive (< 0.6ms average per
    query in a large batch), necessitating a highly optimized approximate index.

    The chosen architecture is IndexIVFPQ, which combines vector quantization
    (IVF) for partitioning the search space and Product Quantization (PQ) for
    vector compression and fast distance calculations.

    Key Parameter Choices:
    - nlist=2048: Number of Voronoi cells. This value provides a good trade-off
      between the coarse quantization step (finding nearest cells) and the
      fine search step (searching within cells). A moderate number of cells
      keeps the inverted lists reasonably small.
    - m_pq=32: Number of subquantizers for PQ. For a 128-dimensional vector,
      this results in 32 sub-vectors of 4 dimensions each. This high number
      of subquantizers yields a more accurate distance approximation, which is
      critical for achieving the 80% recall target with a very low `nprobe`.
    - nprobe=2: The number of cells to visit during search. This is the most
      influential parameter for the speed-recall trade-off. A value of 2 is
      extremely aggressive, minimizing the number of vectors scanned per query
      to meet the sub-millisecond latency goal. The high accuracy from m_pq=32
      is chosen to compensate for this minimal search scope.
    """

    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality (e.g., 128 for SIFT1M).
            **kwargs: Optional parameters (not used in this tuned implementation).
        """
        if dim <= 0:
            raise ValueError("Dimension must be positive.")

        self.dim = dim
        self.is_trained = False

        nlist = 2048
        m_pq = 32
        nbits = 8

        # The quantizer for IVF is a flat L2 index.
        quantizer = faiss.IndexFlatL2(self.dim)
        
        # We explicitly use METRIC_L2, which computes squared Euclidean distance.
        # This is faster and preserves the nearest neighbor ordering.
        self.index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m_pq, nbits, faiss.METRIC_L2)

        # Set the number of cells to visit during search.
        self.index.nprobe = 2

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index. If the index is not yet trained,
        it will be trained on a random subset of the first batch of vectors.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32.
        """
        if xb.shape[1] != self.dim:
            raise ValueError(f"Input vector dimension {xb.shape[1]} does not match index dimension {self.dim}")

        # Faiss requires C-contiguous float32 arrays.
        xb_float32 = np.ascontiguousarray(xb, dtype=np.float32)

        if not self.is_trained:
            # Training is required for IVF and PQ to learn the data distribution.
            # We train on a subset of the data for efficiency. A common heuristic
            # is 30-50 samples per cell.
            ntrain = min(xb_float32.shape[0], self.index.nlist * 50)
            
            # Use a fixed seed for reproducibility of the random sampling.
            np.random.seed(42)
            
            # Select random vectors for training.
            train_indices = np.random.choice(xb_float32.shape[0], size=ntrain, replace=False)
            
            self.index.train(xb_float32[train_indices])
            self.is_trained = True

        # Add the full dataset to the trained index.
        self.index.add(xb_float32)

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
        if not self.is_trained:
            raise RuntimeError("Index has not been trained or populated. Call add() first.")
        
        if xq.shape[1] != self.dim:
            raise ValueError(f"Query vector dimension {xq.shape[1]} does not match index dimension {self.dim}")

        # Faiss requires C-contiguous float32 arrays for queries as well.
        xq_float32 = np.ascontiguousarray(xq, dtype=np.float32)
        
        distances, indices = self.index.search(xq_float32, k)
        return distances, indices