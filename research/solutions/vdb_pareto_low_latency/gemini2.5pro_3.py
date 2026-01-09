import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        This index uses FAISS's IndexIVFPQ, which is highly optimized for
        speed and memory efficiency on large datasets, making it suitable for
        CPU-only environments with strict latency constraints.

        The parameters are aggressively tuned for speed to meet the low-latency
        requirements of this tier.

        Args:
            dim: Vector dimensionality.
            **kwargs: Optional parameters to override index configuration.
                      Supported: 'nlist', 'm', 'nprobe'.
        """
        self.dim = dim
        self.is_trained = False

        # Parameters tuned for low latency on SIFT1M
        nlist = kwargs.get("nlist", 4096)
        m = kwargs.get("m", 32)
        nbits = 8
        self.nprobe = kwargs.get("nprobe", 5)

        if dim % m != 0:
            raise ValueError(f"Dimension {dim} must be divisible by m {m}")

        # Use a flat L2 index for the coarse quantizer
        quantizer = faiss.IndexFlatL2(dim)
        
        # The main index is IVFPQ using the L2 metric
        self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_L2)
        
        # Set the number of cells to visit at search time (nprobe)
        self.index.nprobe = self.nprobe

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index. If the index is not yet trained,
        it will be trained on this initial batch of vectors.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32.
        """
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)

        if not self.is_trained:
            # Training is required before adding vectors to an IVF index
            self.index.train(xb)
            self.is_trained = True
        
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.

        Args:
            xq: Query vectors, shape (nq, dim), dtype float32.
            k: Number of nearest neighbors to return.

        Returns:
            (distances, indices):
                - distances: shape (nq, k), L2-squared distances.
                - indices: shape (nq, k), indices of the nearest neighbors.
        """
        if not self.is_trained:
            # Handle case where search is called before add
            nq = xq.shape[0]
            return (
                np.full((nq, k), -1, dtype=np.float32),
                np.full((nq, k), -1, dtype=np.int64),
            )

        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)

        # FAISS search is highly optimized for batch queries
        distances, indices = self.index.search(xq, k)
        
        return distances, indices