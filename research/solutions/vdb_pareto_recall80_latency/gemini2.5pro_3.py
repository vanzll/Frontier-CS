import faiss
import numpy as np
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality (e.g., 128 for SIFT1M)
            **kwargs: Optional parameters to override defaults.
                      Supported: nlist, m, nprobe.
        """
        self.dim = dim
        self.is_trained = False

        # Parameters tuned for SIFT1M to achieve recall@1 >= 0.80
        # with minimal latency on an 8-core CPU.
        # Strategy: IVFPQ index, which combines partitioning (IVF) with
        # vector compression (PQ) for extremely fast search.
        self.nlist = kwargs.get('nlist', 1024)
        self.m = kwargs.get('m', 16) # dim (128) must be divisible by m
        self.nbits = 8  # bits per sub-quantizer code, 8 is standard
        
        # nprobe is the key search-time parameter for the recall/latency trade-off.
        # Based on FAISS SIFT1M benchmarks, nprobe=12 is a strong candidate
        # to exceed 80% recall while being extremely fast on a multi-core CPU.
        self.nprobe = kwargs.get('nprobe', 12)

        # Coarse quantizer for the IVF part of the index.
        quantizer = faiss.IndexFlatL2(self.dim)
        
        # The main index.
        self.index = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.m, self.nbits, faiss.METRIC_L2)
        
        # Leverage the 8 vCPUs in the evaluation environment for parallelism.
        # This is critical for meeting the sub-millisecond latency target.
        faiss.omp_set_num_threads(8)

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index. If the index is not trained, it will be
        trained on the provided data first.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32
        """
        if not self.is_trained:
            # Training is required before vectors can be added to an IVFPQ index.
            # We train on the database vectors themselves for optimal partitioning.
            self.index.train(xb)
            self.is_trained = True
        
        self.index.add(xb)

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
        if not self.is_trained:
            # Safeguard in case search is called before add.
            # The evaluation flow should prevent this.
            nq = xq.shape[0]
            distances = np.full((nq, k), -1.0, dtype=np.float32)
            indices = np.full((nq, k), -1, dtype=np.int64)
            return distances, indices

        # Set the number of partitions (probes) to visit during the search.
        self.index.nprobe = self.nprobe
        
        distances, indices = self.index.search(xq, k)
        
        return distances, indices