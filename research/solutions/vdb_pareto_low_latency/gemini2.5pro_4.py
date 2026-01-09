import numpy as np
from typing import Tuple
import faiss
import os

class YourIndexClass:
    """
    A FAISS-based vector index optimized for high recall under a strict 
    low-latency constraint.

    This implementation uses an Inverted File with Product Quantization (IVFPQ)
    index. The key to meeting the latency target is to use a fine-grained
    partitioning of the vector space (a high `nlist`) and then searching only
    a very small number of partitions at query time (a low `nprobe`). This
    aggressively prunes the search space, enabling very fast queries at the
    cost of some recall.
    
    Hyperparameters (`nlist`, `m`, `nprobe`) are tuned for the SIFT1M dataset 
    on a multi-core CPU environment, aiming for a query latency well below
    the 2.31ms threshold while maximizing recall@1.
    """

    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality (e.g., 128 for SIFT1M).
            **kwargs: Optional parameters for tuning the index.
                - nlist (int): Number of IVF cells/partitions. Default: 4096.
                - m (int): Number of subquantizers for PQ. Default: 16.
                - nprobe (int): Number of IVF cells to probe at search time. Default: 3.
        """
        self.dim = dim

        # --- Hyperparameter Tuning ---
        # These parameters are critical for the speed-recall trade-off.
        # They are chosen to be aggressive on speed for the low-latency tier.
        
        # nlist: Number of Voronoi cells. A higher nlist leads to a finer
        # partitioning, which can speed up search if nprobe is small.
        self.nlist = kwargs.get("nlist", 4096)
        
        # m: Number of sub-quantizers for Product Quantization (PQ).
        # The dimension `dim` must be a multiple of `m`. For SIFT1M, 128 % 16 == 0.
        self.m = kwargs.get("m", 16)
        
        # nbits: Number of bits per sub-quantizer code. 8 is standard.
        self.nbits = 8
        
        # nprobe: The most important search-time parameter. It determines how many
        # IVF cells to visit. A very low value is required to meet the latency target.
        self.nprobe = kwargs.get("nprobe", 3)

        # --- FAISS Index Setup ---
        
        # Configure FAISS to use multiple threads for performance.
        try:
            # Use sched_getaffinity on Linux for accurate core count in containers.
            n_cpu = len(os.sched_getaffinity(0))
        except AttributeError:
            # Fallback for other OSes or older Python versions.
            n_cpu = os.cpu_count() or 8
        faiss.omp_set_num_threads(n_cpu)

        # The quantizer is a coarse index used to assign vectors to IVF cells.
        # A simple flat index is sufficient and fast for this purpose.
        quantizer = faiss.IndexFlatL2(self.dim)
        
        # The main index combines IVF for partitioning and PQ for compression.
        # METRIC_L2 specifies Euclidean distance, as required for SIFT1M.
        self.index = faiss.IndexIVFPQ(
            quantizer, self.dim, self.nlist, self.m, self.nbits, faiss.METRIC_L2
        )

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index. This includes training the index if it's the
        first time vectors are being added.
        
        Args:
            xb: Base vectors, shape (N, dim), dtype float32.
        """
        if not self.index.is_trained:
            # Training is required before adding vectors to an IVF index.
            # We train it on a representative subset of the data for efficiency.
            # A common heuristic is to use 30-50 samples per centroid (nlist).
            n_train_samples = self.nlist * 40
            if xb.shape[0] > n_train_samples:
                # Randomly sample a subset for faster training.
                indices = np.random.permutation(xb.shape[0])[:n_train_samples]
                xt = xb[indices].astype('float32')
            else:
                xt = xb
            
            self.index.train(xt)

        # Add the vectors to the trained index.
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.

        Args:
            xq: Query vectors, shape (nq, dim), dtype float32.
            k: Number of nearest neighbors to return.

        Returns:
            A tuple (distances, indices):
                - distances: shape (nq, k), L2 distances.
                - indices: shape (nq, k), indices of the nearest neighbors.
        """
        if self.index.ntotal == 0:
            # Return empty/default results if the index is empty.
            distances = np.full((xq.shape[0], k), -1.0, dtype=np.float32)
            indices = np.full((xq.shape[0], k), -1, dtype=np.int64)
            return distances, indices

        # Set the number of cells to visit. This is the critical parameter
        # for balancing speed and accuracy.
        self.index.nprobe = self.nprobe
        
        distances, indices = self.index.search(xq, k)
        
        return distances, indices