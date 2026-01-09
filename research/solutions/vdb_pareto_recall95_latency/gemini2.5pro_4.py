import numpy as np
import faiss
import os
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality (e.g., 128 for SIFT1M)
            **kwargs: Optional parameters (e.g., M, ef_construction for HNSW)
        """
        self.dim = dim
        
        try:
            # Utilize all available CPU cores for parallel processing
            num_threads = os.cpu_count()
            if num_threads:
                faiss.omp_set_num_threads(num_threads)
        except (AttributeError, NotImplementedError):
            # os.cpu_count() might not be available in all environments
            pass

        # Tuned HNSW parameters for a balance of high recall and low latency on SIFT1M
        self.M = int(kwargs.get('M', 48))
        self.efConstruction = int(kwargs.get('ef_construction', 300))
        self.efSearch = int(kwargs.get('ef_search', 128))

        # IndexHNSWFlat provides high accuracy without compression, suitable for the problem's
        # generous memory and latency budget but high recall requirement.
        self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.efConstruction
        
        # Set the search-time parameter using the modern FAISS API.
        # This controls the trade-off between search speed and accuracy.
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(self.index, "efSearch", self.efSearch)

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32
        """
        # FAISS requires C-contiguous float32 arrays for optimal performance.
        # We convert the input array if it's not in the correct format.
        if not xb.flags['C_CONTIGUOUS'] or xb.dtype != np.float32:
            xb = np.ascontiguousarray(xb, dtype=np.float32)
        
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
        # Ensure query vectors are in the correct format for FAISS.
        if not xq.flags['C_CONTIGUOUS'] or xq.dtype != np.float32:
            xq = np.ascontiguousarray(xq, dtype=np.float32)
        
        # Perform the search. The efSearch parameter was set during initialization.
        # The problem allows L2 or L2-squared distances. METRIC_L2 in FAISS
        # returns squared L2, which is faster as it avoids a sqrt operation.
        distances, indices = self.index.search(xq, k)
        
        return distances, indices