import numpy as np
import faiss
from typing import Tuple

class Recall95Index:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.
        Optimized for high recall (>=0.95) with low latency on SIFT1M.
        """
        self.dim = dim
        self.index = None
        
        # HNSW Parameters
        # M=32 provides a dense enough graph for high recall navigation
        self.M = 32
        # Higher ef_construction creates a better quality graph structure (only affects build time)
        self.ef_construction = 150
        # ef_search=80 is empirically sufficient for >95% recall on SIFT1M (typically ~98-99%)
        # while keeping latency well below the 7.7ms limit.
        self.ef_search = 80

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        # Initialize index on first addition
        if self.index is None:
            self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
            self.index.hnsw.efConstruction = self.ef_construction
            # Explicitly set number of threads to utilize available 8 vCPUs
            faiss.omp_set_num_threads(8)
        
        # Ensure data is C-contiguous and float32 for FAISS
        xb_c = np.ascontiguousarray(xb, dtype=np.float32)
        self.index.add(xb_c)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        """
        # Set the search depth (efSearch) just before querying
        self.index.hnsw.efSearch = self.ef_search
        
        # Ensure query data is C-contiguous and float32
        xq_c = np.ascontiguousarray(xq, dtype=np.float32)
        
        # Perform search using FAISS
        # Returns distances (D) and indices (I)
        D, I = self.index.search(xq_c, k)
        
        return D, I