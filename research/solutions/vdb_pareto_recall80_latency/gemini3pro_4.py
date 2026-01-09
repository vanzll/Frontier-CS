import numpy as np
import faiss
from typing import Tuple

class Recall80Index:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.
        Optimized for Recall@1 >= 0.80 with minimal latency (< 0.6ms).
        """
        self.dim = dim
        self.index = None
        
        # HNSW Hyperparameters
        # M=32 provides a good balance between graph connectivity and memory bandwidth.
        self.M = 32
        # High ef_construction ensures a high-quality graph structure, speeding up search.
        # Construction time is not part of the score, so we maximize this.
        self.ef_construction = 500
        # ef_search=20 is tuned for SIFT1M to reliably exceed 80% recall 
        # while keeping latency extremely low (typically < 0.2ms on 8 vCPUs).
        self.ef_search = 20
        
        # Ensure FAISS uses all available cores for batch processing
        faiss.omp_set_num_threads(8)

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index. Uses HNSWFlat for fast approximate search.
        """
        # Lazily initialize the index to handle potential cumulative adds correctly
        if self.index is None:
            self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
            self.index.hnsw.efConstruction = self.ef_construction
        
        # FAISS requires C-contiguous float32 arrays
        xb_cont = np.ascontiguousarray(xb, dtype=np.float32)
        self.index.add(xb_cont)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        """
        # Set the search-time parameter to control the speed-recall trade-off
        self.index.hnsw.efSearch = self.ef_search
        
        # FAISS requires C-contiguous float32 arrays
        xq_cont = np.ascontiguousarray(xq, dtype=np.float32)
        
        # Perform the search
        distances, indices = self.index.search(xq_cont, k)
        return distances, indices