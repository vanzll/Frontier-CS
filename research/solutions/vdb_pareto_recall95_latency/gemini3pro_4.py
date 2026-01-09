import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.
        
        Strategy:
        - Use HNSW (Hierarchical Navigable Small World) for high recall and low latency.
        - M=32: A standard choice for SIFT1M that balances graph complexity and search speed.
        - efConstruction=200: Ensures a high-quality graph is built (construction time is not scored).
        - efSearch=100: Conservatively tuned to guarantee recall@1 >= 0.95. 
          Benchmarks typically show recall > 0.98 with these settings on SIFT1M, 
          while maintaining latency well below the 7.7ms threshold.
        """
        self.dim = dim
        
        # HNSW Hyperparameters
        self.M = 32
        self.ef_construction = 200
        self.ef_search = 100  # Tuned for safely meeting the 0.95 recall gate
        
        # Create HNSW index with Flat storage (exact distances)
        # We do not use PQ or other compression to maximize recall
        self.index = faiss.IndexHNSWFlat(dim, self.M)
        
        # Configure construction parameter
        self.index.hnsw.efConstruction = self.ef_construction
        
        # Faiss utilizes OpenMP for multithreading automatically on the available vCPUs.

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        # Ensure input is float32 and C-contiguous for Faiss
        if not xb.flags.c_contiguous:
            xb = np.ascontiguousarray(xb, dtype=np.float32)
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
            
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        """
        # Ensure query is float32 and C-contiguous
        if not xq.flags.c_contiguous:
            xq = np.ascontiguousarray(xq, dtype=np.float32)
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)
            
        # Set search parameter just before search to ensure it's applied
        self.index.hnsw.efSearch = self.ef_search
        
        # Perform search
        # D: L2 squared distances, I: Indices
        D, I = self.index.search(xq, k)
        
        return D, I