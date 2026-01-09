import numpy as np
import faiss
import time
from typing import Tuple

class VectorIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.
        Uses HNSW with optimized parameters for 95% recall with low latency.
        """
        self.dim = dim
        
        # HNSW parameters optimized for recall >95% with latency <7.7ms
        self.M = kwargs.get('M', 24)  # Reduced from typical 32 for better speed
        self.ef_construction = kwargs.get('ef_construction', 200)  # Construction parameter
        self.ef_search = kwargs.get('ef_search', 64)  # Search parameter - optimized for latency
        
        # Initialize HNSW index
        self.index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        
        # Store for potential future additions
        self.vectors_added = 0
        
    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        if len(xb) == 0:
            return
            
        # Add vectors in batches to avoid memory issues
        batch_size = 50000
        n_vectors = xb.shape[0]
        
        for i in range(0, n_vectors, batch_size):
            end_idx = min(i + batch_size, n_vectors)
            batch = xb[i:end_idx]
            self.index.add(batch)
            
        self.vectors_added += n_vectors
        
    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.
        Optimized for batch queries with fixed efSearch parameter.
        """
        # Set search parameter
        self.index.hnsw.efSearch = self.ef_search
        
        # Perform search
        distances, indices = self.index.search(xq, k)
        
        return distances, indices