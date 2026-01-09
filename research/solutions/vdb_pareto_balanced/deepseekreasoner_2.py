import numpy as np
import faiss
from typing import Tuple

class BalancedTierIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize HNSW index optimized for recall with latency constraint.
        
        Args:
            dim: Vector dimensionality
            **kwargs: HNSW parameters (M, ef_construction, ef_search)
        """
        self.dim = dim
        
        # HNSW parameters optimized for SIFT1M recall with latency < 5.775ms
        # These values were tuned for maximum recall while meeting latency constraint
        self.M = kwargs.get('M', 24)  # Higher M for better recall
        self.ef_construction = kwargs.get('ef_construction', 200)
        self.ef_search = kwargs.get('ef_search', 128)  # High ef_search for recall
        
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        
        # Track if vectors are added
        self.is_trained = True  # HNSW doesn't require training
        self.ntotal = 0

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        
        Args:
            xb: Base vectors, shape (N, dim), dtype float32
        """
        if self.ntotal == 0:
            self.index.add(xb)
        else:
            # Handle cumulative addition
            self.index.add(xb)
        self.ntotal += xb.shape[0]

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.
        
        Args:
            xq: Query vectors, shape (nq, dim), dtype float32
            k: Number of nearest neighbors to return
            
        Returns:
            (distances, indices): L2 distances and indices of nearest neighbors
        """
        # Ensure k doesn't exceed number of vectors in index
        k = min(k, self.ntotal)
        
        # Search with current ef_search parameter
        distances, indices = self.index.search(xq, k)
        return distances.astype(np.float32), indices.astype(np.int64)