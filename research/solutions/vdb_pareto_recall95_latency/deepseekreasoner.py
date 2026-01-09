import numpy as np
import faiss
from typing import Tuple

class Recall95LatencyIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize HNSW index optimized for high recall with latency constraints.
        Parameters tuned for SIFT1M with 95% recall target and <7.7ms latency.
        """
        self.dim = dim
        self.index = None
        
        # HNSW parameters optimized for recall >95% and latency <7.7ms
        # Higher M and ef_search for better recall, within latency budget
        self.M = kwargs.get('M', 32)  # Connections per node
        self.ef_construction = kwargs.get('ef_construction', 400)  # Construction time/quality
        self.ef_search = kwargs.get('ef_search', 128)  # Search quality
        
        # Create the index - using inner product for normalized vectors
        # but we'll use L2 distance for SIFT1M
        self.index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        
    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        if self.index.is_trained:
            self.index.add(xb)
        else:
            # HNSW doesn't need training but check anyway
            self.index.train(xb)
            self.index.add(xb)
    
    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors using HNSW.
        Returns distances and indices.
        """
        # Ensure efSearch is set for this search
        self.index.hnsw.efSearch = self.ef_search
        
        # Search
        distances, indices = self.index.search(xq, k)
        
        return distances.astype(np.float32), indices.astype(np.int64)