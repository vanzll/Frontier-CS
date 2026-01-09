import numpy as np
import faiss

class HighRecallIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize HNSW index optimized for high recall with relaxed latency constraint.
        Parameters tuned for SIFT1M with 2x latency budget (7.7ms).
        """
        self.dim = dim
        
        # HNSW parameters optimized for high recall within 7.7ms
        # Higher M and efSearch for better recall at the cost of higher latency
        self.M = kwargs.get('M', 32)  # Increased from default 16 for better recall
        self.ef_construction = kwargs.get('ef_construction', 400)  # High for quality construction
        self.ef_search = kwargs.get('ef_search', 600)  # High for exhaustive search within time
        
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(dim, self.M)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        
        # For storing vectors and tracking index state
        self.vectors_added = 0
        
    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        self.index.add(xb)
        self.vectors_added += xb.shape[0]
        
    def search(self, xq: np.ndarray, k: int):
        """
        Search for k nearest neighbors.
        Uses HNSW with high ef_search for maximum recall within latency budget.
        """
        # Set efSearch before each search batch
        self.index.hnsw.efSearch = self.ef_search
        
        # Perform search
        distances, indices = self.index.search(xq, k)
        
        return distances.astype(np.float32), indices.astype(np.int64)