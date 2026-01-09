import numpy as np
import faiss

class HNSWIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize HNSW index optimized for high recall within latency constraint.
        Default parameters tuned for SIFT1M dataset.
        """
        # Default parameters optimized for recall with latency < 5.775ms
        self.M = kwargs.get('M', 32)  # Higher M for better recall
        self.ef_construction = kwargs.get('ef_construction', 400)  # High for better graph
        self.ef_search = kwargs.get('ef_search', 128)  # Balanced for recall/speed
        
        # Create the index
        self.index = faiss.IndexHNSWFlat(dim, self.M)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        
        # Enable parallel search
        self.index.parallel_mode = 1  # Use multiple threads during search
        
        # Store vectors for exact distance computation if needed (though HNSWFlat computes L2)
        self.dim = dim
        self.xb = None
        
    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        self.index.add(xb)
        self.xb = xb  # Store for exact computation if needed
        
    def search(self, xq: np.ndarray, k: int) -> tuple:
        """
        Search for k nearest neighbors using HNSW with high recall parameters.
        Returns (distances, indices) with L2 distances.
        """
        # Ensure we have at least k vectors in the index
        if self.index.ntotal < k:
            k = self.index.ntotal
            
        # Set efSearch for this query batch
        self.index.hnsw.efSearch = self.ef_search
        
        # Perform the search
        distances, indices = self.index.search(xq, k)
        
        return distances.astype(np.float32), indices.astype(np.int64)