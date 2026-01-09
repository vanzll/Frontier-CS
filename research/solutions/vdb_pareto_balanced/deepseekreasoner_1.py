import numpy as np
import faiss

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.index = None
        
        # HNSW parameters optimized for recall within latency constraint
        self.M = kwargs.get('M', 24)  # Higher M for better recall
        self.ef_construction = kwargs.get('ef_construction', 200)
        self.ef_search = kwargs.get('ef_search', 80)  # Balanced for recall/latency
        
        # Use multiple threads for batch queries
        faiss.omp_set_num_threads(8)
    
    def add(self, xb: np.ndarray) -> None:
        if self.index is None:
            # Create HNSW index optimized for recall
            self.index = faiss.IndexHNSWFlat(self.dim, self.M)
            self.index.hnsw.efConstruction = self.ef_construction
        
        # Add vectors in batches to avoid memory issues
        batch_size = 100000
        for i in range(0, len(xb), batch_size):
            end = min(i + batch_size, len(xb))
            self.index.add(xb[i:end])
    
    def search(self, xq: np.ndarray, k: int) -> tuple:
        # Set efSearch for query time
        self.index.hnsw.efSearch = self.ef_search
        
        # Search all queries at once (batch optimization)
        distances, indices = self.index.search(xq, k)
        
        return distances.astype(np.float32), indices.astype(np.int64)