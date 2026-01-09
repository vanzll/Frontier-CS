import numpy as np
import faiss

class HNSWIndex:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        
        # HNSW parameters optimized for high recall with good latency
        # M=32 provides good connectivity for high recall
        # efConstruction=200 ensures good graph quality
        # efSearch will be set adaptively during search
        M = kwargs.get('M', 32)
        efConstruction = kwargs.get('ef_construction', 200)
        
        # Create HNSW index with inner product metric (will convert to L2)
        self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = efConstruction
        
        # Track vectors count for indexing
        self.vectors_added = 0
        
    def add(self, xb: np.ndarray) -> None:
        if len(xb) == 0:
            return
            
        # Add vectors to index
        self.index.add(xb)
        self.vectors_added += len(xb)
        
    def search(self, xq: np.ndarray, k: int) -> tuple:
        # Adaptive efSearch based on dataset size
        # Higher efSearch for larger k values to maintain high recall
        ef_search = min(200, max(100, k * 20))
        self.index.hnsw.efSearch = ef_search
        
        # Perform search
        distances, indices = self.index.search(xq, k)
        
        return distances.astype(np.float32), indices.astype(np.int64)