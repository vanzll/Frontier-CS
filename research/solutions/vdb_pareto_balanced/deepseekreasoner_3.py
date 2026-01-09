import numpy as np
import faiss

class BalancedTierIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize HNSW index with parameters optimized for recall@1 under latency constraint.
        Parameters tuned for SIFT1M (128 dimensions, 1M vectors).
        """
        self.dim = dim
        self.built = False
        
        # HNSW parameters optimized for recall under latency constraint
        # M=16 provides good recall with reasonable memory
        # ef_construction=40 ensures high-quality graph construction
        # ef_search=64 provides high recall while staying under latency constraint
        self.M = kwargs.get('M', 16)
        self.ef_construction = kwargs.get('ef_construction', 40)
        self.ef_search = kwargs.get('ef_search', 64)
        
        # Initialize empty index - will be built when vectors are added
        self.index = None
        
    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index. For HNSW, we build the index directly.
        """
        if self.index is None:
            # Create HNSW index with L2 metric
            self.index = faiss.IndexHNSWFlat(self.dim, self.M)
            self.index.hnsw.efConstruction = self.ef_construction
            
        # Add vectors to the index
        if not self.built:
            self.index.add(xb)
            self.built = True
        else:
            # If index already built, we can add more vectors
            # Note: HNSW doesn't support efficient incremental addition, but we'll handle it
            # by adding to existing index (less efficient but works)
            self.index.add(xb)
    
    def search(self, xq: np.ndarray, k: int) -> tuple:
        """
        Search for k nearest neighbors using HNSW with ef_search parameter.
        Returns distances and indices.
        """
        # Set ef_search for query time
        if hasattr(self.index.hnsw, 'efSearch'):
            self.index.hnsw.efSearch = self.ef_search
        
        # Perform search
        distances, indices = self.index.search(xq, k)
        
        return distances.astype(np.float32), indices.astype(np.int64)