import numpy as np
import faiss

class HNSWHighRecallIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize HNSW index optimized for high recall within 7.7ms latency.
        
        Parameters tuned for SIFT1M dataset with relaxed latency constraint.
        """
        self.dim = dim
        self.M = kwargs.get('M', 64)  # High connectivity for better recall
        self.ef_construction = kwargs.get('ef_construction', 500)  # High construction quality
        self.ef_search = kwargs.get('ef_search', 800)  # Aggressive search for high recall
        
        # Create HNSW index with L2 distance
        self.index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        
        # Store vectors for consistent indexing
        self.vectors = None
        
    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        # Store vectors for distance computation if needed
        if self.vectors is None:
            self.vectors = xb
        else:
            self.vectors = np.vstack([self.vectors, xb])
        
        # Add to FAISS index
        self.index.add(xb)
        
    def search(self, xq: np.ndarray, k: int):
        """
        Search for k nearest neighbors.
        
        Note: FAISS HNSW returns squared L2 distances by default.
        """
        # Set efSearch parameter (can be adjusted per search)
        if hasattr(self.index.hnsw, 'efSearch'):
            # Ensure efSearch is at least k
            effective_ef = max(self.ef_search, k * 10)
            self.index.hnsw.efSearch = effective_ef
        
        # Perform search
        distances, indices = self.index.search(xq, k)
        
        # Convert squared L2 distances to actual L2 if needed
        # distances = np.sqrt(distances)
        # Note: For recall calculation, actual distances are not needed,
        # only the ordering matters. We keep squared distances for speed.
        
        return distances.astype(np.float32), indices.astype(np.int64)