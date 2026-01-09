import numpy as np
import faiss

class HighRecallIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize HNSW index optimized for high recall within 7.7ms latency constraint.
        Uses aggressive parameters for maximum recall while staying within latency budget.
        """
        self.dim = dim
        
        # HNSW parameters optimized for high recall with 2x latency budget
        # Based on SIFT1M (128 dimensions, 1M vectors)
        self.M = 48  # Higher M for better recall (64 was too slow)
        self.ef_construction = 500  # High construction quality
        self.ef_search = kwargs.get('ef_search', 700)  # Aggressive search depth
        
        # Create HNSW index with L2 distance
        self.index = faiss.IndexHNSWFlat(dim, self.M)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        
        # Storage for vectors to maintain indices
        self.vectors_added = 0
        self.xb = None

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        if self.xb is None:
            self.xb = xb.astype(np.float32)
        else:
            self.xb = np.vstack([self.xb, xb.astype(np.float32)])
        
        self.index.add(xb.astype(np.float32))
        self.vectors_added += xb.shape[0]

    def search(self, xq: np.ndarray, k: int) -> tuple:
        """
        Search for k nearest neighbors with aggressive parameters for high recall.
        Uses batch processing for efficiency.
        """
        # Convert to float32 for FAISS
        xq_float32 = xq.astype(np.float32)
        
        # Ensure we have enough neighbors to return
        if self.vectors_added < k:
            k = self.vectors_added
        
        # Search with aggressive efSearch for high recall
        # Using the full 2x latency budget for maximum accuracy
        if k > 0:
            distances, indices = self.index.search(xq_float32, k)
        else:
            # Return empty results if no vectors in index
            nq = xq.shape[0]
            distances = np.empty((nq, 0), dtype=np.float32)
            indices = np.empty((nq, 0), dtype=np.int64)
        
        # Handle case where we need exactly k neighbors but have fewer
        if indices.shape[1] < k:
            nq = xq.shape[0]
            full_indices = np.full((nq, k), -1, dtype=np.int64)
            full_distances = np.full((nq, k), np.inf, dtype=np.float32)
            
            if indices.shape[1] > 0:
                full_indices[:, :indices.shape[1]] = indices
                full_distances[:, :distances.shape[1]] = distances
            
            indices = full_indices
            distances = full_distances
        
        return distances, indices