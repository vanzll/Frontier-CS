import numpy as np
import faiss
from typing import Tuple

class HNSWHighRecallIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize HNSW index optimized for high recall within relaxed latency constraint.
        Using parameters tuned for SIFT1M dataset with 2x latency budget.
        """
        self.dim = dim
        
        # HNSW parameters optimized for high recall with 7.7ms latency budget
        self.M = kwargs.get('M', 64)  # High connectivity for better recall
        self.ef_construction = kwargs.get('ef_construction', 200)  # High construction quality
        self.ef_search = kwargs.get('ef_search', 800)  # High search depth for high recall
        
        # Create HNSW index with L2 distance
        self.index = faiss.IndexHNSWFlat(dim, self.M)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        
        # Set number of threads for batch processing optimization
        self.threads = kwargs.get('threads', 8)
        faiss.omp_set_num_threads(self.threads)

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.
        Optimized for batch processing with high recall.
        """
        # Adjust efSearch for optimal recall-latency tradeoff
        # Higher efSearch improves recall but increases latency
        current_ef = self.ef_search
        
        # For k=1 (as used in evaluation), we can use the stored efSearch
        if k != 1:
            # For k > 1, adjust efSearch slightly
            current_ef = max(self.ef_search, k * 100)
        
        self.index.hnsw.efSearch = current_ef
        
        # Perform search
        distances, indices = self.index.search(xq, k)
        
        return distances.astype(np.float32), indices.astype(np.int64)