import numpy as np
import faiss
from typing import Tuple

class HighRecallIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize HNSW index optimized for high recall on SIFT1M.
        
        Parameters tuned for Recall@1 > 0.9914 within 7.7ms latency budget.
        Using M=64 and ef_search=480 provides robust graph navigation.
        """
        self.dim = dim
        self.M = kwargs.get('M', 64)
        self.ef_construction = kwargs.get('ef_construction', 256)
        self.ef_search = kwargs.get('ef_search', 480)
        
        # IndexHNSWFlat: HNSW graph on raw vectors (no quantization)
        # This maximizes recall accuracy at the cost of memory (which fits in 16GB)
        self.index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction
        
    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        # Ensure data is float32 and C-contiguous for Faiss
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
        if not xb.flags.c_contiguous:
            xb = np.ascontiguousarray(xb)
            
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        """
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)
        if not xq.flags.c_contiguous:
            xq = np.ascontiguousarray(xq)
            
        # Set search depth dynamically
        self.index.hnsw.efSearch = self.ef_search
        
        # Faiss handles batch queries efficiently using OpenMP
        D, I = self.index.search(xq, k)
        return D, I