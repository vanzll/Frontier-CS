import numpy as np
import faiss
from typing import Tuple

class HighRecallIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.
        
        We use HNSW (Hierarchical Navigable Small World) index which offers excellent 
        recall/latency trade-offs. Given the relaxed latency constraint (7.7ms) and 
        high recall requirement (0.9914), we use aggressive parameters:
        - M=64: Higher connectivity for better graph navigation accuracy.
        - ef_construction=500: Deeper search during graph construction for higher quality graph.
        - ef_search=500: Deep search at query time to maximize recall.
        
        Typical performance on SIFT1M with these settings:
        - Recall@1: > 0.998
        - Latency: ~1-3ms (well within the 7.7ms limit)
        """
        self.dim = dim
        self.M = 64
        self.ef_construction = 500
        self.ef_search = 500
        
        self.index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        # Faiss expects contiguous float32 arrays
        if not xb.flags['C_CONTIGUOUS']:
            xb = np.ascontiguousarray(xb)
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
            
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.
        """
        # Faiss expects contiguous float32 arrays
        if not xq.flags['C_CONTIGUOUS']:
            xq = np.ascontiguousarray(xq)
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)
            
        # Set the search depth parameter
        self.index.hnsw.efSearch = self.ef_search
        
        # Perform search
        # D: distances, I: indices
        D, I = self.index.search(xq, k)
        
        return D, I