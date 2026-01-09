import numpy as np
import faiss
from typing import Tuple

class LowLatencyIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.
        Optimized for strict latency constraint (< 2.31ms).
        """
        self.dim = dim
        
        # Configuration for HNSW
        # M=24: Slightly reduced graph connectivity (vs standard 32) to reduce 
        # distance computations per hop, ensuring we meet the strict latency target.
        self.M = 24
        
        # ef_construction=100: Invest in graph quality during build time 
        # to allow for lower ef_search at runtime.
        self.ef_construction = 100
        
        # ef_search=40: Runtime search depth.
        # Tuned to safely stay under 2.31ms (targeting ~1.5ms) while 
        # maintaining recall well above the 0.7931 minimum (targeting >0.90).
        self.ef_search = 40
        
        # Initialize FAISS HNSW Flat index
        # HNSWFlat is fast and fits in memory for SIFT1M (1M vectors)
        self.index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        # FAISS handles efficient batch addition and multi-threading
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.
        """
        # Set the runtime parameter specifically for this search
        self.index.hnsw.efSearch = self.ef_search
        
        # Perform batch search
        # FAISS releases GIL and utilizes available CPU cores
        D, I = self.index.search(xq, k)
        
        return D, I