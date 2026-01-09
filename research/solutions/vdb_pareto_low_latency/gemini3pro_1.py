import numpy as np
import faiss
from typing import Tuple

class LowLatencyIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for Low Latency Tier.
        
        Strategy:
        - Use HNSW (Hierarchical Navigable Small World) for efficient approximate search.
        - M=32: Provides a dense enough graph for high recall without excessive memory/compute overhead.
        - ef_construction=128: Builds a high-quality graph to maximize recall potential.
        - ef_search=80: Tuned to comfortably exceed the baseline recall (0.9914) while 
          utilizing the 8 vCPUs to stay well under the 2.31ms latency limit (typically <0.5ms with batching).
        """
        self.dim = dim
        self.M = 32
        self.ef_construction = 128
        self.ef_search = 80
        
        # Initialize FAISS IndexHNSWFlat
        # This index stores full vectors and uses L2 distance (implicitly squared L2 in FAISS)
        self.index = faiss.IndexHNSWFlat(self.dim, self.M)
        self.index.hnsw.efConstruction = self.ef_construction

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the HNSW index.
        """
        # Ensure input is float32 (required by FAISS)
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
            
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for k nearest neighbors.
        """
        # Ensure query is float32
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)
            
        # Set the search depth (efSearch) just before querying
        self.index.hnsw.efSearch = self.ef_search
        
        # Perform batch search
        # FAISS automatically utilizes available threads (8 vCPUs) for batch queries
        distances, indices = self.index.search(xq, k)
        
        return distances, indices