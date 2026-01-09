import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.
        
        Using HNSW (Hierarchical Navigable Small World) graph which offers 
        excellent recall capabilities. 
        M=64 creates a denser graph than standard (M=32), improving recall 
        at the cost of slightly higher memory and latency, which is acceptable 
        given the 7.7ms budget.
        """
        self.dim = dim
        
        # M=64 is selected to maximize graph connectivity for high recall
        M = 64
        self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
        
        # High efConstruction ensures a high-quality graph is built
        # This takes longer to build but improves search efficiency/accuracy
        self.index.hnsw.efConstruction = 500
        
        # efSearch controls the trade-off between search speed and accuracy.
        # A value of 500 is very high (standard is 64-128), ensuring we meet
        # the high recall target (0.9914).
        # Given the generous 7.7ms latency budget and 8 vCPUs for batch processing,
        # this is safe.
        self.ef_search = 500

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        # Ensure input is float32 as required by Faiss
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
            
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.
        """
        # Ensure query is float32
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)
            
        # Set the search depth parameter
        self.index.hnsw.efSearch = self.ef_search
        
        # Perform batch search
        # Faiss utilizes OpenMP to parallelize this across available vCPUs
        distances, indices = self.index.search(xq, k)
        
        return distances, indices