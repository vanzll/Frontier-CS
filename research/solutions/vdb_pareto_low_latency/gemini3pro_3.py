import numpy as np
import faiss
from typing import Tuple

class LowLatencyIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.
        
        Uses HNSW (Hierarchical Navigable Small World) graph which offers 
        excellent recall-latency tradeoffs.
        """
        self.dim = dim
        
        # Configuration for HNSW
        # M=32: Increases graph connectivity, improving recall at slight cost to speed/memory.
        # 32 is a robust value for 1M vectors.
        self.M = 32
        
        # ef_construction: Controls index quality during build. 
        # Higher value = better graph, slower build. 
        # 200 is sufficient for SIFT1M to build a high-quality graph.
        self.ef_construction = 200
        
        # ef_search: Controls recall vs latency during search.
        # We need to stay under 2.31ms. 
        # On standard vCPUs, ef_search=80 usually executes in < 1.0ms 
        # for batch queries while providing high recall (~0.98+).
        self.ef_search = 80
        
        # Create HNSW Flat index (stores full vectors, no compression)
        self.index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        # Ensure data is float32 (FAISS requirement)
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
            
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        """
        # Ensure query is float32
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)
            
        # Set the search-time parameter
        # This determines the number of nodes visited during the graph traversal.
        self.index.hnsw.efSearch = self.ef_search
        
        # Perform batch search
        # FAISS automatically utilizes available threads for batch queries
        distances, indices = self.index.search(xq, k)
        
        return distances, indices