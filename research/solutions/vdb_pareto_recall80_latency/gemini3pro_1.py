import numpy as np
import faiss
from typing import Tuple

class Recall80Index:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.
        
        Strategy:
        - Use HNSW (Hierarchical Navigable Small World) graph with Flat storage.
        - HNSWFlat provides the best latency/recall tradeoff for SIFT1M on CPU.
        - We set a high construction budget (efConstruction=200) to ensure the graph 
          is high quality, which allows us to use a smaller search budget later.
        - We set a low search budget (efSearch=10) to minimize latency while 
          meeting the 80% recall gate.
        """
        self.dim = dim
        
        # M=32 provides a robust graph for 1M vectors
        self.index = faiss.IndexHNSWFlat(dim, 32)
        
        # Increase build quality to improve search speed/accuracy ratio
        self.index.hnsw.efConstruction = 200
        
        # Target efSearch.
        # Benchmarks on SIFT1M/HNSW32 show efSearch=10 yields ~90% Recall@1.
        # This provides a safety margin over the 80% requirement while being fast.
        self.default_ef_search = 10

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        # Faiss handles data copying/formatting internally
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        """
        # Adjust efSearch dynamically: must be at least k
        self.index.hnsw.efSearch = max(self.default_ef_search, k)
        
        # Perform search
        # Faiss automatically utilizes available vCPUs (OpenMP) for batch queries
        D, I = self.index.search(xq, k)
        
        return D, I