import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.
        
        Using HNSW (Hierarchical Navigable Small World) graph which offers 
        state-of-the-art performance for the Recall vs Latency trade-off.
        """
        self.dim = dim
        
        # Configuration for SIFT1M (1M vectors, 128 dim)
        # M=32: Number of connections per node. Good balance for 128D vectors.
        # Higher M improves recall but increases memory and add time slightly.
        self.M = 32
        
        # Create HNSW index with Flat storage (no compression to maximize recall)
        self.index = faiss.IndexHNSWFlat(dim, self.M)
        
        # efConstruction: Controls index quality during build. 
        # Higher = better graph = faster search for same recall.
        # 200 is robust for SIFT1M and fits within construction time limits.
        self.index.hnsw.efConstruction = 200
        
        # Explicitly set number of threads to utilize the 8 vCPUs for batch queries
        faiss.omp_set_num_threads(8)

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        # Faiss handles the HNSW graph construction internally
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        """
        # efSearch: Controls the search depth.
        # Critical parameter for the Latency vs Recall trade-off.
        # For SIFT1M with HNSW M=32:
        # efSearch=40 typically yields Recall@1 > 0.90 (well above the 0.80 gate)
        # while maintaining extremely low latency (amortized < 0.1ms on 8 cores).
        self.index.hnsw.efSearch = 40
        
        # Perform the search using Faiss optimized C++ implementation
        D, I = self.index.search(xq, k)
        return D, I