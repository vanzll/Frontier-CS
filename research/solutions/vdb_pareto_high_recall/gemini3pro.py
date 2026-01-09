import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.
        
        This implementation uses HNSW (Hierarchical Navigable Small World) graph 
        to maximize recall within the relaxed latency constraint (7.7ms).
        """
        self.dim = dim
        
        # HNSW Parameters
        # M: Number of connections per node. 
        # M=64 creates a dense graph, significantly improving recall potential 
        # at the cost of slightly higher memory and construction time.
        M = 64
        
        # Initialize the HNSW index with L2 metric
        self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
        
        # efConstruction: Depth of search during index construction.
        # Higher value (256) ensures a higher quality graph structure.
        self.index.hnsw.efConstruction = 256
        
        # efSearch: Depth of search during queries.
        # Setting this high (512) maximizes recall. 
        # Given the 7.7ms latency budget and batch query support on 8 vCPUs, 
        # we can afford a very deep search to ensure recall > 0.9914.
        self.ef_search = 512
        
        # Set the number of threads for Faiss to utilize all 8 vCPUs
        faiss.omp_set_num_threads(8)

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        # Faiss requires C-contiguous arrays
        if not xb.flags['C_CONTIGUOUS']:
            xb = np.ascontiguousarray(xb, dtype=np.float32)
            
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.
        """
        # Faiss requires C-contiguous arrays
        if not xq.flags['C_CONTIGUOUS']:
            xq = np.ascontiguousarray(xq, dtype=np.float32)
            
        # Set the search depth parameter for the HNSW algorithm
        self.index.hnsw.efSearch = self.ef_search
        
        # Perform batch search
        # D: distances, I: indices
        D, I = self.index.search(xq, k)
        
        return D, I