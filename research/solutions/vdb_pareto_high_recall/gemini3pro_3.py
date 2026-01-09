import numpy as np
import faiss
from typing import Tuple

class HighRecallIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.
        
        Strategy:
        - Use HNSW (Hierarchical Navigable Small World) graph which offers excellent recall-latency trade-offs.
        - The latency constraint (7.7ms) is generous for batch queries on SIFT1M, allowing us to use
          aggressive parameters to maximize recall.
        - M=64 and efConstruction=256 create a high-quality graph.
        - efSearch=512 ensures very high recall (typically >0.998) while keeping latency <1ms per query 
          in batch mode on 8 vCPUs.
        """
        self.dim = dim
        
        # HNSW Parameters
        # M: Number of connections per node. Higher M = better recall, higher memory/latency.
        # M=64 is high but safe within memory/latency limits.
        self.M = kwargs.get('M', 64)
        
        # ef_construction: Depth of search during build. Higher = better graph.
        self.ef_construction = kwargs.get('ef_construction', 256)
        
        # ef_search: Depth of search during query. Higher = better recall.
        # 512 is chosen to virtually guarantee maximizing recall given the relaxed latency budget.
        self.ef_search = kwargs.get('ef_search', 512)

        # Initialize FAISS IndexHNSWFlat (HNSW on raw vectors, L2 distance)
        self.index = faiss.IndexHNSWFlat(self.dim, self.M)
        self.index.hnsw.efConstruction = self.ef_construction

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        # FAISS expects float32 and C-contiguous memory
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
        if not xb.flags.c_contiguous:
            xb = np.ascontiguousarray(xb)
            
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.
        """
        # Prepare query vectors
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)
        if not xq.flags.c_contiguous:
            xq = np.ascontiguousarray(xq)
            
        # Configure search depth
        # Ensure efSearch is at least k
        self.index.hnsw.efSearch = max(self.ef_search, k)
        
        # Perform search (returns distances and indices)
        distances, indices = self.index.search(xq, k)
        
        return distances, indices