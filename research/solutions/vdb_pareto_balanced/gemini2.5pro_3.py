import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        # These parameters are selected to achieve high recall (>0.9914)
        # on the SIFT1M dataset while remaining well within the 5.775ms
        # latency constraint on an 8-core CPU environment.
        # HNSW (Hierarchical Navigable Small World) is chosen for its
        # excellent speed-recall tradeoff on in-memory datasets.
        self.M = kwargs.get('M', 48)
        self.ef_construction = kwargs.get('ef_construction', 500)
        self.ef_search = kwargs.get('ef_search', 256)
        
        # faiss.IndexHNSWFlat stores full vectors without compression, which
        # is ideal for maximizing recall. faiss.METRIC_L2 corresponds to
        # Euclidean distance (faiss returns squared L2 distances).
        self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction
        self.is_built = False

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        # faiss requires float32 data type for its operations.
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
        
        # The `add` method builds the HNSW graph. This process is
        # CPU-intensive but is parallelized by faiss using OpenMP.
        self.index.add(xb)
        self.is_built = True

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.
        """
        if not self.is_built:
            raise RuntimeError("Index has not been built. Call add() first.")

        # Ensure query vectors are also float32.
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)

        # efSearch is the most critical search-time parameter. It controls
        # the size of the priority queue used to traverse the graph. A larger
        # value increases recall at the cost of higher latency.
        self.index.hnsw.efSearch = self.ef_search
        
        # The search method is highly optimized for batch queries and will
        # utilize all available CPU cores.
        distances, indices = self.index.search(xq, k)
        
        return distances, indices