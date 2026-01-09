import numpy as np
import faiss
from typing import Tuple

class Recall95Index:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.
        We use HNSW (Hierarchical Navigable Small World) graph which offers
        excellent recall/latency trade-offs.
        """
        self.dim = dim
        
        # HNSW Parameters
        # M: Number of connections per node. 32 is a robust choice for SIFT1M (128d).
        self.M = 32
        # ef_construction: Depth of search during index build. 
        # Higher values (e.g. 200) improve graph quality at cost of build time.
        self.ef_construction = 200
        # ef_search: Depth of search during query.
        # We need >= 0.95 recall. ef_search=80 is conservative to guarantee >0.95 (typically >0.98)
        # while keeping latency very low (well below the 7.7ms limit).
        self.ef_search = 80
        
        # Create HNSW index with Flat storage (no compression) for max accuracy
        self.index = faiss.IndexHNSWFlat(dim, self.M)
        self.index.hnsw.efConstruction = self.ef_construction
        
        # Configure Faiss to use all available cores (8 vCPUs)
        faiss.omp_set_num_threads(8)

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        # Ensure data is contiguous and float32 for Faiss
        xb = np.ascontiguousarray(xb, dtype=np.float32)
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        """
        # Ensure data is contiguous and float32
        xq = np.ascontiguousarray(xq, dtype=np.float32)
        
        # Set search-time parameter
        self.index.hnsw.efSearch = self.ef_search
        
        # Perform search
        # Faiss returns squared L2 distances and indices
        D, I = self.index.search(xq, k)
        
        return D, I