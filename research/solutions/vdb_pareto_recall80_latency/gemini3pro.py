import numpy as np
import faiss
from typing import Tuple

class Recall80Index:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index using HNSW with Flat storage.
        HNSW is chosen for its superior latency/recall tradeoff on CPU.
        """
        self.dim = dim
        
        # HNSW configuration
        # M=32: robust graph connectivity
        # METRIC_L2: Required metric
        self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_L2)
        
        # efConstruction=100: Invest time in building a high-quality graph
        # to allow faster traversal (lower efSearch) during query time.
        self.index.hnsw.efConstruction = 100
        
        # Utilize all available cores (8 vCPUs) for batch processing
        faiss.omp_set_num_threads(8)

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the HNSW index.
        """
        # Ensure input is float32 and C-contiguous
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
        if not xb.flags.c_contiguous:
            xb = np.ascontiguousarray(xb)
            
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search with optimized parameters for Recall@1 >= 0.80.
        """
        # Ensure query is float32 and C-contiguous
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)
        if not xq.flags.c_contiguous:
            xq = np.ascontiguousarray(xq)
            
        # Set search parameter efSearch.
        # Constraint: Recall@1 >= 0.80.
        # Optimization: Minimize latency (t < 0.6ms).
        # Strategy:
        # On SIFT1M with M=32, efSearch=10 often yields >85% recall.
        # We set efSearch=20 to provide a strong safety margin for the 80% gate,
        # ensuring the score is not 0 due to recall failure.
        # Expected latency for efSearch=20 is approx 0.1-0.2ms, well within the 0.6ms limit.
        self.index.hnsw.efSearch = 20
        
        # Perform batch search
        # Faiss releases GIL and uses OpenMP for parallel execution
        distances, indices = self.index.search(xq, k)
        
        return distances, indices