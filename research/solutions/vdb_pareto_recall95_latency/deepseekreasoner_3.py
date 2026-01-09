import numpy as np
import faiss
from typing import Tuple
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize HNSW index for high recall with latency optimization.
        Parameters tuned for 95%+ recall on SIFT1M.
        """
        self.dim = dim
        
        # HNSW parameters optimized for 95%+ recall within 7.7ms
        # Using higher M and ef_search for better recall
        self.M = kwargs.get('M', 32)  # Connections per node
        self.ef_construction = kwargs.get('ef_construction', 200)  # Construction time/accuracy
        self.ef_search = kwargs.get('ef_search', 128)  # Search accuracy
        
        # Create HNSW index with L2 metric
        self.index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        
        # Set number of threads for parallel search (8 vCPUs)
        faiss.omp_set_num_threads(8)
        
        # Flag to track if vectors are added
        self.is_trained = True  # HNSW doesn't require separate training
        
    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the HNSW index.
        """
        self.index.add(xb)
        
    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors using HNSW.
        Returns (distances, indices) with proper shapes.
        """
        # Ensure ef_search is set for each search
        self.index.hnsw.efSearch = self.ef_search
        
        # Perform the search
        distances, indices = self.index.search(xq, k)
        
        # Ensure proper dtypes and shapes
        return distances.astype(np.float32), indices.astype(np.int64)