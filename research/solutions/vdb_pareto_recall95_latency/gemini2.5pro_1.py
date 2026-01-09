import numpy as np
import faiss
from typing import Tuple

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.

        Args:
            dim: Vector dimensionality (e.g., 128 for SIFT1M)
            **kwargs: Optional parameters (e.g., M, ef_construction for HNSW)
        """
        self.dim = dim

        # Hyperparameters for faiss.IndexHNSWFlat, tuned for the Recall95 Latency Tier.
        # The strategy is to build a high-quality graph (high M and ef_construction)
        # to ensure the 95% recall gate is passed, and then tune ef_search to
        # be as low as possible (for low latency) while remaining safely above the recall gate.
        
        # M: Controls graph connectivity. Higher M improves recall but uses more memory.
        # 48 is a robust choice for SIFT1M, offering a good balance.
        self.M = int(kwargs.get('M', 48))
        
        # ef_construction: Controls graph quality during build time. Higher is better
        # but slower. Build time is a one-off cost, so a high value is affordable.
        self.ef_construction = int(kwargs.get('ef_construction', 400))
        
        # ef_search: The most critical search-time parameter. Controls search depth.
        # A higher value increases recall and latency. Based on benchmarks for a high-quality
        # graph (M=48, efC=400), ef_search=160 should provide recall well above 95%
        # while being significantly faster than more conservative settings.
        self.ef_search = int(kwargs.get('ef_search', 160))

        # Use HNSW with L2 distance, storing full vectors (no compression).
        # This avoids quantization errors, making it easier to hit high recall targets.
        self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction
        
        # Set FAISS to use all available CPU cores for parallelization.
        # The evaluation environment has 8 vCPUs.
        faiss.omp_set_num_threads(8)

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            xb: Base vectors, shape (N, dim), dtype float32
        """
        # Faiss requires float32 input.
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
        
        # For HNSW, adding vectors is what builds the index graph.
        self.index.add(xb)

    def search(self, xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.

        Args:
            xq: Query vectors, shape (nq, dim), dtype float32
            k: Number of nearest neighbors to return

        Returns:
            (distances, indices):
                - distances: shape (nq, k), dtype float32, L2-squared distances
                - indices: shape (nq, k), dtype int64, indices into base vectors
        """
        # Faiss requires float32 input.
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)

        # Set the search-time parameter for HNSW.
        self.index.hnsw.efSearch = self.ef_search
        
        # Perform the batch search.
        distances, indices = self.index.search(xq, k)
        
        return distances, indices