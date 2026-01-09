import numpy as np
import faiss

class LowLatencyIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize the index for vectors of dimension `dim`.
        
        Uses IVFFlat with aggressive parameters optimized for low latency:
        - nlist: 4096 (coarse partitions)
        - nprobe: 2 (very low for speed)
        - Use IVF with flat storage for accurate distance calculations
        """
        self.dim = dim
        self.nlist = kwargs.get('nlist', 4096)
        self.nprobe = kwargs.get('nprobe', 2)
        self.quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFFlat(self.quantizer, dim, self.nlist, faiss.METRIC_L2)
        self.index.nprobe = self.nprobe
        self.built = False
        
    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        
        Args:
            xb: Base vectors, shape (N, dim), dtype float32
        """
        if not self.built:
            # Train on a subset if too many vectors
            n_train = min(100000, len(xb))
            train_vectors = xb[:n_train].copy()
            self.index.train(train_vectors)
            self.built = True
            
        self.index.add(xb)
        
    def search(self, xq: np.ndarray, k: int):
        """
        Search for k nearest neighbors of query vectors.
        
        Args:
            xq: Query vectors, shape (nq, dim), dtype float32
            k: Number of nearest neighbors to return
            
        Returns:
            (distances, indices): L2 distances and indices
        """
        # Set search parameters for low latency
        original_nprobe = self.index.nprobe
        self.index.nprobe = self.nprobe
        
        # Batch search
        distances, indices = self.index.search(xq, k)
        
        # Restore original nprobe (though we always use the same)
        self.index.nprobe = original_nprobe
        
        return distances, indices