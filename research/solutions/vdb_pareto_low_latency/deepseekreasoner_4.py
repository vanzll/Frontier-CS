import numpy as np
import faiss

class LowLatencyIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize a quantized IVF index optimized for low latency.
        Using IVF512 with 2 nprobe for aggressive speed optimization.
        """
        self.dim = dim
        self.M = kwargs.get('M', 8)  # HNSW M parameter if needed
        self.nlist = 512  # Number of IVF clusters
        self.nprobe = 2   # Number of clusters to probe during search
        self.quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFFlat(self.quantizer, dim, self.nlist)
        self.index.nprobe = self.nprobe
        
        # Enable direct map for exact distance computation
        self.index.make_direct_map()
        
    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index with training.
        """
        if not self.index.is_trained:
            # Train on a subset for speed
            subset_size = min(100000, len(xb))
            train_data = xb[:subset_size].copy()
            self.index.train(train_data)
        
        self.index.add(xb)
    
    def search(self, xq: np.ndarray, k: int) -> tuple:
        """
        Search for k nearest neighbors using IVF with low nprobe.
        """
        # Ensure k doesn't exceed number of vectors
        k = min(k, self.index.ntotal)
        
        # Pre-allocate output arrays
        distances = np.empty((len(xq), k), dtype=np.float32)
        indices = np.empty((len(xq), k), dtype=np.int64)
        
        # Perform search
        self.index.search(xq, k, distances, indices)
        
        return distances, indices