import numpy as np
import faiss

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        
        # Use IVF index with minimal probes for extreme speed
        # nlist = 4096 provides good recall with nprobe=1
        nlist = kwargs.get('nlist', 4096)
        
        # Create quantizer
        quantizer = faiss.IndexFlatL2(dim)
        
        # Create IVF index with 4096 centroids
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        
        # Set nprobe for search - using 1 for minimal latency
        # This is critical: small nprobe = faster search
        self.nprobe = kwargs.get('nprobe', 1)
        
        # Track if trained
        self.is_trained = False
        
    def add(self, xb: np.ndarray) -> None:
        """Add vectors to the index."""
        if not self.is_trained:
            # Train on a subset for speed (100K samples)
            n_train = min(100000, len(xb))
            self.index.train(xb[:n_train])
            self.is_trained = True
        
        # Add all vectors
        self.index.add(xb)
        
    def search(self, xq: np.ndarray, k: int) -> tuple:
        """Search for k nearest neighbors."""
        # Set nprobe before search (critical for speed)
        self.index.nprobe = self.nprobe
        
        # Search with batch processing
        D, I = self.index.search(xq, k)
        
        return D.astype(np.float32), I.astype(np.int64)