import numpy as np
import faiss

class LowLatencyIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize a quantized IVF index for low latency search.
        Uses PQ for compression and small nprobe for speed.
        """
        self.dim = dim
        self.index = None
        
        # Optimized parameters for SIFT1M with strict latency constraint
        self.nlist = 1024  # Number of Voronoi cells (clusters)
        self.nprobe = 4    # Number of cells to visit at query time
        self.m = 16        # Number of subquantizers for PQ (must divide dim=128)
        self.nbits = 8     # Bits per subquantizer
        
        # Override with kwargs if provided
        if 'nlist' in kwargs:
            self.nlist = kwargs['nlist']
        if 'nprobe' in kwargs:
            self.nprobe = kwargs['nprobe']
        if 'm' in kwargs:
            self.m = kwargs['m']
        if 'nbits' in kwargs:
            self.nbits = kwargs['nbits']
            
        # Initialize quantizer
        self.quantizer = faiss.IndexFlatL2(dim)
        
        # Create the IVF index with product quantization
        self.index = faiss.IndexIVFPQ(self.quantizer, dim, self.nlist, self.m, self.nbits)
        
        # Disable exhaustive training to save time
        self.index.cp.min_points_per_centroid = 5
        self.index.cp.max_points_per_centroid = 1000000
        
        # Track if index is trained
        self.is_trained = False
        self.vectors_added = 0

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index.
        """
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
            
        # Train on first batch if not trained yet
        if not self.is_trained:
            # Use subset for faster training
            n_train = min(len(xb), 100000)
            train_data = xb[:n_train]
            
            # Train with centroids initialization
            self.index.train(train_data)
            self.is_trained = True
            
        # Add vectors
        self.index.add(xb)
        self.vectors_added += len(xb)

    def search(self, xq: np.ndarray, k: int) -> tuple:
        """
        Search for k nearest neighbors with low latency.
        """
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)
            
        # Set small nprobe for low latency
        self.index.nprobe = self.nprobe
        
        # Search with low nprobe
        distances, indices = self.index.search(xq, k)
        
        return distances, indices