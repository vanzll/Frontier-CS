import numpy as np
import faiss
import time

class IVFIndex:
    def __init__(self, dim: int, **kwargs):
        """
        Initialize IVF index for low latency requirements.
        Uses aggressive parameters to meet 2.31ms constraint.
        """
        self.dim = dim
        
        # IVF parameters optimized for low latency
        # nlist: number of Voronoi cells (clusters)
        # nprobe: number of cells to visit during search (kept very low for speed)
        self.nlist = kwargs.get('nlist', 512)  # Reduced from typical 1024 for faster search
        self.nprobe = kwargs.get('nprobe', 4)  # Very low nprobe for strict latency
        
        # Create the quantizer (coarse quantizer)
        quantizer = faiss.IndexFlatL2(dim)
        
        # Create IVF index with flat storage
        self.index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_L2)
        
        # Training sample size (for initial training)
        self.train_size = kwargs.get('train_size', 100000)
        
        # Track if index is trained
        self.is_trained = False
        
        # Store vectors for training (if needed)
        self.train_vectors = None
        self.total_added = 0
        
        # Enable parallelism for batch operations
        faiss.omp_set_num_threads(8)

    def add(self, xb: np.ndarray) -> None:
        """
        Add vectors to the index with incremental training if needed.
        """
        if not self.is_trained:
            # First addition: train the index
            if self.train_vectors is None:
                self.train_vectors = xb.copy()
            else:
                self.train_vectors = np.vstack([self.train_vectors, xb])
            
            # Train when we have enough samples
            if self.train_vectors.shape[0] >= self.train_size:
                print(f"Training IVF index with {self.train_vectors.shape[0]} vectors")
                self.index.train(self.train_vectors[:self.train_size])
                self.is_trained = True
                
                # Add all training vectors
                self.index.add(self.train_vectors)
                self.total_added = self.train_vectors.shape[0]
                
                # Free training vectors
                self.train_vectors = None
            else:
                # Not enough for training yet, just store
                self.total_added += xb.shape[0]
                return
        else:
            # Index is already trained, add directly
            self.index.add(xb)
            self.total_added += xb.shape[0]
    
    def search(self, xq: np.ndarray, k: int) -> tuple:
        """
        Search for k nearest neighbors with aggressive nprobe for low latency.
        """
        # Set nprobe for this search
        self.index.nprobe = self.nprobe
        
        # Ensure k doesn't exceed total vectors
        actual_k = min(k, self.total_added)
        
        # Perform search
        distances, indices = self.index.search(xq, actual_k)
        
        # If we need more neighbors than available, pad with -1
        if actual_k < k:
            padding = k - actual_k
            distances = np.pad(distances, ((0, 0), (0, padding)), 
                             mode='constant', constant_values=np.finfo(np.float32).max)
            indices = np.pad(indices, ((0, 0), (0, padding)), 
                           mode='constant', constant_values=-1)
        
        return distances.astype(np.float32), indices.astype(np.int64)