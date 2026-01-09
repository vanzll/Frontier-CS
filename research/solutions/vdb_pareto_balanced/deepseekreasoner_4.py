import numpy as np
import faiss
import warnings
warnings.filterwarnings("ignore")

class YourIndexClass:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.M = kwargs.get('M', 24)
        self.ef_construction = kwargs.get('ef_construction', 200)
        self.ef_search = kwargs.get('ef_search', 128)
        self.index = None
        
    def add(self, xb: np.ndarray) -> None:
        if self.index is None:
            self.index = faiss.IndexHNSWFlat(self.dim, self.M)
            self.index.hnsw.efConstruction = self.ef_construction
            self.index.hnsw.efSearch = self.ef_search
        self.index.add(xb)
        
    def search(self, xq: np.ndarray, k: int) -> tuple:
        faiss.omp_set_num_threads(8)
        D, I = self.index.search(xq, k)
        return D.astype(np.float32), I.astype(np.int64)