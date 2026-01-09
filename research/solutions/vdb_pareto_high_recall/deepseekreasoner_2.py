import numpy as np
import faiss

class HighRecallIndex:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.M = kwargs.get('M', 64)
        self.ef_construction = kwargs.get('ef_construction', 400)
        self.ef_search = kwargs.get('ef_search', 800)
        self.index = None
        
    def add(self, xb: np.ndarray) -> None:
        if self.index is None:
            self.index = faiss.IndexHNSWFlat(self.dim, self.M)
            self.index.hnsw.efConstruction = self.ef_construction
            self.index.hnsw.efSearch = self.ef_search
        self.index.add(xb)
    
    def search(self, xq: np.ndarray, k: int):
        if self.index is None:
            raise RuntimeError("Index not built")
        self.index.hnsw.efSearch = self.ef_search
        return self.index.search(xq, k)