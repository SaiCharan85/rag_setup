from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict
import os
from config import EMBEDDING_MODEL

class FaissRetriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.texts = []
        self.ids = []
        self.dim = self.model.get_sentence_embedding_dimension()
        
    def add_documents(self, texts: List[str], embeddings: List[np.ndarray] = None):
        """Add documents to the index"""
        if embeddings is None:
            embeddings = self.model.encode(texts)
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dim)
        
        # Convert embeddings to float32
        embeddings = np.array(embeddings).astype('float32')
        
        # Add to index
        self.index.add(embeddings)
        
        # Store texts and create IDs
        start_idx = len(self.texts)
        self.texts.extend(texts)
        self.ids.extend(range(start_idx, start_idx + len(texts)))
        
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        
        D, I = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx >= 0:
                results.append({
                    'id': self.ids[idx],
                    'text': self.texts[idx],
                    'score': float(score)
                })
        
        return results
        
    def save_index(self, path: str):
        """Save FAISS index and metadata"""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, 'index.faiss'))
        
        # Save metadata
        metadata = {
            'texts': self.texts,
            'ids': self.ids
        }
        np.save(os.path.join(path, 'metadata.npy'), metadata)
        
    def load_index(self, path: str):
        """Load FAISS index and metadata"""
        self.index = faiss.read_index(os.path.join(path, 'index.faiss'))
        metadata = np.load(os.path.join(path, 'metadata.npy'), allow_pickle=True).item()
        self.texts = metadata['texts']
        self.ids = metadata['ids']
