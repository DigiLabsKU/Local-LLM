from sentence_transformers import SentenceTransformer
from typing import List
from torch.cuda import is_available as gpu_is_available
from langchain_core.embeddings.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = SentenceTransformer("sentence-transformers/"+model)
        if gpu_is_available():
            self.model.cuda()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(text).tolist() for text in texts]
    
    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()

    def get_dimensions(self) -> int:
        return self.model.get_sentence_embedding_dimension() or len(self.embed_query("hello world"))
    
