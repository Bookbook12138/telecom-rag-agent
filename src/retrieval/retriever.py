from src.config import settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb

def retrieve_context(query: str, top_k: int = 3):
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(settings.chroma_db_path))
    collection = client.get_collection(
        name=settings.collection_name,
        embedding_function=embedding_fn
    )
    res = collection.query(query_texts=[query], n_results=top_k)
    docs = res["documents"][0]
    return "\n\n".join(docs)
