from functools import lru_cache
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from src.config import settings


@lru_cache(maxsize=1)
def _get_collection():
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh-v1.5"
    )
    client = chromadb.PersistentClient(path=str(settings.chroma_db_path))
    return client.get_collection(
        name=settings.collection_name,
        embedding_function=embedding_fn
    )


def retrieve(query: str, top_k: int = 5):
    """返回带 metadata 的检索结果列表"""
    collection = _get_collection()
    res = collection.query(query_texts=[query], n_results=top_k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    distances = res["distances"][0]
    return [
        {"text": d, "metadata": m, "distance": dist}
        for d, m, dist in zip(docs, metas, distances)
    ]


def retrieve_context(query: str, top_k: int = 5) -> str:
    """兼容旧接口，返回拼接后的 context"""
    results = retrieve(query, top_k)
    return "\n\n---\n\n".join(r["text"] for r in results)