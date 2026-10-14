from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from src.config import settings
from src.ingestion.parse import load_pdf_folder
from src.ingestion.chunker import split_text

def build_vector_db(pdf_dir: Path):
    # 使用本地BGE模型，不需要远程API
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(settings.chroma_db_path))
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=embedding_fn
    )

    doc_list = load_pdf_folder(pdf_dir)
    ids = []
    texts = []
    metadatas = []

    for file_name, full_text in doc_list:
        chunks = split_text(full_text)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{file_name}-{idx}"
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append({"source": file_name})

    collection.add(
        documents=texts,
        ids=ids,
        metadatas=metadatas
    )
    print(f"✅ 成功入库，共 {len(texts)} 个文本块")
    return collection
