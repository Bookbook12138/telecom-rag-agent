from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from src.config import settings
from src.ingestion.parse import load_pdf_folder
from src.ingestion.chunker import split_by_sections


def build_vector_db(pdf_dir: Path):
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh-v1.5"
    )
    client = chromadb.PersistentClient(path=str(settings.chroma_db_path))

    # 已存在则直接加载，不重建
    try:
        collection = client.get_collection(
            name=settings.collection_name,
            embedding_function=embedding_fn
        )
        print(f"✅ 已加载现有 collection: {settings.collection_name}")
        return collection
    except Exception:
        pass

    # 不存在则新建
    collection = client.create_collection(
        name=settings.collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    ids, texts, metadatas = [], [], []
    for file_name, full_text in load_pdf_folder(pdf_dir):Get-ChildItem -Force | Select-Object Name
        chunks = split_by_sections(full_text, source=file_name)
        for c in chunks:
            header = f"[{c.source} | Section {c.section_id} {c.section_title}]"
            ids.append(f"{c.source}-{c.chunk_index}")
            texts.append(f"{header}\n{c.text}")
            metadatas.append({
                "source": c.source,
                "section_id": c.section_id,
                "section_title": c.section_title,
                "chunk_index": c.chunk_index
            })

    collection.add(documents=texts, ids=ids, metadatas=metadatas)
    print(f"✅ 成功入库，共 {len(texts)} 个章节感知 chunk")
    return collection