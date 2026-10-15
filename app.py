import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import sentence_transformers
except ImportError:
    install_package("sentence-transformers")
    import sentence_transformers

from pathlib import Path
import chromadb
from src.config import settings
from src.retrieval.indexer import build_vector_db
from src.generation.agent import answer_question


def _collection_exists() -> bool:
    client = chromadb.PersistentClient(path=str(settings.chroma_db_path))
    try:
        client.get_collection(settings.collection_name)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    if not _collection_exists():
        print("未找到向量库，正在构建...")
        build_vector_db(Path("./data/pdf"))
    else:
        print("已加载现有向量库。")

    print("Telecom RAG Agent (type 'exit' to quit)")
    while True:
        q = input("\n请输入你的问题（输入 exit 退出）：")
        if q.strip().lower() == "exit":
            break
        ans = answer_question(q)
        print("\n==== 回答 ====\n", ans)