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
from src.retrieval.indexer import build_vector_db
from src.generation.agent import answer_question

if __name__ == "__main__":
    pdf_folder = Path("./data/pdf")
    build_vector_db(pdf_folder)
    while True:
        q = input("\n请输入你的问题(输入exit退出)：")
        if q.strip().lower() == "exit":
            break
        ans = answer_question(q)
        print("\n====回答====\n", ans)
