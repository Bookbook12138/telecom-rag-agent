import os
from pathlib import Path
from dotenv import load_dotenv

# 强制加载项目根目录的.env
BASE_DIR = Path(__file__).parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    def __init__(self):
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.deepseek_api_key:
            raise ValueError(f"DEEPSEEK_API_KEY not set in .env，尝试读取路径：{env_path}")
        self.collection_name = "telecom_docs"
        self.chroma_db_path = Path("./chroma_db")
        self.llm_model = os.getenv("DEEPSEEK_LLM_MODEL", "deepseek-chat")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

settings = Settings()
