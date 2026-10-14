from src.retrieval.retriever import retrieve_context
from openai import OpenAI
from src.config import settings

client = OpenAI(
    api_key=settings.deepseek_api_key,
    base_url="https://api.deepseek.com/v1"
)

def answer_question(question: str):
    context = retrieve_context(question)
    prompt = f"""基于下面参考文档回答用户问题：
参考文档：
{context}

用户问题：{question}
请只用参考文档内容回答。
"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"user", "content":prompt}]
    )
    return resp.choices[0].message.content
