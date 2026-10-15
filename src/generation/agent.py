from src.retrieval.retriever import retrieve_context
from openai import OpenAI
from src.config import settings

client = OpenAI(
    api_key=settings.deepseek_api_key,
    base_url="https://api.deepseek.com/v1"
)

SYSTEM_PROMPT = """你是一个通信工程领域的问答助手，专门回答关于6G、频谱管控、协作频谱感知、
动态频谱接入、区块链频谱安全等问题。

回答规则：
1. 仅根据提供的参考文档片段回答问题。
2. 如果参考文档片段中没有直接、明确地包含答案所需的信息，你必须回答“根据提供的文档片段，无法确定”，
   不要根据片段中的零散词汇推测或编造。
3. 不要补充参考文档之外的知识。
4. 每个关键结论后用 [来源: 文件名 | 章节号] 标注依据。
"""


def answer_question(question: str):
    context = retrieve_context(question)
    user_prompt = f"""参考文档片段：
{context}

用户问题：{question}

请按规则回答，并标注来源。"""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    return resp.choices[0].message.content