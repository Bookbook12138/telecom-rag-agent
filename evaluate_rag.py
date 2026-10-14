import csv
import time
from datetime import datetime
from src.retrieval.retriever import retrieve_context
from src.generation.agent import answer_question
from src.config import settings

# ===================== 在这里自定义你的测试问题 =====================
TEST_QUESTIONS = [
    "本文提出的核心技术是什么？",
    "该方案解决了哪些现存问题？",
    "实验的数据集与评价指标是什么？",
    "该方法的局限性有哪些？",
    "文中仿真的参数设置包含哪些内容？",
    "该方案相比其他基线方案的优势是什么？",
    "文章结论部分有哪些展望？",
    "5G网络的商业部署成本是多少？" #域外问题，测试防幻觉
]
# ==================================================================

def run_evaluation():
    output_csv = "rag_eval_result.csv"
    fieldnames = ["timestamp", "question", "retrieved_context", "answer"]
    results = []

    print("===== RAG批量评估开始 =====")
    start_total = time.time()

    for q in TEST_QUESTIONS:
        print(f"\n【正在评估】问题：{q}")
        t0 = time.time()

        # 召回参考上下文
        context_text = retrieve_context(q, top_k=3)
        # 调用agent生成回答
        answer = answer_question(q)

        t1 = time.time()
        print(f"单条问答耗时：{t1 - t0:.2f}s")

        results.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": q,
            "retrieved_context": context_text,
            "answer": answer
        })

    # 写入CSV文件，Excel可直接打开
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # 同时输出txt备份文件
    with open("rag_eval_result.txt", "w", encoding="utf-8") as f_txt:
        for item in results:
            f_txt.write("="*60 + "\n")
            f_txt.write(f"时间：{item['timestamp']}\n")
            f_txt.write(f"问题：{item['question']}\n")
            f_txt.write(f"召回片段：\n{item['retrieved_context']}\n")
            f_txt.write(f"回答：\n{item['answer']}\n")

    total_time = time.time() - start_total
    print(f"\n===== 评估全部完成！总耗时：{total_time:.2f}s =====")
    print(f"结果已保存至：{output_csv} 和 rag_eval_result.txt")

if __name__ == "__main__":
    run_evaluation()
