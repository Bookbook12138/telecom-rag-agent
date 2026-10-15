import json
import time
import csv
from pathlib import Path
from src.retrieval.retriever import retrieve
from src.generation.agent import answer_question

GROUND_TRUTH_PATH = Path("ground_truth.json")
RESULT_CSV = Path("rag_eval_result.csv")
RESULT_TXT = Path("rag_eval_result.txt")
TOP_K = 5


def load_ground_truth():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def section_match(retrieved_section: str, expected_sections: list) -> bool:
    if not retrieved_section:
        return False
    for exp in expected_sections:
        if retrieved_section == exp:
            return True
        if retrieved_section.startswith(exp + "."):
            return True
    return False


def evaluate():
    gt = load_ground_truth()
    queries = gt["queries"]

    rows = []
    recall_hits = 0
    recall_total = 0
    hallucination_cases = 0
    hallucination_hits = 0

    print("===== RAG Recall@5 评估开始 =====\n")

    for item in queries:
        question = item["question"]
        expected = item.get("expected_sections", [])
        is_hallucination_test = len(expected) == 0

        start = time.time()
        retrieved = retrieve(question, top_k=TOP_K)
        answer = answer_question(question)
        elapsed = round(time.time() - start, 2)

        retrieved_sections = [
            r["metadata"].get("section_id", "") for r in retrieved
        ]

        hit = any(section_match(sec, expected) for sec in retrieved_sections)

        if is_hallucination_test:
            hallucination_cases += 1
            refused = any(
                kw in answer for kw in
                ["无法确定", "没有足够", "未提及", "无法回答", "文档中没有", "未提供", "未包含"]
            )
            if refused:
                hallucination_hits += 1
            status = "REFUSED" if refused else "HALLUCINATED"
        else:
            recall_total += 1
            if hit:
                recall_hits += 1
            status = "HIT" if hit else "MISS"

        print(f"【{status}】{question}")
        print(f"  期望章节: {expected}")
        print(f"  检索章节: {retrieved_sections}")
        print(f"  耗时: {elapsed}s\n")

        rows.append({
            "question": question,
            "expected_sections": "|".join(expected) if expected else "(none)",
            "retrieved_sections": "|".join(retrieved_sections),
            "hit": hit,
            "status": status,
            "elapsed_s": elapsed,
            "answer_preview": answer[:120].replace("\n", " ")
        })

    with open(RESULT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    recall_at_5 = recall_hits / recall_total if recall_total else 0.0
    refusal_rate = hallucination_hits / hallucination_cases if hallucination_cases else 0.0

    summary = (
        f"===== 评估汇总 =====\n"
        f"总问题数: {len(queries)}\n"
        f"Recall@5 问题数: {recall_total}\n"
        f"Recall@5 命中数: {recall_hits}\n"
        f"Recall@5: {recall_at_5:.2f}\n"
        f"\n幻觉测试问题数: {hallucination_cases}\n"
        f"正确拒答数: {hallucination_hits}\n"
        f"拒答率: {refusal_rate:.2f}\n"
    )
    print(summary)

    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write(summary)
        f.write("\n\n明细:\n")
        for r in rows:
            f.write(f"[{r['status']}] {r['question']}\n")
            f.write(f"  期望: {r['expected_sections']}\n")
            f.write(f"  检索: {r['retrieved_sections']}\n\n")

    print(f"结果已保存至: {RESULT_CSV} 和 {RESULT_TXT}")


if __name__ == "__main__":
    evaluate()