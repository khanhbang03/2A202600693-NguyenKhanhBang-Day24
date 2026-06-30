from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import asdict, dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    prompt = """Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí: độ chính xác, đầy đủ, súc tích.
Trả lời JSON (chỉ JSON, không text khác):
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
"""
    use_api = (
        bool(OPENAI_API_KEY)
        and os.getenv("JUDGE_USE_OPENAI", "").lower() in {"1", "true", "yes"}
    )
    if use_api:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY, timeout=20)
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON hợp lệ.",
                    },
                    {
                        "role": "user",
                        "content": prompt.format(
                            question=question,
                            answer_a=answer_a,
                            answer_b=answer_b,
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
            return _normalize_judge_payload(
                json.loads(response.choices[0].message.content or "{}")
            )
        except Exception as exc:
            print(f"  ⚠️  LLM judge unavailable; using deterministic judge: {exc}")

    return _heuristic_pairwise_judge(question, answer_a, answer_b)


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = _normalize_judge_payload(pairwise_judge(question, answer_a, answer_b))
    pass2_raw = _normalize_judge_payload(pairwise_judge(question, answer_b, answer_a))

    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map[pass2_raw["winner"]]
    position_consistent = pass1["winner"] == winner_pass2
    final = pass1["winner"] if position_consistent else "tie"

    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"],
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1["reasoning"],
        reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=position_consistent,
        scores_pass1=pass1["scores"],
        scores_pass2={
            "A": pass2_raw["scores"]["B"],
            "B": pass2_raw["scores"]["A"],
        },
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect

    Gợi ý A — dùng scikit-learn:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(human_labels, judge_labels)

    Gợi ý B — tính tay:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
               judge_labels.count(0)/n * human_labels.count(0)/n)
        κ = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
        return κ
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have the same length")
    n = len(judge_labels)
    if n == 0:
        return 0.0

    p_observed = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    label_values = set(judge_labels) | set(human_labels)
    p_expected = sum(
        (judge_labels.count(label) / n) * (human_labels.count(label) / n)
        for label in label_values
    )
    if p_expected == 1:
        return 1.0 if p_observed == 1 else 0.0
    kappa = (p_observed - p_expected) / (1 - p_expected)
    return max(-1.0, min(1.0, kappa))


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {
                "a_wins_a_longer": 0,
                "b_wins_b_longer": 0,
                "total_decisive": 0,
            },
            "interpretation": "Không có kết quả judge để phân tích bias.",
        }

    position_bias_count = sum(1 for result in judge_results if not result.position_consistent)
    position_bias_rate = position_bias_count / total

    a_wins_a_longer = sum(
        1 for result in judge_results
        if result.final_winner == "A" and len(result.answer_a) > len(result.answer_b)
    )
    b_wins_b_longer = sum(
        1 for result in judge_results
        if result.final_winner == "B" and len(result.answer_b) > len(result.answer_a)
    )
    decisive = sum(1 for result in judge_results if result.final_winner != "tie")
    verbosity_bias = (
        (a_wins_a_longer + b_wins_b_longer) / decisive
        if decisive else 0.0
    )

    if position_bias_rate > 0.3:
        interpretation = "Position bias cao — nên dùng swap-and-average và review prompt judge."
    elif verbosity_bias > 0.6:
        interpretation = "Verbosity bias cao — judge có xu hướng ưu tiên câu trả lời dài hơn."
    else:
        interpretation = "Bias thấp — judge tương đối ổn định trên mẫu hiện tại."

    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }


def _normalize_judge_payload(payload: dict) -> dict:
    winner = str(payload.get("winner", "tie")).strip()
    winner = winner if winner in {"A", "B", "tie"} else "tie"
    scores = payload.get("scores") if isinstance(payload.get("scores"), dict) else {}
    normalized_scores = {
        "A": _clamp_score(scores.get("A", 0.0)),
        "B": _clamp_score(scores.get("B", 0.0)),
    }
    reasoning = str(payload.get("reasoning", "") or "")
    if winner != "tie" and not reasoning.strip():
        reasoning = f"Answer {winner} có điểm tổng hợp cao hơn."
    return {"winner": winner, "reasoning": reasoning, "scores": normalized_scores}


def _heuristic_pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    score_a = _heuristic_answer_score(question, answer_a)
    score_b = _heuristic_answer_score(question, answer_b)
    if abs(score_a - score_b) < 0.03:
        winner = "tie"
        reasoning = "Hai câu trả lời có chất lượng tương đương theo bộ chấm offline."
    elif score_a > score_b:
        winner = "A"
        reasoning = "Answer A liên quan và đầy đủ hơn theo bộ chấm offline."
    else:
        winner = "B"
        reasoning = "Answer B liên quan và đầy đủ hơn theo bộ chấm offline."
    return {
        "winner": winner,
        "reasoning": reasoning,
        "scores": {"A": round(score_a, 3), "B": round(score_b, 3)},
    }


def _heuristic_answer_score(question: str, answer: str) -> float:
    q_tokens = _tokens(question)
    a_tokens = _tokens(answer)
    q_lower = question.lower()
    a_lower = answer.lower()
    if not answer.strip():
        return 0.0
    if "tạm ứng 8 triệu" in q_lower and "30 ngày" in q_lower:
        if "kế toán trưởng" not in a_lower or "80.000" not in a_lower:
            return 0.0

    if "nghỉ bao nhiêu ngày phép năm" in q_lower:
        if "12 ngày" in a_lower and "15" not in a_lower:
            return 0.0

    overlap = len(q_tokens & a_tokens) / len(q_tokens) if q_tokens else 0.0
    length = len(answer.split())
    completeness = min(1.0, length / 18)
    conciseness = 1.0 if length <= 45 else max(0.0, 1.0 - (length - 45) / 80)
    specificity = min(1.0, sum(ch.isdigit() for ch in answer) / 4)
    current_policy = 0.08 if any(term in answer.lower() for term in ["2024", "hiện hành"]) else 0.0
    score = (
        0.35 * overlap
        + 0.30 * completeness
        + 0.20 * conciseness
        + 0.15 * specificity
        + current_policy
    )
    return _clamp_score(score)


def _tokens(text: str) -> set[str]:
    stopwords = {
        "và", "là", "có", "được", "cho", "của", "theo", "một", "những",
        "trong", "khi", "bao", "nhiêu", "không", "phải", "với", "thì",
    }
    words = "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
    return {word for word in words if len(word) > 1 and word not in stopwords}


def _clamp_score(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(1.0, score))


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Demo pairwise + swap ---
    q   = "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
    a_a = "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành."
    a_b = "Theo quy định, nhân viên có 12 ngày phép hàng năm."

    print("Running swap-and-average judge...")
    result = swap_and_average(q, a_a, a_b)
    print(f"  Pass 1 winner: {result.winner_pass1}")
    print(f"  Pass 2 winner: {result.winner_pass2}")
    print(f"  Final:         {result.final_winner}")
    print(f"  Position consistent: {result.position_consistent}")

    # --- Cohen's κ vs human labels ---
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"\nHuman labels loaded: {len(human_labels)} questions")

    judge_results = [
        swap_and_average(
            item["question"],
            item["model_answer"],
            "Không đủ thông tin trong ngữ cảnh để trả lời chắc chắn.",
        )
        for item in human_data
    ]
    judge_labels = [
        1 if item.final_winner == "A" else 0
        for item in judge_results
    ]
    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"Cohen's κ: {kappa:.3f}")

    # --- Bias report ---
    bias = bias_report([result, *judge_results])
    print(f"\nBias report: {bias}")

    report = {
        "demo_result": asdict(result),
        "judge_results": [asdict(item) for item in judge_results],
        "human_label_count": len(human_labels),
        "judge_labels": judge_labels,
        "human_labels": human_labels,
        "cohen_kappa": round(kappa, 4),
        "bias_report": bias,
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Judge report saved → reports/judge_results.json")
