from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os, sys, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


def _openai_enabled() -> bool:
    return bool(OPENAI_API_KEY) and os.getenv("RAG_USE_OPENAI", "").lower() in {
        "1", "true", "yes",
    }


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if _openai_enabled():
        try:
            from openai import OpenAI
            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"  ⚠️  OpenAI summarize failed: {exc}")

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if sentence.strip()
    ]
    return " ".join(sentences[:2]) if sentences else text.strip()


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if n_questions <= 0:
        return []
    if _openai_enabled():
        try:
            from openai import OpenAI
            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể "
                            "trả lời. Trả về mỗi câu hỏi trên một dòng."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0,
            )
            raw = (response.choices[0].message.content or "").splitlines()
            questions = [_clean_question(line) for line in raw if line.strip()]
            return questions[:n_questions]
        except Exception as exc:
            print(f"  ⚠️  OpenAI HyQA failed: {exc}")

    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?\n]+", text)
        if len(sentence.strip()) > 10
    ]
    questions = []
    for sentence in sentences[:n_questions]:
        if re.search(r"\d", sentence):
            questions.append(f"Quy định hoặc hạn mức cụ thể trong nội dung này là bao nhiêu?")
        else:
            subject = " ".join(sentence.split()[:8])
            questions.append(f"Nội dung quy định gì về {subject}?")
    return questions


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    context = ""
    if _openai_enabled():
        try:
            from openai import OpenAI
            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Viết đúng một câu ngắn mô tả vị trí và chủ đề của đoạn văn "
                            "trong tài liệu."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
                    },
                ],
                max_tokens=80,
                temperature=0,
            )
            context = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"  ⚠️  OpenAI contextual failed: {exc}")
    if not context:
        title = document_title or "tài liệu nội bộ"
        context = f"Ngữ cảnh: đoạn trích từ {title}, dùng để tra cứu chính sách liên quan."
    return f"{context}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if _openai_enabled():
        try:
            from openai import OpenAI
            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            'Trả về JSON hợp lệ: {"topic":"...", "entities":["..."], '
                            '"category":"policy|hr|it|finance|safety", "language":"vi|en"}.'
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                max_tokens=150,
                temperature=0,
            )
            return _parse_json(response.choices[0].message.content or "")
        except Exception as exc:
            print(f"  ⚠️  OpenAI metadata failed: {exc}")
    return _fallback_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if _openai_enabled():
        try:
            from openai import OpenAI
            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Phân tích đoạn văn và trả về JSON hợp lệ:
{"summary":"tóm tắt 2-3 câu","questions":["câu hỏi 1","câu hỏi 2","câu hỏi 3"],
"context":"một câu mô tả vị trí/chủ đề",
"metadata":{"topic":"...","entities":["..."],
"category":"policy|hr|it|finance|safety","language":"vi|en"}}""",
                    },
                    {
                        "role": "user",
                        "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=400,
                temperature=0,
            )
            parsed = _parse_json(response.choices[0].message.content or "")
            if parsed:
                return parsed
        except Exception as exc:
            print(f"  ⚠️  Enrichment API failed: {exc}")

    # The fallback mirrors the combined schema, so callers retain enriched
    # text and metadata even when credentials or network access are absent.
    title = source or "tài liệu nội bộ"
    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text, n_questions=3),
        "context": f"Ngữ cảnh: đoạn trích từ {title}, dùng để tra cứu chính sách liên quan.",
        "metadata": _fallback_metadata(text),
    }


def _clean_question(value: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", value).strip()
    return cleaned if cleaned.endswith("?") else f"{cleaned}?"


def _parse_json(value: str) -> dict:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _fallback_metadata(text: str) -> dict:
    lowered = text.lower()
    categories = {
        "it": ("mật khẩu", "vpn", "malware", "cntt", "mfa"),
        "finance": ("lương", "chi phí", "tạm ứng", "hoàn trả", "triệu", "vnđ"),
        "hr": ("nghỉ", "nhân viên", "thử việc", "mentor", "đào tạo"),
        "safety": ("an toàn", "sơ cứu", "pccc", "sự cố"),
    }
    category = next(
        (name for name, keywords in categories.items() if any(keyword in lowered for keyword in keywords)),
        "policy",
    )
    headings = re.findall(r"^#{1,6}\s+(.+)$", text, flags=re.MULTILINE)
    topic = headings[0].strip() if headings else " ".join(text.split()[:8]).strip(" .,:;")
    entities = sorted(set(re.findall(r"\b(?:PVI|MFA|VPN|CEO|CNTT|GDPR)\b", text, re.IGNORECASE)))
    return {"topic": topic or "general", "entities": entities, "category": category, "language": "vi"}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
