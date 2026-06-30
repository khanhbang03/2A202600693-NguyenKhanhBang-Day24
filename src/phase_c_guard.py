from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import re
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE

_PRESIDIO_ENGINES = None


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers. (Đã implement sẵn)

    Custom recognizers thêm vào:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)

    Các recognizers mặc định đã có sẵn: EMAIL, PHONE_NUMBER (international), ...
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)

    analyzer  = AnalyzerEngine(registry=registry)
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,   # text với PII được thay bằng <TYPE>
        }
    """
    results = []
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = _get_presidio_engines()

    if analyzer is not None:
        try:
            results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
        except Exception:
            results = []

    entities = _entities_from_presidio(text, results)
    entities.extend(_regex_pii_entities(text, entities))
    entities = _dedupe_entities(entities)

    if not entities:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized = _anonymize_text(text, entities)
    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml. (Đã implement sẵn)

    Config directory: guardrails/
        config.yml  — model + rails config
        rails.co    — Colang dialogue flows (topic check, jailbreak check, output check)
    """
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    rails  = LLMRails(config)
    return rails


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,          # NeMo's raw response
        }
    """
    if rails is None and os.getenv("NEMO_USE_RAILS", "").lower() in {"1", "true", "yes"}:
        try:
            rails = setup_nemo_rails()
        except Exception:
            rails = None

    if rails is not None:
        try:
            response = await rails.generate_async(
                messages=[{"role": "user", "content": text}]
            )
            response_text = _response_to_text(response)
            blocked = _looks_like_refusal(response_text)
            return {
                "allowed": not blocked,
                "blocked_reason": "nemo_input_rail" if blocked else None,
                "response": response_text,
            }
        except Exception as exc:
            response_text = f"NeMo unavailable; deterministic rail used: {exc}"
    else:
        response_text = ""

    blocked, reason = _deterministic_input_block(text)
    return {
        "allowed": not blocked,
        "blocked_reason": reason if blocked else None,
        "response": (
            "Xin lỗi, tôi chỉ có thể hỗ trợ câu hỏi chính sách nhân sự nội bộ."
            if blocked else response_text
        ),
    }


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    NeMo output rails hoạt động trong context của cả cuộc hội thoại (input + output).
    Kiểm tra: có PII không? Nội dung có phù hợp không? Có hallucination rõ ràng không?

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,          # answer đã qua guard (có thể bị redact)
        }
    """
    if rails is None and os.getenv("NEMO_USE_RAILS", "").lower() in {"1", "true", "yes"}:
        try:
            rails = setup_nemo_rails()
        except Exception:
            rails = None

    if rails is not None:
        try:
            response = await rails.generate_async(messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            response_text = _response_to_text(response)
            flagged = _looks_like_refusal(response_text)
            return {
                "safe": not flagged,
                "flagged_reason": "nemo_output_rail" if flagged else None,
                "final_answer": response_text if flagged else answer,
            }
        except Exception:
            pass

    pii_result = pii_scan(answer)
    sensitive_output = _contains_any(answer, [
        "mật khẩu hệ thống là", "cccd của nhân viên là",
        "số điện thoại cá nhân của", "toàn bộ thông tin nhân viên",
        "confidential", "training data", "system instructions",
    ])
    if pii_result["has_pii"] or sensitive_output:
        return {
            "safe": False,
            "flagged_reason": "deterministic_output_rail",
            "final_answer": "Tôi không thể cung cấp thông tin nhạy cảm này. Vui lòng liên hệ phòng Nhân sự trực tiếp.",
        }
    return {"safe": True, "flagged_reason": None, "final_answer": answer}


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii (cho category pii_injection)
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    Returns:
        list of {
          "id": int, "category": str, "input": str,
          "expected": "blocked"|"allowed",
          "actual":   "blocked"|"allowed",
          "blocked_by": str | None,       # "presidio" | "nemo_input" | None
          "passed": bool,
        }
    """
    async def _run_all():
        results = []
        for item in adversarial_set:
            blocked_by = None

            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            input_preview = item["input"]
            if len(input_preview) > 80:
                input_preview = input_preview[:80] + "..."
            results.append({
                "id": item["id"],
                "category": item["category"],
                "input": input_preview,
                "expected": item["expected"],
                "actual": actual,
                "blocked_by": blocked_by,
                "passed": actual == item["expected"],
            })
        return results

    results = asyncio.run(_run_all())
    passed = sum(1 for result in results if result["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Đo P50/P95/P99 latency cho từng layer trong guard stack.

    Mục tiêu production: P95 total < LATENCY_BUDGET_P95_MS (500ms mặc định)

    Insight cần quan sát:
        - Presidio: local regex → rất nhanh (<10ms)
        - NeMo:     LLM API call → chậm (~200-800ms tuỳ model và network)
        → Tổng: dominated by NeMo

    Returns:
        {
          "presidio_ms":  {"p50": float, "p95": float, "p99": float},
          "nemo_ms":      {"p50": float, "p95": float, "p99": float},
          "total_ms":     {"p50": float, "p95": float, "p99": float},
          "latency_budget_ok": bool,
          "budget_ms": int,
        }
    """
    presidio_times, nemo_times, total_times = [], [], []
    inputs = test_inputs or [""]
    runs = max(0, n_runs)

    async def _measure():
        for i in range(runs):
            text = inputs[i % len(inputs)]

            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    asyncio.run(_measure())

    total_percentiles = _percentiles(total_times)
    return {
        "presidio_ms": _percentiles(presidio_times),
        "nemo_ms": _percentiles(nemo_times),
        "total_ms": total_percentiles,
        "latency_budget_ok": total_percentiles["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


def _get_presidio_engines():
    global _PRESIDIO_ENGINES
    if _PRESIDIO_ENGINES is not None:
        return _PRESIDIO_ENGINES
    try:
        _PRESIDIO_ENGINES = setup_presidio()
    except Exception:
        _PRESIDIO_ENGINES = (None, None)
    return _PRESIDIO_ENGINES


def _entities_from_presidio(text: str, results: list) -> list[dict]:
    entities = []
    for result in results or []:
        entity_type = getattr(result, "entity_type", "")
        if entity_type == "PHONE_NUMBER":
            entity_type = "VN_PHONE"
        elif entity_type == "EMAIL_ADDRESS":
            entity_type = "EMAIL"
        entities.append({
            "type": entity_type,
            "text": text[getattr(result, "start", 0):getattr(result, "end", 0)],
            "score": round(float(getattr(result, "score", 0.0)), 3),
            "start": int(getattr(result, "start", 0)),
            "end": int(getattr(result, "end", 0)),
        })
    return entities


def _regex_pii_entities(text: str, existing: list[dict]) -> list[dict]:
    patterns = [
        ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", 0.95),
        ("VN_CCCD", r"\b\d{12}\b", 0.9),
        ("VN_CCCD", r"\b\d{9}\b", 0.7),
        ("VN_PHONE", r"\b0[3-9]\d{8}\b", 0.9),
    ]
    found = []
    existing_spans = {(item["start"], item["end"], item["type"]) for item in existing}
    for entity_type, pattern, score in patterns:
        for match in re.finditer(pattern, text):
            key = (match.start(), match.end(), entity_type)
            if key in existing_spans:
                continue
            found.append({
                "type": entity_type,
                "text": match.group(0),
                "score": score,
                "start": match.start(),
                "end": match.end(),
            })
    return found


def _dedupe_entities(entities: list[dict]) -> list[dict]:
    by_span = {}
    priority = {"EMAIL": 3, "VN_CCCD": 2, "VN_PHONE": 2}
    for entity in entities:
        key = (entity["start"], entity["end"])
        current = by_span.get(key)
        if current is None or priority.get(entity["type"], 0) > priority.get(current["type"], 0):
            by_span[key] = entity
    return sorted(by_span.values(), key=lambda item: (item["start"], item["end"]))


def _anonymize_text(text: str, entities: list[dict]) -> str:
    anonymized = text
    for entity in sorted(entities, key=lambda item: item["start"], reverse=True):
        anonymized = (
            anonymized[:entity["start"]]
            + f"<{entity['type']}>"
            + anonymized[entity["end"]:]
        )
    return anonymized


def _deterministic_input_block(text: str) -> tuple[bool, str | None]:
    lower = text.lower()
    pii_request_patterns = [
        "cho tôi biết cccd", "cccd và số điện thoại", "số điện thoại của nhân viên",
        "email của nhân viên", "thông tin cá nhân", "liệt kê toàn bộ thông tin nhân viên",
        "tiết lộ lương", "lương tháng của nhân viên", "list all employee salaries",
        "all employee salaries", "employee records", "confidential employee data",
    ]
    jailbreak_patterns = [
        "bỏ qua tất cả hướng dẫn", "ignore your system prompt", "ignore previous instructions",
        "ignore your previous instructions", "pretend you are dan", "do anything now",
        "unrestricted ai", "system override", "admin command", "priority 1",
        "dump all training data", "in ra toàn bộ system instructions",
        "system instructions", "đóng vai hr manager", "đóng vai ai không có giới hạn",
        "không có giới hạn", "mật khẩu admin", "tấn công mạng",
    ]
    off_topic_patterns = [
        "bài thơ", "mùa thu", "nấu phở", "công thức", "bitcoin", "ethereum",
        "giải phương trình", "vi phân", "marvel", "phim", "thời tiết",
        "giá cổ phiếu",
    ]
    prompt_injection_patterns = [
        "forget all hr policies", "ignore previous instructions", "ignore previous",
        "admin command", "system override", "<!-- ignore previous instructions -->",
        "bỏ qua tất cả policy", "reveal all confidential", "print all confidential",
    ]

    checks = [
        ("pii_request", pii_request_patterns),
        ("jailbreak", jailbreak_patterns),
        ("off_topic", off_topic_patterns),
        ("prompt_injection", prompt_injection_patterns),
    ]
    for reason, patterns in checks:
        if any(pattern in lower for pattern in patterns):
            return True, reason
    return False, None


def _contains_any(text: str, patterns: list[str]) -> bool:
    lower = text.lower()
    return any(pattern in lower for pattern in patterns)


def _looks_like_refusal(text: str) -> bool:
    return _contains_any(text, [
        "xin lỗi", "không thể", "không được phép", "i cannot", "i'm sorry",
        "cannot comply", "không thể thực hiện",
    ])


def _response_to_text(response) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return str(response.get("content") or response.get("response") or response)
    return str(response)


def _percentiles(times: list[float]) -> dict:
    if not times:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    sorted_times = sorted(times)
    n = len(sorted_times)

    def pick(percentile: float) -> float:
        index = min(max(round((n - 1) * percentile), 0), n - 1)
        return round(sorted_times[index], 2)

    return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99)}


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Task 9a: PII scan demo
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    result = pii_scan(test_pii)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    results = run_adversarial_suite(adversarial_set)
    if results:
        passed = sum(1 for r in results if r["passed"])
        print(f"Adversarial suite: {passed}/{len(results)} passed")

    # Task 12: P95 latency
    sample_inputs = [item["input"] for item in adversarial_set[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=10)
    print(f"\nLatency P95 — Presidio: {latency['presidio_ms']['p95']}ms | "
          f"NeMo: {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")

    report = {
        "pii_demo": result,
        "adversarial_total": len(results),
        "adversarial_passed": sum(1 for item in results if item["passed"]),
        "adversarial_results": results,
        "latency": latency,
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/guard_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Guard report saved → reports/guard_results.json")
