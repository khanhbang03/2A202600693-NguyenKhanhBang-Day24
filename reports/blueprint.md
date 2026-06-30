# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyễn Khánh Băng  
**Ngày:** 30/06/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~0.03ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~0.02ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

Kết quả lấy từ `reports/guard_results.json` sau khi chạy `python src/phase_c_guard.py`.
Trong lần đo này, input rail chạy bằng deterministic/local fallback nên latency thấp hơn NeMo API thật.

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 0.01 | 0.03 | 0.03 | <10ms |
| NeMo Input Rail | 0.01 | 0.02 | 0.02 | <300ms |
| RAG Pipeline | N/A | N/A | N/A | <2000ms |
| NeMo Output Rail | N/A | N/A | N/A | <300ms |
| **Total Guard** | 0.02 | **0.06** | 0.06 | **<500ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** Guard stack đang đạt budget trong local deterministic mode. Khi bật NeMo API thật, bottleneck dự kiến sẽ là NeMo input/output rail; tối ưu bằng cache cho câu hỏi lặp, rule-based prefilter trước LLM rail, timeout ngắn, và fallback refusal an toàn khi model chậm.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.658 |
| Worst metric | answer_relevancy |
| Dominant failure distribution | factual |
| Cohen's κ | 0.5455 |
| Adversarial pass rate | 20 / 20 |
| Guard P95 latency | 0.06 ms |

---

## Nhận xét & Cải tiến

Guard stack hoạt động tốt trên adversarial suite: Presidio bắt được PII trực tiếp như CCCD, số điện thoại, email; input rail chặn được jailbreak, off-topic và prompt injection. Điểm yếu lớn nhất của RAG nằm ở `answer_relevancy`, nghĩa là prompt trả lời cần ép model bám sát câu hỏi hơn và tránh trả lời thiếu trọng tâm. Distribution `factual` có nhiều worst-metric cases nhất theo cluster, nên cần rà lại prompt template và cách chọn context cho các câu hỏi tra cứu đơn giản. Nếu deploy production thật, tôi sẽ bật NeMo API, đo lại latency với traffic thật, thêm cache/prefilter rule-based, logging cho blocked requests, và tạo CI gate fail nếu adversarial pass rate dưới 90% hoặc RAGAS faithfulness dưới ngưỡng.
