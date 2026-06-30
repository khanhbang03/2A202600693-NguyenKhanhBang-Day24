# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Khánh Bằng  
**Ngày:** 30/06/2026

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.9003 | 0.5283 | 0.3679 |
| answer_relevancy | 0.6982 | 0.4351 | 0.2071 |
| context_precision | 0.9074 | 0.6229 | 0.4315 |
| context_recall | 0.9652 | 0.7604 | 0.5214 |
| **avg_score** | **0.8677** | **0.5867** | **0.3820** |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---:|---|---|---:|---|
| 1 | multi_hop | Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT? | 0.1447 | faithfulness |
| 2 | multi_hop | So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0 về độ dài tối thiểu, thời hạn đổi và MFA. | 0.1682 | faithfulness |
| 3 | adversarial | Nhân viên được nghỉ bao nhiêu ngày phép năm? | 0.1738 | faithfulness |
| 4 | adversarial | Khi phát hiện malware trên máy tính công ty, nhân viên có nên tự xử lý không? | 0.1748 | faithfulness |
| 5 | multi_hop | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp hàng tháng và số ngày phép năm theo v2024 là bao nhiêu? | 0.2163 | faithfulness |
| 6 | adversarial | Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không? | 0.2330 | faithfulness |
| 7 | adversarial | Mật khẩu phải có tối thiểu bao nhiêu ký tự? | 0.2343 | faithfulness |
| 8 | adversarial | Nhân viên Manager có thể dùng VPN cá nhân (như NordVPN) khi WFH để tăng bảo mật thêm không? | 0.2787 | faithfulness |
| 9 | multi_hop | Nhân viên tạm ứng 4 triệu và một nhân viên khác tạm ứng 7 triệu: quy trình phê duyệt khác nhau thế nào? | 0.2860 | faithfulness |
| 10 | multi_hop | Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu? | 0.2975 | faithfulness |

---

## 3. Failure Cluster Matrix

Mỗi ô = số câu có `worst_metric` = row, thuộc distribution = col.

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 1 | 8 | 6 | 15 |
| answer_relevancy | 16 | 6 | 2 | 24 |
| context_precision | 3 | 5 | 2 | 10 |
| context_recall | 0 | 1 | 0 | 1 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** answer_relevancy

**Lý do phân tích:**

Distribution `factual` có số lượng failure nhiều nhất trong ma trận vì chiếm 20 câu và phần lớn worst metric của nhóm này là `answer_relevancy` (16/20). Điểm này không có nghĩa factual là nhóm có chất lượng thấp nhất: avg_score của factual vẫn cao nhất (0.8677), nhưng khi factual sai thì lỗi thường là câu trả lời chưa bám sát đúng wording hoặc ràng buộc của câu hỏi. Metric `answer_relevancy` là điểm yếu chủ đạo toàn bộ tập (24/50), cho thấy retrieval thường lấy được context đủ tốt nhưng prompt sinh câu trả lời còn thiếu trọng tâm, trả lời quá ngắn, hoặc bỏ qua điều kiện cụ thể như phiên bản chính sách, ngưỡng tiền, số ngày, đối tượng áp dụng. Với corpus HR policy tiếng Việt, nhiều câu hỏi có wording gần giống nhau nhưng khác version hoặc điều kiện, nên model dễ trả lời đúng chủ đề nhưng chưa đúng ý hỏi.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating | Ép model trích dẫn đúng evidence từ context, giảm temperature, thêm rule "không suy đoán nếu context thiếu", và yêu cầu nêu rõ phiên bản policy khi có version conflict. |
| context_recall | Missing relevant chunks | Tăng hybrid retrieval top-k, thêm query expansion cho từ đồng nghĩa HR tiếng Việt, và cải thiện chunking để không tách rời bảng/điều kiện quan trọng. |
| context_precision | Too many irrelevant chunks | Thêm reranking mạnh hơn, filter theo metadata/version/ngày hiệu lực, và ưu tiên chunk cùng tài liệu khi câu hỏi cần thông tin liên quan gần nhau. |
| answer_relevancy | Answer doesn't match question | Cải thiện prompt template: nhắc lại constraint của câu hỏi, trả lời theo schema ngắn gọn, kiểm tra đủ các phần hỏi trước khi trả lời cuối. |

---

## 6. Nhận xét về Adversarial Distribution

Adversarial là distribution yếu nhất: avg_score chỉ 0.3820, thấp hơn multi_hop (0.5867) và factual (0.8677). Trong bottom 10 có 5 câu adversarial: rank 3, 4, 6, 7, 8. Các câu này đều rơi vào `faithfulness`, cho thấy pipeline dễ bị đánh lừa bởi version conflicts (ví dụ phép năm v2023 vs v2024, password policy v1.0 vs v2.0), negation traps, hoặc câu hỏi cố tình gợi ý hành vi không được phép như tự xử lý malware hay dùng VPN cá nhân. Điều này cho thấy retrieval/generation cần nhận diện policy hiện hành và các mệnh đề phủ định tốt hơn, thay vì chỉ trả lời theo chunk có overlap cao nhất.
