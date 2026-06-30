# LLM Judge Bias Report — Phase B

**Sinh viên:** Nguyễn Khánh Bằng  
**Ngày:** 30/06/2026  
**Judge model:** gpt-4o-mini config, deterministic offline judge fallback used in this run

---

## 1. Pairwise Judge Results

Chạy `pairwise_judge()` trên 10 câu tương ứng với `human_labels_10q.json`. Answer A là model answer, Answer B là fallback "Không đủ thông tin trong ngữ cảnh để trả lời chắc chắn."

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---:|---|---|---|
| 1 | Nghỉ khi kết hôn | A | Answer A liên quan và đầy đủ hơn |
| 2 | Mua thiết bị 55 triệu | tie | Hai câu trả lời tương đương theo bộ chấm offline |
| 3 | Thưởng Tết tối thiểu | A | Answer A liên quan và đầy đủ hơn |
| 4 | Senior 9 năm: phép và lương | A | Answer A liên quan và đầy đủ hơn |
| 5 | Hoàn trả khóa học 25 triệu | A | Answer A liên quan và đầy đủ hơn |
| 6 | Tạm ứng 8 triệu quá hạn | A | Answer A liên quan và đầy đủ hơn |
| 7 | Manager 12 năm: phụ cấp và phép | A | Answer A liên quan và đầy đủ hơn |
| 8 | Số ngày phép năm | A | Answer A liên quan và đầy đủ hơn |
| 9 | Thử việc có nghỉ phép năm không | A | Answer A liên quan và đầy đủ hơn |
| 10 | Dùng VPN cá nhân khi WFH | B | Answer B liên quan và đầy đủ hơn |

---

## 2. Swap-and-Average Results

Chạy `swap_and_average()` trên cùng 10 cặp. Pass 2 đã được convert về lại không gian A/B gốc.

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---:|---|---|---|---|
| 1 | A | A | A | Yes |
| 2 | tie | tie | tie | Yes |
| 3 | A | A | A | Yes |
| 4 | A | A | A | Yes |
| 5 | A | A | A | Yes |
| 6 | A | A | A | Yes |
| 7 | A | A | A | Yes |
| 8 | A | A | A | Yes |
| 9 | A | A | A | Yes |
| 10 | B | B | B | Yes |

**Position bias rate:** 0.0% (= 0 case NOT consistent / 11 total judged cases, tính cả demo trong `reports/judge_results.json`)

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 6 label=1, 4 label=0)  
**Judge labels:** `[1, 0, 1, 1, 1, 1, 1, 1, 1, 0]`

| Question ID | Human Label | Judge Label | Agree? |
|---:|---:|---:|---|
| 1 | 1 | 1 | Yes |
| 5 | 0 | 0 | Yes |
| 12 | 1 | 1 | Yes |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 1 | Yes |
| 29 | 0 | 1 | No |
| 33 | 1 | 1 | Yes |
| 41 | 0 | 1 | No |
| 46 | 1 | 1 | Yes |
| 50 | 0 | 0 | Yes |

**Cohen's κ:** 0.5455  
**Interpretation:** moderate agreement

Kết quả có 8/10 câu đồng thuận với human label. Hai case lệch là câu 29 (tạm ứng 8 triệu) và câu 41 (phép năm theo version cũ), đều là các câu cần hiểu điều kiện/phiên bản chính xác nên judge offline lexical còn hơi dễ dãi với câu trả lời có overlap cao.

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):

- A thắng + A dài hơn B: 4 / 10 decisive cases
- B thắng + B dài hơn A: 1 / 10 decisive cases
- **Verbosity bias rate:** 50.0%

**Kết luận:** Verbosity bias ở mức trung bình, chưa vượt ngưỡng đáng lo 60% nhưng vẫn cần theo dõi. Judge có xu hướng chọn câu trả lời có nhiều token hơn khi câu trả lời đó cũng có nhiều overlap với câu hỏi, điều này có thể gây vấn đề trong production vì câu dài không đồng nghĩa với câu đúng. Nên giữ swap-and-average, thêm tiêu chí factuality rõ hơn, và chấm dựa trên reference/ground truth thay vì chỉ so sánh độ đầy đủ bề mặt.

---

## 5. Nhận xét chung

κ = 0.5455 chưa vượt 0.6, nên LLM judge trong lần chạy này đạt mức moderate nhưng chưa đủ tin cậy để tự động quyết định chất lượng production một mình. Position bias không đáng lo trong mẫu hiện tại vì cả 11/11 cases đều consistent sau khi swap. Swap-and-average vẫn hữu ích vì nó tạo cơ chế phát hiện bất ổn nếu judge đổi quyết định khi đảo vị trí answer. Trong production, nên dùng judge như một tín hiệu bổ sung trong eval pipeline, kết hợp với RAGAS, human spot-check, golden set có ground truth rõ ràng, và logging các case judge/human disagreement để cải thiện prompt chấm điểm.
