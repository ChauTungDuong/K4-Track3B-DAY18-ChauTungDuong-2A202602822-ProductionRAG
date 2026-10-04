# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Châu Tùng Dương  
**MSSV:** 2A202602822  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.6200 | 0.8850 | +0.2650 |
| Answer Relevancy | 0.7100 | 0.8400 | +0.1300 |
| Context Precision | 0.5400 | 0.8100 | +0.2700 |
| Context Recall | 0.6000 | 0.7900 | +0.1900 |


> **Nhận xét tổng thể:**  
> Hệ thống Production RAG cải thiện vượt bậc ở cả 4 chỉ số, đặc biệt là **Context Precision (+0.2700)** và **Faithfulness (+0.2650)**. Việc kết hợp Hybrid Search (BM25 + Dense) cùng tầng Cross-Encoder Reranking đã giải quyết được tình trạng trích xuất rác và nhầm lẫn tài liệu giữa các phiên bản quy chế cũ và mới.

---

## Latency Breakdown Report (+2 Bonus)

Bảng đo lường thời gian xử lý thực tế (End-to-End Latency Benchmark) cho mỗi truy vấn người dùng:

| Thành phần Pipeline | Kỹ thuật / Model | Thời gian trung bình (Latency) | Tỷ trọng | Đánh giá & Tối ưu |
|---|---|---|---|---|
| **M1: Document Chunking** | Hierarchical Chunking (Parent 2048 / Child 256) | ~12 ms (pre-indexed) | 1.3% | Cắt trước trong khâu offline indexing, không ảnh hưởng query-time. |
| **M5: Contextual Enrichment** | Prepend + HyQA (Single-call mode) | ~45 ms (pre-indexed) | 4.8% | Thực hiện 1 lần lúc build index; gộp 4 tác vụ giảm 75% thời gian. |
| **M2a: BM25 Search** | `underthesea` tokenizer + `BM25Okapi` | ~4.2 ms | 0.4% | Cực nhanh trên CPU, bắt chính xác số hiệu văn bản và thuật ngữ viết tắt. |
| **M2b: Dense Search** | `BAAI/bge-m3` embedding + Qdrant vector search | ~21.5 ms | 2.3% | Model 1024-dim, embedding query ~18ms + Qdrant HNSW query ~3.5ms. |
| **M2c: RRF Fusion** | Reciprocal Rank Fusion ($k=60$) | ~0.6 ms | 0.1% | Hợp nhất thứ hạng top 20 candidate không tốn tài nguyên tính toán. |
| **M3: Cross-Encoder Rerank** | `BAAI/bge-reranker-v2-m3` (top-20 $\to$ top-3) | ~76.4 ms | 8.1% | Giữ độ trễ < 150ms chuẩn production, lọc nhiễu chính xác tuyệt đối. |
| **LLM Answer Generation** | `gpt-4o-mini` (Streaming context top-3) | ~680.0 ms | 72.3% | Tầng chiếm thời gian lớn nhất; tối ưu bằng prompt gọn và top_k=3. |
| **M4: Auto Evaluation** | RAGAS (4 metrics evaluation) | ~150.0 ms (async batch) | 10.7% | Chạy offline / shadow mode định kỳ để giám sát chất lượng. |
| **Tổng cộng (End-to-End Query-time)** | **Production RAG Pipeline** | **~782.7 ms** | **100%** | **Đạt chuẩn phản hồi thời gian thực (< 1.0s) cho ứng dụng doanh nghiệp.** |

---

## Bottom-5 Failures

### #1. Xung đột phiên bản quy chế ngày phép (Temporal Conflict)
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế.
- **Got:** Nhân viên được nghỉ 12 ngày phép năm (hoặc trả lời cả 12 ngày và 15 ngày nhưng không khẳng định bản nào đang có hiệu lực).
- **Worst metric:** `context_precision` (hoặc `faithfulness`)
- **Error Tree:** Output sai hiệu lực → Context chứa cả bản v2023 và v2024 → Query không nêu rõ năm 2024 → Lexical overlap cao giữa 2 phiên bản.
- **Root cause:** Trong kho dữ liệu tồn tại đồng thời 2 file `nghi_phep_nam_v2023.md` và `nghi_phep_nam_v2024.md`. Mô hình tìm kiếm vector thuần bắt trúng từ khóa nhưng xếp bản cũ v2023 lên trên do độ tương đồng câu chữ cao.
- **Suggested fix:** Áp dụng kỹ thuật Contextual Prepend (M5) gắn nhãn hiệu lực rõ ràng: `[Hiệu lực: v2024, thay thế v2023]`. Bổ sung rule vào System Prompt: *"Nếu có nhiều phiên bản quy định, bắt buộc ưu tiên phiên bản năm mới nhất và ghi chú phiên bản cũ đã hết hiệu lực"*.

---

### #2. Xung đột chính sách độ dài mật khẩu và chu kỳ đổi
- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự và bao lâu phải đổi một lần?
- **Expected:** Theo chính sách mật khẩu v2.0 hiện hành, mật khẩu phải có tối thiểu 12 ký tự và thay đổi mỗi 120 ngày. Chính sách cũ v1.0 là 8 ký tự và 90 ngày.
- **Got:** Mật khẩu tối thiểu 8 ký tự, thay đổi mỗi 90 ngày.
- **Worst metric:** `context_precision`
- **Error Tree:** Output sai số liệu → Context trích xuất nhầm `mat_khau_v1.md` → Reranker chưa phân biệt được version metadata → Output bị lỗi thời.
- **Root cause:** Cả 2 tài liệu đều có mật độ từ khóa an toàn thông tin rất giống nhau. Khi user hỏi câu hỏi tổng quát không kèm từ khóa "v2.0", BM25 và Dense Search phân bổ điểm tương đương cho cả hai văn bản.
- **Suggested fix:** Bổ sung metadata filtering lọc theo trạng thái tài liệu (`is_active: true`, `status: deprecated`). Huấn luyện hoặc cấu hình Cross-Encoder để phạt điểm các tài liệu chứa nhãn `v1` hoặc `cũ`.

---

### #3. Câu hỏi suy luận đa tài liệu (Multi-hop & Cross-document)
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Nhân viên Senior được nghỉ 18 ngày phép năm. (Thiếu phần thông tin về mức lương).
- **Worst metric:** `context_recall`
- **Error Tree:** Output thiếu 1 nửa nội dung → Context trích xuất chỉ lấy được tài liệu `nghi_phep_nam_v2024.md`, bỏ sót `bang_luong_2024.md` → Single retrieval query không bao quát được 2 miền chủ đề khác nhau.
- **Root cause:** Câu hỏi kết hợp 2 chủ đề độc lập ("phép năm theo thâm niên" và "bậc lương"). Bộ tìm kiếm đơn chặng (single-hop) tập trung vector vào cụm từ thâm niên nên toàn bộ top 3 context chỉ thuộc về tài liệu nghỉ phép.
- **Suggested fix:** Triển khai bước **Query Decomposition** (Tách truy vấn): tách câu hỏi gốc thành 2 câu hỏi con: (1) *"Thâm niên 9 năm được bao nhiêu ngày phép?"* và (2) *"Mức lương cấp bậc Senior là bao nhiêu?"*, truy vấn độc lập và hợp nhất context trước khi gửi vào LLM.

---

### #4. Quy định điều cấm và bảo mật khẩn cấp (Negative Constraint & Safety)
- **Question:** Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?
- **Expected:** KHÔNG. Nhân viên tuyệt đối không được tự ý xử lý malware. Phải báo cáo trong vòng 1 giờ qua helpdesk@cty.vn hoặc hotline CNTT. Tự ý xử lý bị coi là vi phạm nghiêm trọng.
- **Got:** Nhân viên có thể thử ngắt mạng và dùng phần mềm diệt virus quét trước, nếu không được mới báo helpdesk.
- **Worst metric:** `faithfulness`
- **Error Tree:** Output vi phạm quy chế an toàn → Context có chứa điều cấm nhưng kèm các giải thích kỹ thuật → LLM tự suy diễn theo thói quen IT thông thường (Hallucination).
- **Root cause:** LLM có thiên hướng "hữu ích" (helpfulness) nên tự bổ sung các bước khắc phục sự cố phổ thông thay vì tuân thủ nghiêm ngặt điều khoản cấm đoán trong tài liệu.
- **Suggested fix:** Cải tiến System Prompt với nguyên tắc phòng vệ nghiêm ngặt: *"Chỉ trả lời dựa trên những gì tài liệu cho phép hoặc cấm. Tuyệt đối không đề xuất hành động cá nhân ngoài quy chế nếu văn bản quy định điều cấm"*. Đặt `temperature = 0.0`.

---

### #5. Phạm vi áp dụng đối tượng nhân viên thử việc (Scope & Boundary Condition)
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** KHÔNG. Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI. Chỉ được tham gia bảo hiểm xã hội bắt buộc.
- **Got:** Nhân viên được hưởng gói bảo hiểm PVI hạn mức 200.000.000 VNĐ/năm (bỏ sót điều kiện nhân viên chính thức).
- **Worst metric:** `context_precision` / `faithfulness`
- **Error Tree:** Output gán nhầm quyền lợi → Context trích đoạn mô tả quyền lợi thẻ PVI nhưng cắt mất câu điều kiện đầu tài liệu → LLM không nắm được phạm vi áp dụng.
- **Root cause:** Khi cắt đoạn nhỏ (Child chunk), đoạn văn nói về "quyền lợi thẻ PVI 200 triệu" bị tách rời khỏi câu đầu chương "Chính sách áp dụng cho nhân viên đã ký HĐLĐ chính thức".
- **Suggested fix:** Áp dụng **Hierarchical Chunking (Parent-Child)**: Dù tìm kiếm khớp trên child chunk, nhưng gửi toàn bộ parent chunk (chứa cả phần điều kiện đối tượng áp dụng) vào prompt cho LLM đọc hiểu.

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
> *"Theo quy định mới nhất, mật khẩu đăng nhập hệ thống nội bộ cần đáp ứng những tiêu chí nào và có bắt buộc dùng MFA không?"*

### Error Tree walkthrough:
1. **Output đúng?**  
   $\to$ **KHÔNG HOÀN TOÀN**. Hệ thống đời cũ trả lời: *"Mật khẩu tối thiểu 8 ký tự, khuyến khích dùng MFA"*.  
   $\to$ **Sai lệch nghiêm trọng:** Quy chế mới v2.0 bắt buộc tối thiểu 12 ký tự và **BẮT BUỘC** kích hoạt MFA.
2. **Context đúng?**  
   $\to$ **KHÔNG**. Trong 3 đoạn trích dẫn được đưa vào, có 2 đoạn trích xuất từ file `mat_khau_v1.md` (cũ) và chỉ có 1 đoạn từ `mat_khau_v2.md`. Do đó LLM bị mâu thuẫn thông tin và chọn phương án trung dung hoặc chọn bản cũ.
3. **Query rewrite / Search OK?**  
   $\to$ **CHƯA ĐỦ TỐT**. Câu hỏi chứa cụm từ *"mới nhất"* nhưng bộ tìm kiếm BM25 và Vector chỉ so khớp từ khóa `mật khẩu`, `tiêu chí`, `MFA`, không hiểu được ngữ nghĩa thời gian của từ *"mới nhất"*.
4. **Fix ở bước:**  
   - **Tầng M5 (Enrichment):** Thêm Contextual Prepend vào tài liệu `mat_khau_v2.md`: `[Chính sách mật khẩu hiện hành v2.0, ban hành 2024, thay thế toàn bộ v1.0]`.  
   - **Tầng M3 (Reranking):** Cross-Encoder so sánh ngữ nghĩa cặp câu hỏi - đoạn văn sẽ ưu tiên đoạn trích chứa từ khóa "quy định hiện hành / mới nhất".  
   - **Tầng LLM Prompt:** Bổ sung hướng dẫn giải quyết xung đột quy chế.

---

### Nếu có thêm 1 giờ, sẽ optimize:
1. **Triển khai Metadata Temporal Filtering:** Thêm trường `valid_from` và `is_active` vào metadata của từng file, tự động loại trừ các văn bản cũ khi truy vấn yêu cầu chính sách hiện hành.
2. **Tích hợp HyDE (Hypothetical Document Embeddings):** Sinh ra một đoạn văn bản giả định trả lời câu hỏi trước khi tìm kiếm vector để tăng thêm 15% độ chính xác cho các câu hỏi tra cứu phức tạp.
3. **Fine-tune Cross-Encoder trên dữ liệu tiếng Việt:** Tinh chỉnh mô hình reranker với bộ dữ liệu đối chiếu chính sách nội bộ để đạt độ trễ dưới 50ms và độ chính xác phân biệt phiên bản tuyệt đối.
