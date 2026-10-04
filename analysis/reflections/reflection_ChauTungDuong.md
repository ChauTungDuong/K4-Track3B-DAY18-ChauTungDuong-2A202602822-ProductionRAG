# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Châu Tùng Dương  
**MSSV:** 2A202602822  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Qua quá trình thực hành trực tiếp và quan sát luồng dữ liệu chạy qua từng module, mình đúc kết bảng đối chiếu giữa các khái niệm lý thuyết trên lớp và các đoạn code cụ thể đã triển khai:

| Lecture Concept | Module | Hàm cụ thể | Trải nghiệm & Quan sát thực tế khi chạy pipeline |
|---|---|---|---|
| **Semantic Chunking** | M1 | `chunk_semantic()` | Khi tách câu bằng regex rồi tính cosine similarity giữa các vector nhúng (dùng `all-MiniLM-L6-v2`), ngưỡng `0.85` hoạt động khá mượt. Điểm mình thấy tâm đắc nhất là các câu trong cùng một điều khoản quy chế được gom liền mạch, không còn bị tình trạng chém ngang câu hay ngắt dòng thô thiển như naive chunking (`\n\n`). |
| **Hierarchical Chunking (Parent-Child)** | M1 | `chunk_hierarchical()` | Cắt đoạn phân cấp giải quyết đúng bài toán "tìm kiếm thì cần ngắn, hiểu thì cần rộng". Mình chia parent 2048 ký tự và child 256 ký tự, gán chặt `parent_id`. Khi chạy kiểm thử, child chunk giúp vector search bắt cực nhạy các ý nhỏ, nhưng lúc tổng hợp trả lời, việc đưa parent chunk vào prompt giúp LLM nắm trọn vẹn điều kiện áp dụng mà không bị cụt ý. |
| **Structure-Aware Chunking** | M1 | `chunk_structure_aware()` | Phân tích tài liệu dựa trên Markdown header (`#`, `##`) giữ được trọn vẹn từng bảng biểu và danh sách gạch đầu dòng. Việc đưa tên heading vào trường `metadata["section"]` cực kỳ hữu ích cho khâu lọc và định vị tài liệu sau này. |
| **Vietnamese Word Segmentation** | M2 | `segment_vietnamese()` | Xử lý từ ghép tiếng Việt bằng `underthesea`. Trải nghiệm đáng nhớ nhất ở đây là việc thư viện hay nối từ bằng dấu gạch dưới (`nghỉ_phép`), nếu không chủ động `.replace("_", " ")` thì bộ từ khóa của `BM25Okapi` sẽ lệch hoàn toàn với câu gõ thông thường của người dùng. |
| **Hybrid Search & RRF Fusion** | M2 | `reciprocal_rank_fusion()` | Sự kết hợp giữa BM25 và Dense Search (`bge-m3`) qua thuật toán RRF ($k=60$) bù trừ cho nhau rất rõ. Với các câu hỏi chứa số hiệu văn bản ("v2024", "PVI", "MFA"), Dense Search đôi khi bị trôi sang các văn bản có ngữ nghĩa na ná nhau, nhưng BM25 ngay lập tức kéo đúng tài liệu chứa chính xác từ khóa lên đầu. RRF cộng điểm theo thứ hạng xếp hạng nên không lo chuyện lệch thang đo giữa hai bên. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | Ban đầu mình khá e ngại về độ trễ của Cross-Encoder khi chạy trên CPU máy cá nhân. Tuy nhiên khi benchmark thực tế lọc từ 20 candidate xuống top 3, thời gian chỉ mất khoảng ~76ms. Đoạn trích chứa thông tin mới nhất luôn được chấm điểm áp đảo và đẩy lên rank 0, loại bỏ hoàn toàn các đoạn văn nhiễu khỏi ngữ cảnh của LLM. |
| **RAGAS 4-Metrics Evaluation** | M4 | `evaluate_ragas()` | Thay vì đánh giá cảm tính bằng mắt, bộ 4 chỉ số RAGAS phản ánh bức tranh rất trực quan. Nhìn vào kết quả, Faithfulness tăng vọt từ 0.62 lên 0.885 và Context Precision tăng từ 0.54 lên 0.81 chứng minh rõ rệt giá trị của tầng Reranking và Hybrid Search so với bản Naive ban đầu. |
| **Diagnostic Tree & Failure Analysis** | M4 | `failure_analysis()` | Logic cây chẩn đoán tự động phân loại lỗi theo metric thấp nhất (như `faithfulness` thấp $\to$ hallucination, `context_precision` thấp $\to$ nhiễu tài liệu) giúp mình định hướng ngay cần can thiệp vào prompt, reranker hay bộ chunking thay vì phải mò mẫm thủ công. |
| **Contextual Prepend & HyQA Enrichment** | M5 | `_enrich_single_call()` | Kỹ thuật làm giàu văn bản trước khi index là điểm sáng lớn. Thay vì tốn 4 lần gọi API riêng lẻ, mình thiết kế một prompt duy nhất để LLM vừa tóm tắt, sinh 3 câu hỏi giả thuyết (HyQA), viết câu ngữ cảnh (Prepend) và trích xuất metadata. Cách này vừa tiết kiệm 75% chi phí API vừa rút ngắn thời gian xử lý rất nhiều. |

---

## Phần 2: Khó khăn & Quá trình Debug thực tế (Challenges & Debugging)

Trong suốt quá trình làm bài lab, mình đã đối mặt với một số lỗi kỹ thuật khá thú vị và tích lũy được nhiều kinh nghiệm xử lý thực tế:

### 1. Lỗi lệch token tiếng Việt giữa BM25 và truy vấn người dùng
- **Hiện tượng:** Khi chạy thử nghiệm tìm kiếm với câu hỏi `"Nhân viên được nghỉ bao nhiêu ngày phép năm?"`, BM25 trả về kết quả trắng hoặc điểm số bằng 0, mặc dù trong dữ liệu có hẳn văn bản quy định nghỉ phép.
- **Quá trình debug:** Mình in thử danh sách token mà `BM25Search.index()` nhận được và token sau khi hàm `search()` tách từ. Kết quả phát hiện `underthesea` khi tokenize văn bản gốc đã ghép thành token `nghỉ_phép`, `ngày_phép`, trong khi ở câu truy vấn người dùng gõ tách rời các từ, dẫn đến BM25 không khớp được từ khóa nào.
- **Cách khắc phục:** Trong hàm `segment_vietnamese()`, sau khi gọi `word_tokenize(..., format="text")`, mình thêm ngay `.replace("_", " ")`. Nhờ vậy toàn bộ từ ghép được chuẩn hóa thành dạng từ cách nhau bởi khoảng trắng thông thường, BM25 nhận diện chính xác 100%.

### 2. Lỗi deprecation khi thao tác với thư viện `qdrant-client >= 1.9`
- **Hiện tượng:** Khi gọi hàm tìm kiếm trên vector store, chương trình ném cảnh báo hoặc lỗi không tìm thấy phương thức `search()` quen thuộc của Qdrant.
- **Quá trình debug:** Kiểm tra lại phiên bản thư viện cài đặt trong môi trường (`qdrant-client 1.12+`), tra cứu release note chính thức của Qdrant thì được biết phương thức `search()` đã chuyển thành `query_points()`, cấu trúc trả về cũng đóng gói trong đối tượng `.points`.
- **Cách khắc phục:** Cập nhật lại phương thức gọi:
  ```python
  response = self.client.query_points(
      collection_name=collection_name,
      query=query_vector,
      limit=top_k
  )
  results = response.points
  ```
  Đồng thời, mình viết thêm cơ chế dự phòng tự động chuyển sang `QdrantClient(":memory:")` nếu kết nối tới Docker Qdrant cục bộ gặp sự cố.

### 3. Tránh lỗi crash thư viện `FlagEmbedding` bằng `CrossEncoder`
- **Hiện tượng:** Ban đầu thử import `FlagReranker` từ gói `FlagEmbedding` thì gặp lỗi xung đột phiên bản với `transformers` và `tokenizers` trên môi trường Python mới.
- **Quá trình debug:** Thư viện `FlagEmbedding` đóng gói khá nhiều dependency cũ gây xung đột môi trường. Bản chất mô hình `BAAI/bge-reranker-v2-m3` là một kiến trúc Cross-Encoder chuẩn của HuggingFace.
- **Cách khắc phục:** Mình chuyển thẳng sang dùng lớp `CrossEncoder` có sẵn trong thư viện `sentence_transformers`:
  ```python
  from sentence_transformers import CrossEncoder
  self.model = CrossEncoder("BAAI/bge-reranker-v2-m3")
  ```
  Cách này vừa ổn định, chạy nhanh, vừa tránh được hoàn toàn các xung đột thư viện không đáng có.

### 4. Quản lý tài nguyên và bộ nhớ khi chạy nhiều model nặng trên máy cá nhân
- **Trải nghiệm:** Cả hai mô hình `bge-m3` và `bge-reranker-v2-m3` đều có kích thước tương đối lớn (~2.2GB mỗi model). Khi chạy test liên tục, việc nạp đi nạp lại model có thể gây nghẽn RAM hoặc làm máy bị đơ.
- **Giải pháp:** Mình áp dụng cơ chế nạp lười (lazy loading) và lưu instance model trong bộ nhớ dùng chung, kết hợp bọc khối suy luận trong `with torch.inference_mode():` để tắt autograd graph của PyTorch, giúp giảm thiểu tiêu thụ RAM và tăng tốc độ suy luận đáng kể.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Tên dự án: **Hệ thống Hỏi đáp Kỹ thuật & Tra cứu Quy trình Nội bộ (Internal Engineering Wiki & SOP Assistant)**

#### 1. Hiện trạng & Bài toán thực tế
- **Kiến trúc hiện tại:** Hệ thống đang dùng Naive RAG cơ bản: cắt tài liệu theo độ dài cố định 500 ký tự và nhúng vector bằng text-embedding model thông thường.
- **Vấn đề tồn đọng:**
  - Tài liệu kỹ thuật của team chứa nhiều đoạn mã code (code blocks), bảng biểu và các bước hướng dẫn triển khai. Khi cắt thô theo số ký tự, các đoạn code bị cắt đứt gãy giữa chừng, làm mất ngữ cảnh khiến LLM trả lời sai syntax.
  - Các tài liệu hướng dẫn (SOP) có nhiều phiên bản cập nhật (ví dụ quy trình release v1.0 và v2.0). Hệ thống thường xuyên trích nhầm tài liệu cũ do câu từ tương đồng.
  - Tỷ lệ Faithfulness còn thấp vì khi context trích xuất bị thiếu thông tin quan trọng, mô hình có xu hướng tự "bịa" thêm các bước kỹ thuật không có thật trong wiki.

#### 2. Kế hoạch áp dụng các kỹ thuật từ Lab 18
1. **Chiến lược Chunking:**
   - Thay thế toàn bộ naive chunking bằng **Structure-Aware Chunking**: phân tích tài liệu theo Markdown heading, tuyệt đối không cắt ngang các khối code block ```bash / ```python hay các bảng thông số môi trường.
   - Kết hợp **Hierarchical Chunking (Parent-Child)**: lưu parent chunk bao trọn một quy trình hoàn chỉnh (2048 ký tự), child chunk đại diện cho từng bước thao tác nhỏ (256 ký tự).
2. **Nâng cấp tầng tìm kiếm (Hybrid Search):**
   - Triển khai song song **BM25 tiếng Việt** (có chuẩn hóa từ ghép) để bắt chính xác các từ khóa kỹ thuật, tên biến, đường dẫn API hoặc mã lỗi (`HTTP 502`, `ERR_CONNECTION_REFUSED`).
   - Kết hợp **Dense Search** bằng `BAAI/bge-m3` để bắt ý nghĩa câu hỏi khi developer hỏi theo cách diễn đạt tự nhiên. Hợp nhất kết quả bằng **RRF** ($k=60$).
3. **Thêm tầng lọc Cross-Encoder Reranker:**
   - Tích hợp `bge-reranker-v2-m3` để sàng lọc từ top 20 candidate xuống top 3 đoạn trích thực sự chuẩn xác nhất trước khi gửi vào LLM, giúp triệt tiêu nhiễu thông tin.
4. **Làm giàu dữ liệu (Enrichment):**
   - Áp dụng **Contextual Prepend** để gắn thông tin ngữ cảnh vào từng chunk: `[Dự án: Auth-Service | Tài liệu: Quy trình Onboarding v2.0 (Áp dụng từ 2024)]`.
   - Sinh câu hỏi giả thuyết **HyQA** để bao quát các cách hỏi khác nhau của thành viên mới khi tra cứu quy trình.
5. **Giám sát & Đánh giá chất lượng liên tục:**
   - Xây dựng bộ test set gồm 40 câu hỏi thực tế từ các issue thường gặp.
   - Đưa pipeline đánh giá **RAGAS 4 chỉ số** vào CI/CD để tự động chạy kiểm thử hồi quy mỗi khi có tài liệu wiki mới được cập nhật, đặt ngưỡng chặn: `Faithfulness >= 0.85` và `Context Precision >= 0.80`.

#### 3. Kế hoạch triển khai theo tuần (Timeline)
- **Tuần 1:** Chuẩn hóa toàn bộ corpus tài liệu Markdown trên kho Wiki nội bộ; viết script tiền xử lý với Structure-Aware Chunking và Contextual Prepend (M1, M5).
- **Tuần 2:** Cài đặt cơ sở dữ liệu vector Qdrant, hoàn thiện bộ tìm kiếm lai Hybrid Search kết hợp BM25 và Dense Search qua thuật toán RRF (M2).
- **Tuần 3:** Tích hợp tầng Cross-Encoder Reranker (M3), tinh chỉnh prompt hệ thống có kèm quy tắc chống hallucination và đo đạc độ trễ phản hồi (< 800ms).
- **Tuần 4:** Thiết lập quy trình đánh giá tự động bằng RAGAS (M4), kiểm thử trên bộ test set 40 câu hỏi nội bộ, nghiệm thu và triển khai thử nghiệm cho 1 team phát triển sản phẩm.
