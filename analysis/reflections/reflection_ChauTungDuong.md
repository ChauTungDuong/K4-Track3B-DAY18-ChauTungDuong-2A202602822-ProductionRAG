# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Châu Tùng Dương  
**MSSV:** 2A202602822  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Bảng đối chiếu giữa các khái niệm lý thuyết cốt lõi trên lớp và các hàm thực thi cụ thể trong mã nguồn của 5 modules:

| Lecture Concept | Module | Hàm cụ thể | Ghi nhận & Phân tích kỹ thuật qua thực nghiệm |
|---|---|---|---|
| **Semantic Chunking** | M1 | `chunk_semantic()` | Tách câu bằng regex và tính độ tương đồng cosine giữa các vector nhúng liền kề (`all-MiniLM-L6-v2`). Ngưỡng `0.85` phân tách đoạn tương đối chính xác theo biến chuyển nội dung. Các điều khoản liên quan được giữ liền mạch, khắc phục hiện tượng ngắt câu dở dang của phương pháp cắt thô (`\n\n`). |
| **Hierarchical Chunking (Parent-Child)** | M1 | `chunk_hierarchical()` | Cấu trúc Parent (2048 ký tự) và Child (256 ký tự) liên kết qua `parent_id`. Khâu tìm kiếm vector thực hiện trên child chunk để tối ưu độ nhạy từ khóa, nhưng khi chuyển tiếp cho LLM tạo câu trả lời thì nạp toàn bộ parent chunk, đảm bảo đầy đủ bối cảnh và điều kiện áp dụng. |
| **Structure-Aware Chunking** | M1 | `chunk_structure_aware()` | Phân tích cú pháp Markdown header (`#`, `##`, `###`) giúp bảo toàn nguyên vẹn cấu trúc bảng biểu số liệu và danh sách quy định. Tên heading được lưu vào `metadata["section"]`, tạo điều kiện lọc dữ liệu theo chương mục. |
| **Vietnamese Word Segmentation** | M2 | `segment_vietnamese()` | Tách từ ghép tiếng Việt qua `underthesea`. Do thư viện mặc định nối từ ghép bằng dấu gạch dưới (`nghỉ_phép`) trong khi người dùng nhập khoảng trắng, việc chuẩn hóa bằng `.replace("_", " ")` là cần thiết để không gian từ vựng của `BM25Okapi` khớp với câu truy vấn. |
| **Hybrid Search & RRF Fusion** | M2 | `reciprocal_rank_fusion()` | Kết hợp BM25 và Dense Search (`BAAI/bge-m3`) qua thuật toán Reciprocal Rank Fusion ($k=60$). BM25 bắt chính xác các định danh và số hiệu quy chế (như "v2024", "PVI", "MFA"), còn Dense Search bắt ý nghĩa ngữ cảnh. Thuật toán RRF xử lý gộp thứ hạng hiệu quả mà không phụ thuộc vào thang điểm thô khác nhau của hai mô hình. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | Tái xếp hạng top 20 ứng viên từ tầng Hybrid Search thành top 3 đoạn trích chính xác nhất bằng `BAAI/bge-reranker-v2-m3`. Độ trễ đo được khoảng ~76.4ms trên CPU, đáp ứng tốt yêu cầu production (< 150ms). Đoạn văn có hiệu lực mới nhất được đẩy lên rank 0, loại bỏ các đoạn gây nhiễu. |
| **RAGAS 4-Metrics Evaluation** | M4 | `evaluate_ragas()` | Đánh giá tự động toàn diện pipeline với 4 chỉ số chuẩn: Faithfulness (0.8850 vs 0.6200 của baseline), Context Precision (0.8100 vs 0.5400), Context Recall (0.7900 vs 0.6000), Answer Relevancy (0.8400 vs 0.7100). Số liệu phản ánh sự cải thiện rõ rệt về độ chính xác và khả năng kiểm soát hallucination. |
| **Diagnostic Tree & Failure Analysis** | M4 | `failure_analysis()` | Áp dụng cây chẩn đoán lỗi tự động dựa trên Worst Metric. Việc xác định metric thấp nhất giúp định tuyến nguyên nhân (ví dụ: `faithfulness` thấp do prompt/temperature, `context_precision` thấp do thiếu rerank) để đưa ra đề xuất kỹ thuật tương ứng. |
| **Contextual Prepend & HyQA Enrichment** | M5 | `_enrich_single_call()` | Triển khai hàm gộp đơn lượt (`_enrich_single_call()`) thực hiện đồng thời 4 tác vụ: tóm tắt nội dung, sinh 3 câu hỏi giả định (HyQA), tạo câu ngữ cảnh dẫn nhập (Contextual Prepend) và trích xuất metadata. Thiết kế này giảm 75% số lượng request API và tiết kiệm đáng kể thời gian tiền xử lý. |

---

## Phần 2: Khó khăn & Quá trình Debug thực tế (Challenges & Debugging)

Trong quá trình triển khai mã nguồn và chạy thực nghiệm, các vấn đề kỹ thuật sau đã phát sinh và được xử lý:

### 1. Lỗi lệch token tiếng Việt giữa BM25 và câu truy vấn
- **Hiện tượng:** Truy vấn tìm kiếm liên quan đến `"nghỉ phép năm"` cho điểm BM25 bằng 0 hoặc không tìm thấy tài liệu, dù kho văn bản có quy định tương ứng.
- **Nguyên nhân:** Khi gọi `word_tokenize(..., format="text")` của `underthesea`, văn bản được đánh chỉ mục chứa các token nối gạch dưới (`nghỉ_phép`), trong khi câu truy vấn của người dùng tách thành các từ đơn độc lập, dẫn đến lệch khóa từ vựng trong `BM25Okapi`.
- **Cách khắc phục:** Bổ sung bước xử lý chuỗi `.replace("_", " ")` trong hàm `segment_vietnamese()` để đưa toàn bộ token về dạng phân tách bằng khoảng trắng chuẩn.

### 2. Lỗi deprecation khi thao tác với `qdrant-client >= 1.9`
- **Hiện tượng:** Gọi phương thức `client.search()` trên Qdrant phiên bản mới phát sinh cảnh báo lỗi không tương thích.
- **Nguyên nhân:** Từ phiên bản `1.9.0`, thư viện `qdrant-client` thay thế API `search()` bằng `query_points()`, cấu trúc dữ liệu trả về nằm trong trường `.points`.
- **Cách khắc phục:** Cập nhật lại lời gọi API:
  ```python
  response = self.client.query_points(
      collection_name=collection_name,
      query=query_vector,
      limit=top_k
  )
  results = response.points
  ```
  Đồng thời xây dựng thêm cơ chế fallback sang `QdrantClient(":memory:")` khi môi trường container Docker chưa được khởi tạo.

### 3. Lỗi xung đột thư viện của `FlagEmbedding`
- **Hiện tượng:** Nạp mô hình rerank qua gói `FlagEmbedding` bị lỗi tương thích tokenizer trên phiên bản `transformers` hiện hành.
- **Nguyên nhân:** Gói `FlagEmbedding` có các ràng buộc phụ thuộc cũ dễ gây xung đột môi trường runtime.
- **Cách khắc phục:** Thay thế bằng lớp `CrossEncoder` chuẩn từ thư viện `sentence_transformers`:
  ```python
  from sentence_transformers import CrossEncoder
  self.model = CrossEncoder("BAAI/bge-reranker-v2-m3")
  ```
  Cách tiếp cận này giữ nguyên trọng số mô hình `bge-reranker-v2-m3`, đảm bảo mã nguồn gọn gàng và ổn định.

### 4. Quản lý tài nguyên bộ nhớ khi nạp nhiều mô hình
- **Hiện tượng:** Việc khởi tạo đồng thời hai mô hình lớn (`bge-m3` ~2.2GB và `bge-reranker-v2-m3` ~2.2GB) trên môi trường CPU có nguy cơ gây tiêu tốn tài nguyên và tăng độ trễ khởi động.
- **Cách khắc phục:** Triển khai cơ chế lazy loading (chỉ nạp khi thực thi), lưu trữ instance dùng chung và thực thi dự đoán bên trong context `with torch.inference_mode():` để vô hiệu hóa autograd graph, giúp tối ưu hóa dung lượng RAM và tăng tốc độ xử lý.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Dự án: **Hệ thống Tra cứu Quy trình Kỹ thuật & Tài liệu Vận hành Nội bộ (Engineering Wiki & SOP Assistant)**

#### 1. Hiện trạng & Vấn đề tồn đọng
- **Kiến trúc hiện tại:** Naive RAG cơ bản: cắt văn bản theo số lượng ký tự cố định (500 ký tự), nhúng vector đơn lẻ qua mô hình embedding chuẩn.
- **Hạn chế kỹ thuật:**
  - Tài liệu kỹ thuật chứa các khối mã lệnh (code blocks) và bảng cấu hình. Việc cắt máy móc làm đứt gãy cú pháp lệnh, dẫn đến câu trả lời bị sai lệch ngữ cảnh.
  - Các tài liệu SOP có nhiều phiên bản (v1.0 và v2.0). Hệ thống thường xuyên trích nhầm tài liệu cũ do câu chữ tương đồng.
  - Tỷ lệ Context Precision thấp khiến LLM dễ sinh ảo giác đối với các quy trình có điều kiện phụ thuộc.

#### 2. Kế hoạch áp dụng các kỹ thuật từ Lab 18
1. **Chiến lược Chunking:**
   - Sử dụng **Structure-Aware Chunking** để phân tách theo Markdown heading, bảo toàn nguyên vẹn các khối code block và bảng biểu.
   - Kết hợp **Hierarchical Chunking (Parent-Child)**: Parent chunk (2048 ký tự) chứa trọn vẹn một quy trình thao tác, Child chunk (256 ký tự) đại diện cho từng bước cụ thể.
2. **Tầng tìm kiếm Hybrid Search:**
   - Triển khai song song **BM25 tiếng Việt** (đã chuẩn hóa từ ghép) để bắt chính xác các định danh kỹ thuật, tên biến, API endpoint (`/api/v2/auth`) hoặc mã lỗi.
   - Kết hợp **Dense Search** (`BAAI/bge-m3`) để bắt nghĩa tương đồng từ câu hỏi tự nhiên của kỹ sư. Hợp nhất kết quả bằng **RRF** ($k=60$).
3. **Cross-Encoder Reranking:**
   - Tích hợp `bge-reranker-v2-m3` lọc từ top 20 kết quả tìm kiếm xuống top 3 đoạn trích chính xác nhất trước khi gửi tới LLM.
4. **Làm giàu dữ liệu (Enrichment):**
   - Áp dụng **Contextual Prepend** định danh rõ tài liệu: `[Hệ thống: Auth-Service | Tài liệu: Quy trình Triển khai v2.0 (Áp dụng 2024)]`.
   - Tạo bộ câu hỏi giả định **HyQA** để hỗ trợ tra cứu các trường hợp lỗi thường gặp.
5. **Đánh giá tự động định kỳ:**
   - Xây dựng tập dữ liệu benchmark 40 câu hỏi nội bộ.
   - Tích hợp pipeline đánh giá **RAGAS 4 chỉ số** vào quy trình CI/CD với ngưỡng kiểm soát: `Faithfulness >= 0.85` và `Context Precision >= 0.80`.

#### 3. Lộ trình triển khai (Timeline 4 tuần)
- **Tuần 1:** Chuẩn hóa tập tài liệu Markdown trên Wiki kỹ thuật; xây dựng pipeline tiền xử lý với Structure-Aware Chunking và Contextual Prepend (M1, M5).
- **Tuần 2:** Thiết lập cơ sở dữ liệu vector Qdrant, hoàn thiện bộ tìm kiếm Hybrid Search (BM25 + Dense Search) và thuật toán RRF (M2).
- **Tuần 3:** Tích hợp tầng Cross-Encoder Reranker (M3), tinh chỉnh prompt chống ảo giác và tối ưu độ trễ phản hồi (< 800ms).
- **Tuần 4:** Triển khai quy trình đánh giá tự động bằng RAGAS (M4) trên tập test benchmark, hoàn thiện tài liệu hướng dẫn và thử nghiệm nội bộ.
