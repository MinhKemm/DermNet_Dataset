# DermNet Dataset

Repository xử lý chú thích hình thái tổn thương và dữ liệu hỏi–đáp DermNet.
Chuẩn QA mới `2.0.0` tách **5 nhiệm vụ × 4 hình thức**, sinh 20 tổ hợp cho
bản bài báo và 20 biến thể vận hành từ cùng nguồn quy tắc.

## Bộ chuẩn QA mới

- [Thiết kế, 20 tổ hợp và cách dùng](docs/DERMNET_QA_STANDARD_V2.md).
- [Kết quả tự rà soát và giới hạn xác nhận](docs/DERMNET_QA_REVIEW_V2.md).
- [Nguồn cấu hình prompt tiếng Việt](Phase_2/config/qa_standard_vi.json).
- [Công cụ ghép prompt, kiểm tra QA và xuất TSV](Phase_2/qa_prompts.py).

Năm nhiệm vụ: Lesion recognition, Attribute recognize, Location, Lesion
Reasoning, Diagnose. Bốn `type` giữ nguyên từ dataset cũ: Short_answer,
Multi_choice, Judgement, Fill_in_blank. Diagnose lấy bệnh danh đã có, không
tự dự đoán lại. Không ép mỗi ảnh có đủ 20 câu.

## Dùng nhanh — không gọi mô hình/API

Từ thư mục gốc repo, với Python 3.10 trở lên:

```powershell
python -m unittest discover -v
python -m Phase_2.qa_prompts catalog --profile all --output qa_prompts_40.json
```

Chỉ phần chuẩn mới sử dụng thư viện chuẩn Python, không cần cài các mô hình
để xuất prompt và chạy test này. Nếu file đầu ra đã có, lệnh từ chối ghi đè.
Xem tài liệu thiết kế để chuẩn bị `context.json`, gắn dữ liệu vào một prompt,
kiểm tra JSON đáp án và chuyển sang TSV ứng viên.

## Phạm vi và trạng thái

`Phase_1` giữ luồng quan sát/chuẩn hóa cũ. `Phase_2/pipeline.py` là luồng QA
văn bản cũ; bộ chuẩn mới không tự thay thế nó hoặc mở inference khi import.
`Phase_2/VLMEvalKit` là mã đánh giá/vendor và các TSV benchmark hiện có.

Công cụ mới chưa có runner multimodal hoặc bộ điều phối worker. Đường dẫn ảnh
không thay thế ảnh đính kèm. Test kiểm tra hợp đồng phần mềm, không xác nhận
ảnh được đọc đúng hay QA được bác sĩ duyệt. Lịch chạy nhiều worker trong tài
liệu là đặc tả cho giai đoạn tích hợp, chưa được khởi chạy.
