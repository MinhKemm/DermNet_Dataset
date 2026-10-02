# Biên bản kiểm tra bộ prompt v3

Ngày: 2026-10-03. Phạm vi: nội dung prompt và điều hướng tài liệu, không đánh giá lâm sàng hoặc chạy model.

## Kết quả đã kiểm tra

| Kiểm tra | Kết quả |
|---|---|
| Số file prompt | 40: 20 paper + 20 runtime |
| Ma trận nhiệm vụ/dạng | Mỗi profile đủ 5 × 4, không trùng/thiếu tổ hợp |
| ID và provenance | Mỗi file có ID v3 và ID v2 kế thừa tương ứng |
| Nguồn v2 | SHA-256 khớp giá trị trong README, không bị sửa |
| Quy tắc chung | 5–10 QA/ảnh, tối đa 10; được ít hơn 5; không lặp một sự thật bằng bốn dạng; cho phép nhiều nhãn đúng |
| Điểm sửa theo nhiệm vụ | Patch/Crust, vị trí rộng/nhiều vị trí, sáu thuộc tính, Boundary hai khía cạnh, bỏ Reasoning chưa chắc, Diagnose giữ nhãn nguồn |
| Ràng buộc hình thức | Trắc nghiệm bốn lựa chọn/một đáp án; Có/Không; điền đúng một chỗ trống |
| Chính sách giữa hai profile | 20 cặp có phần quy tắc QA giống nhau; chỉ khác hướng dẫn phương pháp/thao tác vận hành |
| Điều hướng | 40 liên kết prompt trong mục lục và liên kết mới từ tài liệu liên quan đều trỏ tới file tồn tại |

Các kiểm tra trên được thực hiện bằng script Python đọc file, đối chiếu ma trận/ID, nội dung bắt buộc, SHA-256 và đường dẫn. Kiểm tra cụm từ/đối chiếu văn bản không chứng minh prompt sẽ luôn được mô hình tuân thủ.

## Kiểm thử hồi quy code giữ nguyên

Đã chạy `python -B -m unittest Phase_2.tests.test_qa_prompts Phase_2.tests.test_qa_validation -v` với Python 3.12 của workspace: **31/31 đạt**.

Đây là kiểm thử hợp đồng v2 của Phase 2 được giữ nguyên, không phải validator cho JSON v3 và không phải đánh giá độ chính xác ảnh của bộ prompt mới.

## Chưa thực hiện

- Không chạy mô hình/API, subagent hoặc cron.
- Không sinh lại QA; không sửa ảnh, dedup, taxonomy nguồn hoặc QA cũ.
- Không nối v3 vào validator/TSV cũ, không sửa runner Python.
- Không chứng nhận độ chính xác lâm sàng, hiệu quả dataset hay kết quả bài báo.

Bước đánh giá ảnh sau này cần mở từng ảnh thật, kiểm tra phạm vi/nhãn/căn cứ và ghi rõ ai duyệt. Không dùng biên bản kiểm tra tĩnh này thay cho bước đó.
