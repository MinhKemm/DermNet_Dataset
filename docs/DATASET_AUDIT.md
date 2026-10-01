# Rà soát dataset nguồn

Tên nguồn hiện tại: `DermNet_Val_VI.tsv`, `DermNet_Test_VI.tsv`, `DermNet_Val_EN.tsv`, `DermNet_Test_EN.tsv`. Việc đổi tên giữ nguyên nội dung dữ liệu; các tên Reasoning_Fix bên dưới là tên file lịch sử đã dọn.

Kiểm tra bốn TSV đang dùng trong `Phase_2/VLMEvalKit/LMUData`, ngày 2026-09-08. Đây là kiểm tra cấu trúc, đối chiếu dữ liệu và đường dẫn ảnh; chưa xác minh toàn bộ đáp án chuyên môn.

## Cập nhật sau khi sửa

Đã khôi phục 35 ảnh từ commit `96a1a9f0`; Pillow xác nhận ảnh đọc được. Kiểm thử cả bốn TSV hiện không còn ảnh thiếu.

Đã sửa 140 câu hỏi: 78 câu chẩn đoán (39 × 2 ngôn ngữ) lấy nhãn từ `final_canonical_vi`, trường `TRICH_XUAT_JSON.Danh_muc_benh`; 62 câu được chuẩn hóa dấu sau nhãn lựa chọn. Nhật ký trước/sau và nguồn: `Phase_2/VLMEvalKit/scripts/dataset_repairs.json`. Script tái lập: `scripts/repair_dataset_labels.py`.

Runner chạy lại reasoning, các câu sửa khớp nhật ký và dòng có đáp án nguồn thay đổi, rồi merge vào Excel riêng. Đã đối chiếu thành công cả 26 Excel đầu vào. Kết quả mới dùng `outputs/answer-format-v3`, không dùng checkpoint cũ. Các số liệu và danh sách dưới đây là lỗi phát hiện **trước khi sửa**, không phải lỗi còn tồn tại. Chất lượng dịch và đáp án chuyên môn chưa được xác nhận toàn bộ.

## Kết quả trước khi sửa

| Kiểm tra | Val, mỗi ngôn ngữ | Test, mỗi ngôn ngữ |
|---|---:|---:|
| Tổng dòng | 4.000 | 19.133 |
| Index trùng | 0 | 0 |
| Thiếu trường bắt buộc | 0 | 0 |
| Dòng không tìm thấy ảnh trong checkout | 17 | 67 |
| Câu chứa chẩn đoán Có/Không vô nghĩa | 4 | 35 |
| Câu Anh có nhãn lựa chọn thiếu dấu, ảnh hưởng kiểm tra đáp án | 5 | 16 |

Đã đối chiếu theo index: hai ngôn ngữ khớp tập index, image_path, category, type, đáp án trắc nghiệm và nhãn Có/Không ↔ Yes/No. Kiểm tra ảnh đã chuẩn hóa Unicode NFC và không phân biệt hoa/thường; số thiếu là số dòng, không phải số ảnh duy nhất. Với mặc định `MISSING_IMAGE_POLICY=fail`, runner sẽ dừng vì ảnh thiếu.

## Các trường hợp cần xử lý

- Val có câu chẩn đoán vô nghĩa tại index: 514, 1707, 1710, 3408.
- Test: 2415, 2426, 2434, 3879, 7515, 8202, 8268, 8389, 8397, 8415, 8440, 8462, 8581, 8607, 8614, 8702, 8852, 9122, 9163, 9561, 9580, 9925, 9950, 12967, 13791, 15785, 15790, 15875, 16014, 18959, 18970, 19012, 19024, 19025, 19029.
- Nhãn trắc nghiệm Anh thiếu dấu tại Val: 1233, 1390, 1408, 1460, 1957; Test: 2029, 3863, 4365, 4546, 5195, 5589, 6267, 7740, 7839, 10180, 13596, 14750, 15616, 16358, 16461, 18529. Ví dụ `B round` thay vì `B. round`. Đây là lỗi định dạng, không đồng nghĩa chữ cái đáp án bị sai. Danh sách này chỉ xét nhãn được đáp án tham chiếu.
- Một số lựa chọn tiếng Anh bị dịch trùng hoặc chưa tự nhiên; cần duyệt cùng bản Việt, không thể xác nhận nội dung chỉ bằng kiểm tra định dạng.

Các câu vô nghĩa cần khôi phục nội dung từ nguồn tạo câu hỏi; chưa tự suy đoán tên bệnh. Cần bổ sung ảnh hoặc sửa đường dẫn theo ảnh đã xác minh trước khi chạy đủ dataset.

## Dọn file cũ

Đã loại `LMUData/DermNet_Val_4k-2_Reasoning_Fix.tsv` khỏi cây làm việc: file có 571 dòng nhưng chỉ 66 dòng Lesion_Reasoning và không được luồng shell hiện tại sử dụng trực tiếp. Có thể khôi phục từ lịch sử Git.

Giữ các tên dataset Reasoning_Fix trong mã đăng ký và shell vì runner còn dùng để tạo mini TSV mới từ `category == Lesion_Reasoning` trong dataset runtime. Val nguồn có 554 câu reasoning; Test có 2.713. Không xóa dataset nguồn hoặc kết quả thí nghiệm khác.
