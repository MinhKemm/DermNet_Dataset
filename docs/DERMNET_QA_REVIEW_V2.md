# Tự đánh giá chuẩn QA DermNet 2.0.0

Ngày rà soát: 01/10/2026. Người thực hiện: agent viết mã tự rà soát;
**không phải phản biện độc lập hoặc xác nhận của bác sĩ**.

## Kết luận

Đủ 20 tổ hợp nhiệm vụ-hình thức và hai profile, có mã ghép prompt, kiểm tra
cấu trúc/nhãn và xuất TSV ứng viên. Chưa xác nhận chất lượng QA trên ảnh thật.
Không tự chạy mô hình, tạo automation hay sửa dataset đang dùng.

## Những điểm đã phát hiện và sửa

| Vấn đề | Rủi ro | Cách sửa và bằng chứng |
|---|---|---|
| Chỉ xoay vòng bốn hình thức mà không tách nhiệm vụ | Không kiểm tra đủ độ phủ, Reasoning biến thành hỏi tên | Nguồn cấu hình 5 task × 4 format, quy tắc riêng cho 20 cặp; test kiểm tra catalog 20/profile, 40 tổng. |
| Có thể truyền bệnh danh qua metadata cho nhiệm vụ khác | Nhãn bệnh chi phối quan sát/lời giải | Chỉ giữ trường chú thích hình thái/quan sát được phép; loại trường Diagnosis và tên file nguồn, tách gold riêng cho Diagnose. Regression test từng thất bại rồi qua sau sửa. Chuỗi tự do vẫn cần rà thủ công. |
| QA Size/Quantity có thể có nhãn hợp lệ nhưng thiếu căn cứ | Đáp án đúng từ điển nhưng không trả lời được từ đầu vào | Validator yêu cầu căn cứ/quy tắc được khai báo đã duyệt và người trả lời được xem; không tự tạo ngưỡng hoặc đo ảnh. Test thiếu/có/ẩn căn cứ. |
| Đầu ra vừa báo không có ảnh/mâu thuẫn vừa sinh QA | Đưa câu không có cơ sở vào TSV | Lý do chặn toàn tác vụ không được cùng tồn tại với qas; test bắt cả image_unavailable và image_label_conflict. |
| Bệnh danh có thể bị mô hình đổi | Sai gold so với yêu cầu người dùng | source_labels phải đúng nhãn nguồn, đáp án mở/MCQ đúng gold, phán định đối chiếu claim_label với gold. Test cả bốn hình thức. |
| MCQ sai khóa/trùng lựa chọn/sai trường | Chấm sai hoặc người trả lời có nhiều lựa chọn tương đương | Kiểm tra đủ A-D, nội dung đáp án khớp khóa, nhãn đúng trường và trùng chữ sau chuẩn hóa Unicode/hoa thường. Tương đương nghĩa và hai lựa chọn cùng đúng vẫn cần duyệt ảnh. |
| Reasoning MCQ trộn lời giải với tên loại | Đánh giá hình thức nhận diện thay vì căn cứ | Chặn lựa chọn chỉ là nhãn loại tổn thương; regression test thất bại rồi qua sau sửa. Vẫn cần duyệt chất lượng nội dung lời giải. |
| Quantity chỉ có ba nhãn nhưng MCQ yêu cầu bốn | Mô hình tự thêm nhãn ngoài chuẩn | Bỏ qua hình thức con không đủ lựa chọn; test chặn nhãn thứ tư Không có. Không buộc mọi thuộc tính có cả bốn hình thức. |
| Chỉ kiểm tra file test mới, bỏ qua khả năng discovery | Có thể báo test qua nhưng lệnh gốc không chạy test | Thêm package Phase_2 để python -m unittest discover -v tìm toàn bộ test first-party mới. |
| Ghi đè artifact có sẵn | Mất dữ liệu của người dùng | CLI tạo file theo chế độ exclusive; test kiểm tra nội dung file cũ còn nguyên. |

## Kiểm tra đã thực hiện

- Viết test trước khi triển khai; quan sát các test thất bại do chức năng chưa có.
- Bổ sung test tái hiện các lỗi self-review, quan sát thất bại trước khi sửa.
- `python -m unittest discover -v`: 31 test qua tại thời điểm ghi báo cáo,
  gồm test dữ liệu giả định cho 20 tổ hợp, CLI và chuyển JSON → TSV.
- Test CLI chạy subprocess thật với các file JSON trong thư mục tạm, không
  gọi mô hình/API, không dùng dữ liệu bệnh nhân thật.
- Catalog kiểm tra 20 tổ hợp/profile, mã prompt riêng cho 40 biến thể.

Các test ở đây là test first-party của chuẩn QA. Chúng không đại diện cho
toàn bộ benchmark/vendor VLMEvalKit, test API bên ngoài hoặc inference trên GPU.

## Đánh giá theo năm mặt

- **Đúng quy tắc:** năm task riêng, bốn type cũ giữ nguyên, gold Diagnose không
  được sửa, Reasoning yêu cầu căn cứ/lời giải, câu thiếu dữ liệu được bỏ qua.
- **Dễ đọc:** văn bản chuẩn ở config JSON; một module offline cho thao tác;
  tài liệu mô tả input/output và ví dụ 20 cặp. Không chép tay 40 bản khác nhau.
- **Phù hợp repo:** phần bổ sung độc lập, không thay đổi luồng cũ hay data,
  không giả định pipeline chỉ văn bản đã có khả năng đọc ảnh.
- **An toàn:** đầu vào kiểm tra trước khi ghép; dữ liệu nguồn không được coi
  là chỉ dẫn; không có API key, không gọi mạng, không ghi đè file đầu ra.
- **Hiệu năng:** phép ghép prompt/kiểm tra offline, không mở toàn bộ ảnh;
  lịch worker mới là đặc tả, chưa có lời hứa chính xác thời điểm hoàn tất.

## Các điểm chưa thể xác nhận bằng code

1. Ảnh có thực sự được đính kèm và mô hình nhìn đúng tổn thương hay không.
2. Nhãn nguồn, cây vị trí và ánh xạ bệnh danh đã được chuyên môn duyệt chưa.
3. Quantity cần ngưỡng rõ; Size còn khoảng chồng lấn/thiếu, không tự đổi thành
   các lớp đơn nhãn. Các khai báo reviewed/visible_to_answerer là trách nhiệm
   bên cấp dữ liệu, không phải bằng chứng AI đã kiểm tra ảnh.
4. Nhiễu MCQ khác nghĩa, cùng mức và chỉ một lựa chọn đúng về hình ảnh.
5. Câu Reasoning đúng về mặt phân biệt, không chỉ có evidence/rationale hợp lệ
   về kiểu dữ liệu. Các câu cùng ảnh có nhất quán hay không.
6. Hiệu quả so với bộ cũ và mức cân bằng 20 tổ hợp/sáu thuộc tính trên dữ liệu thật.

## Bước chạy thật được khuyến nghị

Chốt từ điển/quy tắc còn thiếu → nối runner multimodal gửi ảnh thật → pilot
có chọn mẫu và duyệt → đo độ đúng, tỷ lệ bỏ qua, p95 thời gian → mới tăng số
worker. Tách đọc ảnh mù nhãn khỏi lượt sinh QA có nhãn nếu muốn kiểm tra độc lập.
