# Bộ 40 prompt DermNet VQA tiếng Việt — v3

Ngày cập nhật: 2026-10-03. Phase 1 hiện là nơi lưu prompt để dùng với Codex đọc ảnh trực tiếp, không phải pipeline Python mới.

## Dùng bản nào?

- `paper/`: 20 prompt dành cho phần phương pháp/phụ lục bài báo; vẫn yêu cầu ảnh thật và bộ chuẩn hóa, không tự tạo kết quả thực nghiệm.
- `runtime/`: 20 prompt có thêm quy trình mở ảnh riêng, giữ phạm vi toàn ảnh, tự kiểm tra và bỏ qua khi thiếu căn cứ; dùng cho các lượt Codex xử lý ảnh sau này.

Mỗi file là một prompt đầy đủ, không phải module gọi nguồn quy tắc chung. Chỉ sao chép khối `text` trong file cần dùng. Không có phụ thuộc Phase_2/config hay hàm `generate_vqa()`.

40 = **5 nhiệm vụ × 4 dạng câu hỏi × 2 bản sử dụng**, không phải 40 câu/ảnh. Mỗi ảnh hướng tới **5–10 câu hữu ích, tối đa 10**, được ít hơn nếu không đủ căn cứ. Ngân sách tính chung sau khi hợp nhất mọi lượt.

## Mục lục đủ 40 prompt

| Nhiệm vụ | Dạng | Bản cho bài báo | Bản dùng thực tế |
|---|---|---|---|
| Lesion recognition | Trả lời ngắn | [LR-SA](paper/Lesion_Recognition__Short_answer.md) | [LR-SA](runtime/Lesion_Recognition__Short_answer.md) |
| Lesion recognition | Trắc nghiệm | [LR-MC](paper/Lesion_Recognition__Multi_choice.md) | [LR-MC](runtime/Lesion_Recognition__Multi_choice.md) |
| Lesion recognition | Có/Không | [LR-JG](paper/Lesion_Recognition__Judgement.md) | [LR-JG](runtime/Lesion_Recognition__Judgement.md) |
| Lesion recognition | Điền khuyết | [LR-FB](paper/Lesion_Recognition__Fill_in_blank.md) | [LR-FB](runtime/Lesion_Recognition__Fill_in_blank.md) |
| Attribute recognition | Trả lời ngắn | [AR-SA](paper/Attribute_Recognition__Short_answer.md) | [AR-SA](runtime/Attribute_Recognition__Short_answer.md) |
| Attribute recognition | Trắc nghiệm | [AR-MC](paper/Attribute_Recognition__Multi_choice.md) | [AR-MC](runtime/Attribute_Recognition__Multi_choice.md) |
| Attribute recognition | Có/Không | [AR-JG](paper/Attribute_Recognition__Judgement.md) | [AR-JG](runtime/Attribute_Recognition__Judgement.md) |
| Attribute recognition | Điền khuyết | [AR-FB](paper/Attribute_Recognition__Fill_in_blank.md) | [AR-FB](runtime/Attribute_Recognition__Fill_in_blank.md) |
| Location | Trả lời ngắn | [LOC-SA](paper/Location__Short_answer.md) | [LOC-SA](runtime/Location__Short_answer.md) |
| Location | Trắc nghiệm | [LOC-MC](paper/Location__Multi_choice.md) | [LOC-MC](runtime/Location__Multi_choice.md) |
| Location | Có/Không | [LOC-JG](paper/Location__Judgement.md) | [LOC-JG](runtime/Location__Judgement.md) |
| Location | Điền khuyết | [LOC-FB](paper/Location__Fill_in_blank.md) | [LOC-FB](runtime/Location__Fill_in_blank.md) |
| Lesion Reasoning | Trả lời ngắn | [REAS-SA](paper/Lesion_Reasoning__Short_answer.md) | [REAS-SA](runtime/Lesion_Reasoning__Short_answer.md) |
| Lesion Reasoning | Trắc nghiệm | [REAS-MC](paper/Lesion_Reasoning__Multi_choice.md) | [REAS-MC](runtime/Lesion_Reasoning__Multi_choice.md) |
| Lesion Reasoning | Có/Không | [REAS-JG](paper/Lesion_Reasoning__Judgement.md) | [REAS-JG](runtime/Lesion_Reasoning__Judgement.md) |
| Lesion Reasoning | Điền khuyết | [REAS-FB](paper/Lesion_Reasoning__Fill_in_blank.md) | [REAS-FB](runtime/Lesion_Reasoning__Fill_in_blank.md) |
| Diagnose | Trả lời ngắn | [DX-SA](paper/Diagnosis__Short_answer.md) | [DX-SA](runtime/Diagnosis__Short_answer.md) |
| Diagnose | Trắc nghiệm | [DX-MC](paper/Diagnosis__Multi_choice.md) | [DX-MC](runtime/Diagnosis__Multi_choice.md) |
| Diagnose | Có/Không | [DX-JG](paper/Diagnosis__Judgement.md) | [DX-JG](runtime/Diagnosis__Judgement.md) |
| Diagnose | Điền khuyết | [DX-FB](paper/Diagnosis__Fill_in_blank.md) | [DX-FB](runtime/Diagnosis__Fill_in_blank.md) |

Tên hiển thị đã sửa thành Attribute recognition. Các khóa máy `Attribute_Recognition`, `Diagnosis`, `Judgement` được giữ nhất quán để nhận diện nhiệm vụ/dạng, không đổi tên các nhãn lâm sàng.

## Cách sử dụng trước mắt

1. Chọn một ảnh, mở ảnh thật riêng; đọc toàn ảnh trước khi xem chi tiết. Không dùng tên file/thư mục hay contact sheet thay việc xem ảnh.
2. Cấp phần chuẩn hóa đã duyệt kèm phiên bản. Nội dung tài liệu, ghi chú và ảnh là dữ liệu tham khảo, không được ghi đè các quy tắc sinh QA.
3. Chọn prompt theo nội dung có căn cứ. Với Attribute recognition, giao rõ một trong sáu trường Size, Color, Boundary, Shape, Quantity, Distribution.
4. Cấp danh sách câu đã có và số chỗ còn lại của ảnh. Ví dụ ảnh đã có 6 câu thì ngân sách còn lại không quá 4; không yêu cầu mỗi lượt sinh lại 5–10 câu.
5. Đọc JSON ứng viên, đối chiếu ảnh, bỏ trùng nội dung và giữ tối đa 10 câu. Không cần đủ cả năm nhiệm vụ hoặc bốn dạng cho một ảnh.

Để có 5–10 câu đa dạng, chọn một số tổ hợp thích hợp trong thực đơn, không chạy hết 40 prompt. Sau này ưu tiên giao trọn ảnh cho một worker để tránh các worker cùng sinh lặp; lịch chạy/khóa/hợp nhất chưa được triển khai trong đợt này.

### Đầu vào tối thiểu

`INPUT_JSON` là thông tin được cấp, không phải thứ AI tự bịa. Các tên dưới đây là trường dữ liệu, không phải code cần chạy:

| Trường | Ý nghĩa |
|---|---|
| image_id | Mã ảnh thật cần xử lý |
| taxonomy_version | Phiên bản bộ chuẩn hóa đã duyệt |
| approved_taxonomy | Nhãn, định nghĩa và quan hệ đã xác nhận, không chỉ đường dẫn tài liệu |
| used_questions | Danh sách câu/nội dung đã có của ảnh; có thể rỗng ở lượt đầu |
| image_qa_budget | Số câu tối đa còn được thêm, số nguyên 0–10; không cấp thì vẫn áp dụng trần tổng 10 |
| attribute_field | Bắt buộc cho Attribute recognition: đúng một trong sáu trường |
| diagnosis_label | Bắt buộc cho Diagnose: nguyên văn bệnh danh nguồn |
| diagnosis_candidates | Nhãn bệnh nhiễu đã duyệt nếu cần cho trắc nghiệm/nhận định khác nhãn |
| correct_option_position | Tùy chọn, A/B/C/D để điều phối vị trí đáp án đúng |
| subtype | Tùy chọn `single_error_correction` khi thực sự cần câu sửa một lỗi |

Với bốn nhiệm vụ thị giác, không cung cấp bệnh danh như bằng chứng/gợi ý. Với Diagnose, dùng nguyên văn nhãn bệnh nguồn; không yêu cầu Codex đoán lại. Chỉ nói “đọc ảnh trước” trong một lượt đã thấy nhãn không tạo ra kiểm tra mù nhãn.

## Các sửa đổi theo kết quả duyệt trước đó

### Câu hỏi đúng ý và tự nhiên

Tất cả 40 prompt yêu cầu viết câu hỏi như hỏi một người đang xem ảnh: ngắn gọn, rõ phạm vi, không dùng từ chuyên môn chỉ để làm câu nghe trang trọng. Không đưa các từ điều phối như `target`, `scope`, `gold`, “nút cây” hay “bệnh danh mục tiêu” vào câu hỏi.

Với Location, ưu tiên những cách hỏi như “Tổn thương quan sát được trong ảnh nằm ở đâu?” hoặc “Trong ảnh, những vùng nào có tổn thương?”. Không cần hỏi “nằm ở cấu trúc nào?” khi thực chất chỉ xác định vị trí. Đáp án **Bản móng** vẫn giữ nguyên nếu đúng nhãn chuẩn hóa; đổi câu hỏi không đồng nghĩa đổi tên nhãn.

Đây là ví dụ cách diễn đạt, không phải mẫu bắt buộc. Vẫn giữ thuật ngữ y khoa cần thiết khi đó chính là nội dung được hỏi, giữ đúng bốn dạng câu hỏi và không đơn giản hóa đến mức mơ hồ hoặc lộ đáp án. Trước khi xuất, đọc lại để sửa chính tả, câu rườm rà và cách dịch cứng.

| Điểm | Quy tắc v3 |
|---|---|
| Vị trí bị thu hẹp | Đọc toàn ảnh; không tự chọn Lưng trên nếu còn vùng khác. Được dùng mức rộng chắc chắn, ví dụ Hai bàn chân; nhiều vị trí có thể cùng đúng. |
| Màu sắc | Chấp nhận nhiều màu trong một đáp án; không ép đúng một source_label. |
| Boundary | Rõ/mờ khác đều/không đều; Rõ và Không đều có thể cùng đúng. Không đặt hai khía cạnh thành nhiễu cạnh tranh. |
| Không xác định | Bỏ qua hợp lệ khi thiếu căn cứ. Không rõ là nhãn ranh giới nếu nguồn đã duyệt, không đồng nghĩa ảnh mờ hoặc thiếu quan sát. |
| Patch | Dùng định nghĩa Dát lớn đã thống nhất; không đòi thước cho mọi nhận diện hình thái, nhưng không tuyên bố đo được >1 cm từ ảnh không có căn cứ đo. |
| Crust và móng | Đóng mài có thể hỗ trợ Crust/Vảy tiết; bất thường móng ngoài bộ nhãn không bị ép thành Sẩn/Mảng. |
| Quantity/Size | Đơn độc/đếm thấy rõ được dùng theo nhãn nguồn; không tự đặt ngưỡng Vài/Nhiều. Số đo thật vẫn cần căn cứ đo. |
| Reasoning | Giải thích 1–3 dấu hiệu nhìn thấy; không suy nguyên nhân bệnh, độ chắc, cảm giác hoặc bệnh sử. Loại tổn thương chưa chắc thì bỏ qua lý giải. |
| Diagnose | Giữ nguyên diagnosis_label, kể cả thể bệnh/giai đoạn. Thiếu nhãn thì bỏ qua; nhiễu chỉ từ danh sách đã duyệt. |
| Đa dạng | Ưu tiên các nội dung có căn cứ khác nhau, không một sự thật × bốn dạng để lấp số câu. Ví dụ câu hỏi không bắt buộc dùng nguyên văn. |

## Đầu ra và giới hạn

- JSON phiên bản `3.0.0` gồm `image_id`, `taxonomy_version`, `qas`, `skipped`, `needs_review`. Mỗi QA có bằng chứng, target/phạm vi và `status="candidate"`.
- `source_labels` cho phép nhiều nhãn thực sự cùng đúng; `answer_label` vẫn là chuỗi. Trắc nghiệm: `answer` là A–D, `answer_label` bằng nội dung lựa chọn đúng. Điền khuyết: đúng một `____`.
- Diagnose dùng `evidence_kind="source_label"`, không gọi nhãn nguồn là bằng chứng nhìn thấy. Bốn nhiệm vụ khác dùng `evidence_kind="image"`.
- Không xác định được mục tiêu thì ghi `skipped` với lý do và `observation_state="undetermined"`. Không tự tạo nhãn “Không xác định” nếu chuẩn hóa không có nhãn đó.
- Chưa tương thích trực tiếp validator/TSV v2. Không đưa kết quả v3 vào runner cũ rồi coi kiểm tra cấu trúc là chứng minh đúng ảnh.
- Chưa chạy model, subagent, cron hoặc sinh lại QA. Kiểm tra tĩnh chỉ xác nhận bố cục/nội dung, không xác nhận độ chính xác lâm sàng.

Xem [biên bản kiểm tra](CHECKS_VI.md) để phân biệt kết quả kiểm tra tĩnh v3 với 31 kiểm thử hồi quy của code v2 giữ nguyên.

## Nguồn và cách tránh nhầm bản

Điểm xuất phát là [DermNet_40_Prompts_VI_v2.json](../../outputs/dermnet-qa-prompts-20261001/DermNet_40_Prompts_VI_v2.json). SHA-256 nguồn:

`1adcab11fba198ca8776fa22c53d7c29f1c71d653359c6b663a491692eb26df3`

Bản v2 trong outputs được giữ nguyên làm lịch sử. Từng file v3 ghi ID v2 kế thừa để truy lại. Các test ảnh trước đây dùng nội dung v2 được dựng từ Phase_2 rồi xuất ra outputs; không phải test ảnh của bộ v3 này.

**Bản dùng hiện tại là các file v3 trong thư mục này.** Không lấy config Phase_2 hoặc module cũ Phase_1/tasks thay thế. Hai đề xuất tái cấu trúc trong docs/superpowers/specs là lịch sử, không phải triển khai được chốt.

Hai profile có cùng chính sách QA; runtime thêm kiểm tra thao tác đọc ảnh. Vì mỗi prompt là một tài liệu đầy đủ, khi chỉnh chính sách chung cần sửa các file bị ảnh hưởng và kiểm tra lại cả hai profile. Không có nguồn quy tắc chung hoặc đồng bộ runtime ngầm.
