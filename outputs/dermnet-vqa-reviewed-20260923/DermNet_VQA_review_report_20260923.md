# Báo cáo rà soát bộ câu hỏi DermNet VQA

Ngày rà soát: 2026-09-23

## Phạm vi

Rà soát toàn bộ dòng còn lại trong ba bản `cleaned_final` sau vòng lọc trước; không ghi đè các bản nguồn. Kiểm tra tự động câu hỏi/đáp án, cấu trúc, lựa chọn MCQ, khóa đáp án, trùng lặp, trường dữ liệu và khả năng mở ảnh từ thư mục chạy `Phase_2/VLMEvalKit`. Ảnh được đối chiếu thủ công cho các nhóm xung đột/nhãn đáng ngờ có mục tiêu; không khẳng định bác sĩ đã duyệt từng ảnh trong toàn bộ tập.

## Kết quả cuối

| Bộ dữ liệu | Dòng sau rà soát | Ảnh | Bị cách ly | Sai tiền tố đường dẫn runner | Ảnh thiếu | MCQ | MCQ còn lỗi |
|---|---:|---:|---:|---:|---:|---:|---:|
| Val_4k | 2,721 | 2,387 | 98 | 0 | 0 | 571 | 0 |
| Test_1of3 | 7,891 | 1,827 | 333 | 0 | 0 | 1,603 | 0 |
| Test | 23,681 | 5,483 | 990 | 0 | 0 | 4,797 | 0 |

## Phạm vi so với đầu vào thực tế

Ba file được rà ở đây là các bản `cleaned_final` ngày 23/09, sau đó được kiểm tra và xử lý thêm trong lượt này. Các TSV trong Downloads chỉ được dùng để xác nhận lỗi đường dẫn ở snapshot cũ, không dùng làm đầu vào nội dung cho lượt rà soát. Không ghi đè các file nguồn.

| Bộ dữ liệu | Dòng `cleaned_final` rà soát | Cách ly trong lượt này | Dòng đầu ra |
|---|---:|---:|---:|
| Val_4k | 2,819 | 98 | 2,721 |
| Test_1of3 | 8,224 | 333 | 7,891 |
| Test | 24,671 | 990 | 23,681 |

Các kiểm tra cuối đều đạt: không có dòng thiếu trường chính, trùng index hoặc Q+A trên cùng ảnh; không có ảnh thiếu/sai tiền tố đường dẫn; MCQ đủ bốn lựa chọn khác nhau và khóa đáp án khớp; không còn phân bố ở trường hình dạng/bề mặt, thuật ngữ cũ, dấu hiệu chủ quan/không nhìn thấy, lỗi hình thái trong Lesion_Recognition, mẫu leakage đã biết hoặc Lesion_Reasoning.
Lưu ý về Lesion_Reasoning: nhóm này đã có 0 dòng trong cả ba `cleaned_final` đầu vào trước lượt rà soát này. Vì vậy các prompt reasoning lỗi trong snapshot cũ không được diễn đạt lại ở đây; không có câu reasoning mới được sinh thêm.

Số dòng có ít nhất một hiệu chỉnh được ghi sổ: Val_4k=1,285; Test_1of3=2,570; Test=7,697. Các dòng có thể nhận nhiều hiệu chỉnh; bộ Test_1of3 là tập con của Test nên không cộng hai bộ này thành số ảnh riêng.

Đối soát split: Test_1of3 có 0 source_index và 0 ảnh không thấy trong Test; giao ảnh giữa Val_4k và Test là 0.

Câu Judgement vẫn có tỷ lệ Có/Không cần tính đến khi chấm điểm: Val_4k 294 Có / 327 Không; Test_1of3 920 Có / 972 Không; Test 2761 Có / 2924 Không. Test_1of3 là mẫu con của Test, không cộng hai bộ này.

## Kiểm tra đường dẫn gốc

- File nguồn trong Downloads cho Val_4k: 4,000/4,000 dòng có chuỗi thư mục lặp `dermnet-output/dermnet-output/images`.
- File nguồn trong Downloads cho Test_1of3: 19,104/19,104 dòng có chuỗi thư mục lặp `dermnet-output/dermnet-output/images`.
- File nguồn trong Downloads cho Test: 57,400/57,400 dòng có chuỗi thư mục lặp `dermnet-output/dermnet-output/images`.
- Kiểm tra phân biệt đúng tầng dữ liệu: ba `cleaned_final.tsv` đã được vòng làm sạch trước chuẩn hóa đường dẫn (0 đường dẫn lặp, 0 sai tiền tố, 0 ảnh thiếu); lượt này xác nhận lại từng đường dẫn và mỗi đường dẫn đều mở được từ runner.
- Ba `reviewed.tsv` tiếp tục có 0 đường dẫn lặp, 0 sai tiền tố và 0 ảnh thiếu. Từ `Phase_2/VLMEvalKit`, tiền tố đúng là `../../dermnet-output/images/...`; dạng `../../dermnet-output/dermnet-output/images/...` trong TSV Downloads gốc là sai và đã được loại trước khi tạo `cleaned_final`.
- Lưu ý: hai TSV Test tìm thấy trong Downloads mang ngày 02/06; chúng được dùng để kiểm chứng lỗi path ở các bản cục bộ đó. Nội dung được rà trong lượt này là các `cleaned_final` ngày 23/09, không lấy bản Downloads cũ làm đầu vào nội dung.
- Đối soát tổng hợp: `DermNet_VQA_path_audit_20260923.tsv`.

## Các nhóm đã xử lý

- Chuẩn hóa thuật ngữ: `tổn thương thực thể` → `tổn thương cơ bản`; `đóng mài/đóng mày`, `mài` trong gold label → `vảy tiết`; `bóng nước` → `bọng nước`; đồng nhất một số màu và chính tả.
- Kiểm tra cụ thể lỗi path `../../dermnet-output/dermnet-output/images/...`: lỗi có trong toàn bộ TSV gốc ở Downloads; bản `cleaned_final` đã sửa trước lượt này. Đã xác nhận lại mọi đường dẫn của 3 đầu ra từ thư mục runner; không còn chuỗi root lặp và tất cả ảnh đều tồn tại.
- Sửa MCQ sai thuật ngữ/lựa chọn ở một số dòng xác định; kiểm định lại sau chuẩn hóa, loại MCQ có lựa chọn trùng hoặc gold key không hợp lệ.
- Đưa câu hỏi/đáp án về đúng nhóm: tách pattern phân bố khỏi hình thái/bề mặt; chuyển dấu hiệu móng, tóc/lông và sắc tố ra khỏi `Lesion_Recognition`; ghi rõ câu hỏi về tổn thương cơ bản hay biến đổi thứ phát.
- Bổ sung bắt lỗi các nhãn mật độ/cụm/theo dải còn nằm trong trường bề mặt; chuyển cấu hình dạng lưới thuần sang trường hình dạng/cấu hình và cách ly câu gộp trường không thể tách an toàn. Chuẩn hóa subcategory `Lesion_Type` thành `Primary_Lesion_Type`.
- Cách ly các câu `Judgement` xung đột với JSON/ảnh hoặc mất liên kết JSON; MCQ có lựa chọn trùng; câu trộn nhiều trường không thể tách an toàn; và câu có ảnh X-quang hoặc ảnh mẫu nước tiểu thay vì ảnh da.
- Cách ly `Mảng` còn dùng như đáp án hình dạng vì dữ liệu không cho biết đó là patch phẳng hay plaque gồ; cách ly `Phân bố không đều` bị gộp với tăng sắc tố trong trường đặc điểm bề mặt (Test index=64).
- Cách ly thêm nhãn `sẩn nốt` không phân biệt được sẩn với nốt (Test source_index=15671), gold `Tăng sắc tố viêm` không chuẩn (Test source_index=49064), và các câu bề mặt còn trộn với cấu hình dạng lưới; giữ nguyên dữ liệu gốc trong source/ledger.
- Các bệnh danh đã nêu được giữ đúng phạm vi: Purpura → ban xuất huyết; urticarial vasculitis → viêm mạch mày đay; telangiectasia → giãn mao mạch; morphoea → xơ cứng bì khu trú; metastatic/ocular melanoma giữ thông tin vị trí/di căn; leukaemia cutis → xâm nhiễm bạch cầu ở da; mycosis fungoides → u sùi dạng nấm; solar lentigo → lentigo do nắng.
- Tên bệnh tiếng Anh còn lại trong nguồn chưa được dịch hàng loạt; cần một bảng thuật ngữ được duyệt để tránh dịch sai nghĩa hoặc làm mất thông tin bệnh.

## Sổ lỗi và bản đầu ra

Sổ lỗi ghi từng dòng được sửa/cách ly, gồm `source_index`, lý do, câu hỏi/đáp án trước và sau hiệu chỉnh, tiểu trường sau sửa: `DermNet_VQA_issue_ledger_20260923.tsv`.

Ba TSV `*.reviewed.tsv` là bản giữ lại để dùng tiếp. Số dòng cách ly được loại khỏi các TSV này nhưng giữ trong bản nguồn cũ và có dấu trong sổ lỗi.

## Giới hạn cần biết

Đây là rà soát dữ liệu bằng quy tắc chuyên môn, đối chiếu JSON và kiểm tra hình ảnh có trọng điểm. Quy tắc tự động có thể phát hiện lỗi cấu trúc/ngôn ngữ nhưng không thay thế hội chẩn da liễu cho mọi ảnh. Các dòng có gold lâm sàng còn chưa được xác minh ảnh trực tiếp đã bị cách ly thay vì tự sửa đáp án. `clean_core` được thay bằng `rule_checked_visual_not_fully_adjudicated` để không ngụ ý toàn bộ ảnh đã được bác sĩ xác nhận.
Một số mẫu câu vẫn được tái sử dụng trên nhiều ảnh. Chúng không bị xóa nếu cặp câu-đáp án không trùng trên cùng ảnh; mức đa dạng template cần được cân nhắc khi chấm khả năng tổng quát hóa.

## Kiểm tra còn lại theo file

- Val_4k: 2,721 dòng; 2,387 ảnh; thiếu trường=0; trùng index/Q+A=0/0; đường dẫn sai tiền tố/thiếu ảnh=0/0; MCQ lỗi=0; phân bố sai trường=0; thuật ngữ cũ=0; dấu hiệu không quan sát được=0; lỗi ontology Lesion_Recognition=0; hair sai tiểu trường=0; Judgement leakage mẫu đã biết=0; Lesion_Reasoning=0; câu hỏi duy nhất=1,115; dòng dùng câu hỏi lặp=1,801.
- Test_1of3: 7,891 dòng; 1,827 ảnh; thiếu trường=0; trùng index/Q+A=0/0; đường dẫn sai tiền tố/thiếu ảnh=0/0; MCQ lỗi=0; phân bố sai trường=0; thuật ngữ cũ=0; dấu hiệu không quan sát được=0; lỗi ontology Lesion_Recognition=0; hair sai tiểu trường=0; Judgement leakage mẫu đã biết=0; Lesion_Reasoning=0; câu hỏi duy nhất=2,769; dòng dùng câu hỏi lặp=5,489.
- Test: 23,681 dòng; 5,483 ảnh; thiếu trường=0; trùng index/Q+A=0/0; đường dẫn sai tiền tố/thiếu ảnh=0/0; MCQ lỗi=0; phân bố sai trường=0; thuật ngữ cũ=0; dấu hiệu không quan sát được=0; lỗi ontology Lesion_Recognition=0; hair sai tiểu trường=0; Judgement leakage mẫu đã biết=0; Lesion_Reasoning=0; câu hỏi duy nhất=7,320; dòng dùng câu hỏi lặp=16,997.
