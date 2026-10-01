# Rà soát chuẩn hóa gần trùng trong Shape

## Kết luận

Đã rà đủ 344 nhãn Shape. Chỉ đề xuất chuẩn hóa khi khác biệt chủ yếu là cách viết, tiền tố Hình/Dạng, trật tự từ hoặc một biến thể từ ngữ rất gần. Các nhãn có qualifier hoặc có thể mang nghĩa khác được giữ nguyên hoặc đánh dấu để duyệt.

Báo cáo này không đề xuất chuyển nhãn sang trường khác, không thay đổi cấu trúc dữ liệu và không chỉnh sửa câu hỏi/đáp án nguồn.

## Tóm tắt

- Tổng nhãn được rà: **344**.
- Nhãn cần đổi sang canonical độ tin cậy cao: **46**.
- Nhãn được giữ làm canonical của các nhóm đó: **29**.
- Nhãn thuộc nhóm ứng viên gần nghĩa cần duyệt: **19**.
- Nhãn giữ nguyên: **250**.
- Nhóm chuẩn hóa gần trùng: **29**.
- Nhóm ứng viên cần duyệt: **6**.

TSV có số lần xuất hiện riêng cho Val, Test và Test 1/3. Không cộng các cột thành số ảnh duy nhất vì các tập có thể chồng lặp.

## Các nhóm từ gần nhau và canonical đề xuất

| Các từ đang dùng | Từ chuẩn đề xuất | Xử lý | Ghi chú |
| --- | --- | --- | --- |
| Bia bắn, Bia đích, Dạng bia, Hình bia bắn | Bia đích | Duyệt các từ này trước khi gộp | Các nhãn gần nhau nhưng mức độ tương đương chưa chắc chắn. Cần xác nhận: Bia bắn, Bia đích, Dạng bia, Hình bia bắn. |
| Bầu dục dài, Bầu dục kéo dài | Bầu dục kéo dài | Chuẩn hóa các biến thể gần trùng | Cùng hình dạng và qualifier kéo dài; không gộp với Bầu dục. |
| Bờ nham nhở, Mép nham nhở, Nham nhở, Rìa nham nhở | Bờ nham nhở | Duyệt các từ này trước khi gộp | Cùng mô tả bờ nham nhở; giữ riêng từ Nham nhở nếu thiếu ngữ cảnh. Cần xác nhận: Nham nhở. |
| Bản đồ, Dạng bản đồ, Hình bản đồ | Dạng bản đồ | Chuẩn hóa các biến thể gần trùng | Cùng mô tả dạng bản đồ; khác tiền tố. |
| Chùm, Dạng chùm, Thành chùm | Dạng chùm | Duyệt các từ này trước khi gộp | Các cách nói gần nhau; kiểm tra câu nguồn trước khi thống nhất cách diễn đạt. Cần xác nhận: Chùm, Dạng chùm, Thành chùm. |
| Dạng dải, Dải | Dạng dải | Chuẩn hóa các biến thể gần trùng | Cùng cụm từ; khác tiền tố Dạng. |
| Dạng lưới, dạng mạng lưới, lưới, Mạng lưới | Dạng mạng lưới | Chuẩn hóa các biến thể gần trùng | Cùng cách gọi dạng lưới/mạng lưới; giữ riêng vân lưới mờ và dạng ô lưới. |
| Dạng múi, dạng thùy, Nhiều thùy, Thuỳ, Thùy múi, đa thùy | Dạng thùy múi | Duyệt các từ này trước khi gộp | Các cụm gần nghĩa nhưng có thể khác nhau về số lượng/mức độ thùy. Cần xác nhận: Dạng múi, dạng thùy, Nhiều thùy, Thuỳ, Thùy múi, đa thùy. |
| Dạng vòm, hình vòm, Vòm | Dạng vòm | Chuẩn hóa các biến thể gần trùng | Cùng khái niệm vòm; khác tiền tố hoặc chữ hoa/thường. |
| Dạng nhẫn, Dạng vòng, Hình nhẫn, Hình vòng, Nhẫn, vòng | Dạng vòng | Duyệt các từ này trước khi gộp | Cùng cách gọi hình vòng; giữ riêng nhẫn và các qualifier vòng hở/một phần. Cần xác nhận: Dạng nhẫn, Hình nhẫn, Nhẫn. |
| Dạng đường, Đường | Dạng đường | Chuẩn hóa các biến thể gần trùng | Cùng cụm từ; khác tiền tố Dạng. |
| Dạng đường thẳng, Đường thẳng | Dạng đường thẳng | Chuẩn hóa các biến thể gần trùng | Cùng cụm từ; khác tiền tố Dạng. |
| Bán cầu, Hình bán cầu | Hình bán cầu | Chuẩn hóa các biến thể gần trùng | Cùng cụm từ; khác tiền tố Hình. |
| Bầu dục, Dạng bầu dục, Hình bầu dục | Hình bầu dục | Chuẩn hóa các biến thể gần trùng | Cùng một hình dạng; khác tiền tố Hình/Dạng. |
| Chữ nhật, hình chữ nhật | Hình chữ nhật | Chuẩn hóa các biến thể gần trùng | Cùng một hình dạng; khác tiền tố Hình và chữ hoa/thường. |
| Cung, Dạng cung, Hình cung | Hình cung | Chuẩn hóa các biến thể gần trùng | Cùng mô tả hình cung; khác tiền tố. |
| Cánh bướm, Hình cánh bướm | Hình cánh bướm | Chuẩn hóa các biến thể gần trùng | Cùng cụm từ; khác tiền tố Hình. |
| Hình tam giác, Tam giác | Hình tam giác | Chuẩn hóa các biến thể gần trùng | Cùng một hình dạng; khác tiền tố Hình. |
| Hình thoi, Thoi | Hình thoi | Chuẩn hóa các biến thể gần trùng | Cùng một hình dạng; khác tiền tố Hình. |
| Hình tròn, tròn | Hình tròn | Chuẩn hóa các biến thể gần trùng | Cùng một hình dạng; khác tiền tố Hình và chữ hoa/thường. |
| Hình tròn nhỏ, Tròn nhỏ | Hình tròn nhỏ | Chuẩn hóa các biến thể gần trùng | Cùng hình dạng và qualifier nhỏ; không gộp với Hình tròn. |
| Hình vòng cung, Vòng cung | Hình vòng cung | Chuẩn hóa các biến thể gần trùng | Cùng cụm từ; khác tiền tố Hình. |
| Hình đa giác, Đa giác | Hình đa giác | Chuẩn hóa các biến thể gần trùng | Cùng một hình dạng; khác tiền tố Hình. |
| Hình đồng xu, Đồng tiền, Đồng xu | Hình đồng xu | Chuẩn hóa các biến thể gần trùng | Cùng cách gọi hình đồng xu/đồng tiền; không gộp với hình tròn chung. |
| bất quy tắc, Không đều | Không đều | Duyệt các từ này trước khi gộp | Có thể gần nghĩa nhưng cần xác nhận source context trước khi gộp. Cần xác nhận: bất quy tắc, Không đều. |
| Bất đối xứng, Không đối xứng | Không đối xứng | Chuẩn hóa các biến thể gần trùng | Cùng nghĩa; khác tiền tố phủ định. |
| Lõm giữa, Lõm trung tâm, Trung tâm lõm | Lõm trung tâm | Chuẩn hóa các biến thể gần trùng | Cùng mô tả vùng lõm ở trung tâm; khác trật tự từ. |
| Bờ không rõ, Bờ mờ, Mờ ranh giới, Ranh giới không rõ, Ranh giới mờ | Ranh giới không rõ | Chuẩn hóa các biến thể gần trùng | Cùng mô tả ranh giới không rõ; khác cách diễn đạt. |
| Bờ không đều, Ranh giới không đều | Ranh giới không đều | Chuẩn hóa các biến thể gần trùng | Cùng mô tả bờ/ranh giới không đều. |
| bờ rõ, Bờ viền rõ, Ranh giới rõ | Ranh giới rõ | Chuẩn hóa các biến thể gần trùng | Cùng mô tả ranh giới rõ; khác cách diễn đạt. |
| Bờ khá rõ, Bờ tương đối rõ, Giới hạn khá rõ, Ranh giới tương đối rõ | Ranh giới tương đối rõ | Chuẩn hóa các biến thể gần trùng | Các cách diễn đạt gần nhau về ranh giới tương đối rõ. |
| Tròn bầu dục, Tròn-bầu dục | Tròn bầu dục | Chuẩn hóa các biến thể gần trùng | Chỉ chuẩn hóa dấu nối; không gộp với Bầu dục. |
| Vòng hở, Vòng không hoàn toàn, Vòng một phần | Vòng không hoàn toàn | Chuẩn hóa các biến thể gần trùng | Cùng mô tả vòng không khép kín; giữ nguyên qualifier không hoàn toàn. |

Các nhóm ghi Duyệt các từ này trước khi gộp là ứng viên, chưa áp dụng tự động. Bất quy tắc và Không đều chỉ nên nhập chung sau khi xác nhận câu nguồn; Tròn-bầu dục chỉ đổi dấu nối thành Tròn bầu dục, không nhập với Bầu dục.

## Giữ nguyên các khác biệt có ý nghĩa

- Giữ qualifier như nhỏ, kéo dài, một phần, hở, rõ, mờ, nhẹ, thấp và ngoằn ngoèo.
- Không gộp hình tròn với hình đồng xu, hình cầu với hình tròn, vòng hoàn chỉnh với vòng hở, hoặc hình bầu dục với bầu dục kéo dài.
- Không đổi nhãn trong dữ liệu nguồn cho đến khi nhóm duyệt mapping.

## Cách dùng bảng

Lọc cột action theo Chuẩn hóa gần trùng để xem các ánh xạ có thể áp dụng trên bản sao. Lọc theo Ứng viên gần nghĩa, cần duyệt để xác nhận bằng source context. Các dòng Giữ nguyên không cần chuẩn hóa theo đợt này.

## Kiểm tra

Bảng giữ đủ một dòng cho mỗi nhãn nguồn và bảo toàn số đếm tách theo từng split. Đây là rà soát từ vựng và mức độ gần nhau của nhãn, không phải thẩm định dấu hiệu trên từng ảnh.
