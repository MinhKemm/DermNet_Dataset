# Đối chiếu Excel nguồn và kế hoạch chạy

Phạm vi: 27 Excel trong DermNet_Dataset/Phase_2/VLMEvalKit/outputs; không sử dụng Excel từ Dermnet-QA.

- 26 file khớp index/ảnh với dataset hiện tại và câu hỏi ngoài reasoning không đổi. Đã chạy validate_source trên từng file: 26/26 đạt.
- SmolVLM-256M Val tại T20260611-111520 lệch 2.900 ảnh trên 4.000 index, toàn bộ câu hỏi khác. Không dùng làm nguồn merge; manifest chuyển lượt này thành full, giữ Excel cũ.
- Các file Val VI tương thích có 554 câu hỏi reasoning thay đổi; Test VI có 2.711 câu hỏi reasoning thay đổi trên tổng 2.713 dòng reasoning. Các dòng khác giữ nguyên câu hỏi.
- Gemma4-12B-it Val EN có câu hỏi trùng dataset hiện tại; vẫn nằm trong danh sách vá theo yêu cầu áp dụng logic prompt mới. Những lỗi nhãn đáp án ngoài reasoning không được patch này tự sửa.
- Giữ cả ba Excel Gemma3 Val khác timestamp, không tự chọn bản mới nhất hoặc xóa lịch sử.

Manifest scripts/dermnet_jobs.txt hiện có 41 lượt: 26 patch và 15 full. DeepSeek Tiny chạy full Val/Test tiếng Việt. Server 2 × 96 GB vượt ngưỡng chọn model; điều này chưa chứng minh environment và adapter chạy được thực tế.

Kiểm tra nguồn và merge đã thêm đối chiếu image_path sau chuẩn hóa phần đường dẫn images, kiểm tra câu hỏi/category/type ngoài reasoning, và kiểm tra metadata của patch mới. Có test từ chối Excel trùng index nhưng khác ảnh.

Bốn config đã được bổ sung: Gemma4-12B-it trỏ google/gemma-4-12B với AutoModel; Janus/Phi dùng NF4, double quant, bfloat16; LLaVA dùng load_4bit của builder. Status nguồn ghi commit ba6451f4 nhưng config tại commit đó không có các alias này, nên không khẳng định cấu hình tái dựng giống nguyên trạng thí nghiệm cũ. All đã qua kiểm tra đăng ký và dry-run toàn bộ 41 lượt.

Excel nguồn và ảnh chưa bị sửa trong đợt kiểm tra này. Còn ảnh thiếu và lỗi nội dung đã ghi trong DATASET_AUDIT.md. Không có kết quả inference GPU mới trong đợt kiểm tra.
# Kế hoạch hiện tại

Danh sách trong báo cáo bên dưới là lịch sử kiểm tra. Kế hoạch mới đã bỏ Gemma/tiếng Anh và thêm bốn Excel DeepSeek người dùng cung cấp: xem [DeepSeek và luồng chạy Việt](DEEPSEEK_RUN_PLAN.md). Hiện có 18 lượt, gồm 14 full + 4 vá.
