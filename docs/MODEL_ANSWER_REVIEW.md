# Rà soát cách trả lời và Lesion Reasoning

## Kết luận

Đã sửa luồng tạo prompt theo cột `type` và bổ sung lại lệnh patch an toàn. Kiểm thử cục bộ kiểm chứng việc xây prompt, hợp đồng đáp án đóng, tách/gộp reasoning và giữ nguyên kết quả gốc khi patch lỗi. Chưa chạy suy luận bằng trọng số model trên GPU; chưa thể kết luận model luôn tuân thủ prompt hoặc trả lời đúng chuyên môn.

## Định dạng mong đợi

| Loại câu hỏi | Cách trả lời |
|---|---|
| Multi_choice | Chỉ chữ cái viết hoa: `A`, `ACD`; không giải thích |
| Judgement | Việt: `Có`/`Không`; Anh: `Yes`/`No` |
| Fill_in_blank | Thuật ngữ hoặc cụm từ còn thiếu |
| Short_answer | Cụm từ hoặc một câu ngắn, trả lời trực tiếp |

`category` chỉ nhóm nội dung (chẩn đoán, màu sắc, hình dạng, phân bố, đặc tính, nhận diện tổn thương, reasoning). `type` quyết định hình thức trả lời. Vì vậy một câu thuộc Lesion_Reasoning nhưng có type Judgement vẫn chỉ trả lời Có/Không; không ép mọi câu reasoning thành đoạn giải thích dài.

## Các thay đổi đã thực hiện

- Thêm `DermNetDataset` và đăng ký các tên dataset đang chạy. Cờ `force_use_dataset_prompt` khiến đường inference sử dụng prompt theo từng dòng dữ liệu.
- Dùng hướng dẫn thống nhất cho bốn loại câu hỏi bằng tiếng Việt/Anh. LLaVA và Vintern không thêm hướng dẫn cũ khi đã có hướng dẫn chung.
- DeepSeek không còn system prompt bắt mọi đáp án thành đoạn văn 1–3 câu; sửa nhận diện dự phòng cho nhận định “là phù hợp”. Hướng dẫn DermNet được giới hạn theo dataset.
- Sửa 1.527 ô đáp án đóng ở dữ liệu Anh bằng đối chiếu cùng index với bản Việt: Val có 144 Judgement; Test có 700 Judgement và 683 Multi_choice. Đây là sửa cấu trúc nhãn, không phải xác minh chuyên môn hay chất lượng dịch.
- Bổ sung `scripts/dermnet_reasoning_patch.py`: tách reasoning, yêu cầu đủ prediction, từ chối thiếu index/trùng index/câu trả lời trống/lỗi/thiếu ảnh, sao lưu rồi mới ghi kết quả bằng thay thế file nguyên tử.
- Patch chỉ cập nhật question, answer, category, type, prediction của các dòng reasoning và xóa score cũ của những dòng đó. Những dòng khác giữ nguyên dữ liệu.
- Kết quả prompt mới nằm trong `outputs/answer-format-v2`. Patch dùng dấu vân tay nội dung dataset và prompt trong tên checkpoint để tránh tái sử dụng bản vá khác nội dung.

## Số lượng Lesion Reasoning

| Dataset | Tổng reasoning | Judgement | Fill_in_blank | Short_answer | Multi_choice |
|---|---:|---:|---:|---:|---:|
| Val, mỗi ngôn ngữ | 554 | 182 | 188 | 184 | 0 |
| Test, mỗi ngôn ngữ | 2.713 | 915 | 919 | 878 | 1 |

Đây là số lượng trong dataset nguồn. Nếu chạy với chính sách bỏ ảnh thiếu, số dòng runtime có thể thấp hơn; file `*.missing-images.txt` ghi lại các dòng bị bỏ.

## Vấn đề còn tồn tại cần xem

1. Có ít nhất 39 câu gốc chứa “chẩn đoán Có/Không”, xuất hiện tương ứng ở cả Việt và Anh: 4 câu Val, 35 câu Test. Ví dụ Val index 514: “Hình ảnh này phù hợp với chẩn đoán Không. Nhận định này có đúng không?” nhưng đáp án là Có. Cần đối chiếu nguồn tạo câu hỏi để khôi phục tên bệnh; không suy đoán từ đáp án nhị phân. Val có index 514, 1707, 1710, 3408.
2. Chất lượng dịch còn có dấu hiệu lỗi, ví dụ cụm “Atypical melanocytic nodule flies”. Kiểm tra định dạng không chứng minh nội dung dịch đúng.
3. Đã cập nhật `all/resume` về kế hoạch cũ: 8 lượt full Việt và 8 lượt patch Việt, lọc theo GPU nếu dùng auto. Các file kết quả cũ cần được chép lên server trước khi chạy; thiếu file sẽ dừng trước inference. Các lượt tiếng Anh gọi riêng bằng lệnh full. Mặc định thiếu ảnh cũng dừng, chỉ bỏ dòng khi chủ động chọn skip.
4. `CustomVQADataset.evaluate` chưa có triển khai chấm điểm; runner đang dùng `--mode infer`. File prediction hoàn thành không đồng nghĩa đã có đánh giá độ đúng. Các score cũ của dòng được vá phải chấm lại.
5. Prompt là hướng dẫn cho model, không bảo đảm tuyệt đối đầu ra hợp lệ. Cần chạy mẫu trên server cho mỗi model × type × ngôn ngữ rồi kiểm tra prediction thực tế trước lượt chạy lớn.

## Kiểm chứng lại

Tại `Phase_2/VLMEvalKit`:

```bash
python -m unittest tests.test_dermnet_prompt tests.test_deepseek_vl2_instruction tests.test_dermnet_dataset_contract tests.test_dermnet_reasoning_patch
bash -n run_phase2.sh
```

Chạy riêng bản vá bằng đường dẫn tuyệt đối đến file kết quả cũ; chạy lại cùng lệnh khi bị gián đoạn:

```bash
bash run_phase2.sh patch deepseek_vl2_tiny DermNet_Val_4k-2_mac_relative /absolute/path/result.xlsx
```

Kiểm thử cục bộ không thay thế việc chạy mẫu với trọng số model trên server GPU.
