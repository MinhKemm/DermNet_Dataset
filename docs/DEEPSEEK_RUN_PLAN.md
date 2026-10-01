# Danh sách chạy tiếng Việt

Manifest hiện tại: `Phase_2/VLMEvalKit/scripts/dermnet_jobs.txt`. Server 2 × 96 GB chọn 18 lượt: 14 full và 4 vá. Không chọn Gemma hoặc dataset tiếng Anh. Các file lịch sử vẫn được giữ, không xóa kết quả cũ.

| Model | Val VI | Test VI | Cấu hình |
|---|---|---|---|
| deepseek_vl2_small | Full | Full | deepseek-ai/deepseek-vl2-small, BF16 |
| deepseek_vl2_int8 | Vá | Vá | deepseek-ai/deepseek-vl2, load_in_8bit=True |
| deepseek_vl2_tiny | Vá | Vá | deepseek-ai/deepseek-vl2-tiny, BF16 16-bit |

## Excel nguồn

Bốn file người dùng cung cấp trong Downloads có hậu tố `(1).xlsx`. Bản sao trong repo giữ nguyên nội dung, SHA256 khớp nguồn:

- `outputs/deepseek_vl2_tiny/source/deepseek_vl2_tiny_DermNet_Val_VI.xlsx`
- `outputs/deepseek_vl2_tiny/source/deepseek_vl2_tiny_DermNet_Test_VI.xlsx`
- `outputs/deepseek_vl2_int8/source/deepseek_vl2_int8_DermNet_Val_VI.xlsx`
- `outputs/deepseek_vl2_int8/source/deepseek_vl2_int8_DermNet_Test_VI.xlsx`

Các đường dẫn trên tương đối với `Phase_2/VLMEvalKit`. Dù tên nguồn Test có `1of3`, mỗi file chứa đủ 19.133 dòng của Test VI hiện tại. Val có 4.000 dòng. Cả bốn vượt qua kiểm tra index, ảnh và metadata/câu hỏi cho phép của công cụ vá.

Vá vẫn tách mini TSV, chạy model, merge vào Excel riêng và giữ nguyên nguồn. Mini gồm reasoning và các sửa dữ liệu được luồng hiện tại cho phép. Tên file không chứng minh trọng số đã dùng trong lần chạy lịch sử; cấu hình vá áp dụng theo yêu cầu người dùng và đăng ký hiện tại.

## Chạy từ root

```bash
bash Phase_2/VLMEvalKit/run_phase2.sh plan
bash Phase_2/VLMEvalKit/run_phase2.sh all
# Chạy tiếp sau gián đoạn:
bash Phase_2/VLMEvalKit/run_phase2.sh resume
```

Đã chạy 37 kiểm thử thành công, bao gồm dry-run toàn bộ 18 lượt. Chưa chạy inference thực tế trên GPU trong lần sửa này.
