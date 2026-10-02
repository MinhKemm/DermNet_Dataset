# Tích hợp main mới, code phục hồi và dedup - 02/10/2026

## Phạm vi được người dùng yêu cầu

Tích hợp code phục hồi và kết quả dedup vào main, push thường, không force-push.
QA cũ tạm dừng vì mục tiêu là sinh lại VQA. Giữ QA, kết quả và công cụ cũ như
snapshot lịch sử; không chạy mô hình, không sửa JSON/TSV/đáp án để che lỗi thiếu ảnh.
Lượt này không triển khai kiến trúc runner sinh VQA đã đề xuất trong chat.

## Nguồn được giữ

- Main remote ở lần fetch đầu: `92697940b3438a89457f374fde3915e4de9c52e0`.
- Main remote có thêm commit trong lúc tích hợp:
  `e8671bfbf9db262bc87dc3d456fd095ab772a204` ("Phase 1 chạy lại").
  Đã dừng trước push, fetch và ghép commit này; không ghi đè lịch sử mới.
- Nhánh phục hồi: `1242765ac435bc0853ef177fa095a33b6321864a`.
- Lịch sử cũ `d57b5ace` vẫn là ancestor của nhánh phục hồi và lịch sử tích hợp.
- Giữ rename ba ảnh từ `Mucocoele of the lip/` sang `Mucocele of the lip/`
  của commit main mới. Không đổi byte hay nội dung pixel của ba ảnh này.
- Giữ code HPC/runner, chuẩn prompt QA v2 và các artifact nghiên cứu đã phục hồi.
- Giữ nguyên tree `Phase_1/` và `list_images.txt` của main tại `e8671bfb`,
  gồm pipeline/prompt/taxonomy mới và việc bỏ registry cũ. Không chạy pipeline mới.
- Thêm toàn bộ 82 file snapshot dedup, gồm bằng chứng nguồn, bảng review,
  nhật ký, ánh xạ đường dẫn, archive và ảnh G050 chờ duyệt.
- Thư mục làm việc chính và các công việc đang làm được sao lưu riêng trước
  tích hợp. Không publish `.env`, khóa bí mật, cache Python hoặc `tasks/` cá nhân.

## Trạng thái ảnh sau tích hợp

| Kiểm tra | Kết quả |
|---|---:|
| Ảnh trong active set | 6.992 |
| Ảnh chờ duyệt nhãn ngoài active set | 1 |
| Nội dung pixel duy nhất được bảo toàn | 6.993 |
| Nhóm trùng byte trong active set | 0 |
| Nhóm trùng pixel trong active set | 0 |
| Đường dẫn gốc bỏ khỏi active set và được archive | 65 |
| Nội dung pixel bị mất | 0 |

Chỉ áp dụng 65 mục trong `removal_journal.json` sau khi đối chiếu SHA-256
của ảnh gốc và archive. 64 bản dư bị bỏ, một ảnh đại diện được giữ riêng chờ
duyệt; không chọn nhãn cho G050. Archive có kiểm tra CRC và SHA-256 đủ 65 mục.

Các inventory và `verification.json` của dedup giữ nguyên snapshot tại
`2255b768`. Chênh lệch tên thư mục ba ảnh Mucocele được xử lý khi kiểm chứng,
không viết lại bằng chứng lịch sử. Kết quả quét lại nằm trong
[integration_verification.json](../outputs/image-dedup-20261002/integration_verification.json).

## G050 và khôi phục

G050 vẫn nằm ngoài cây ảnh gán nhãn. Hai nhãn cũ chưa được xác nhận;
không tự chuyển bệnh danh hoặc sửa annotation liên quan.

- [Báo cáo dedup gốc](../outputs/image-dedup-20261002/REPORT_VI.md).
- [Ảnh G050 chờ duyệt](../outputs/image-dedup-20261002/pending_label_review/G050/21.jpg).
- [Script khôi phục](../outputs/image-dedup-20261002/restore_images.py)
  mặc định chỉ dry-run; không chạy `--apply` khi kiểm tra tích hợp.

## Giới hạn

Ảnh sạch trùng chính xác không có nghĩa đã kiểm chứng mọi nhãn lâm sàng hoặc
loại hết ảnh gần trùng/crop. JSON và TSV cũ vẫn có tham chiếu ảnh không còn
trong active set; không chứng nhận các snapshot đó dùng để inference hiện tại.
Giữ `MISSING_IMAGE_POLICY=fail`; không xóa test hoặc đổi sang skip để che lỗi.
Các kiểm thử offline không chứng minh độ chính xác đọc ảnh, GPU hoặc HPC.

## Kiểm tra trước khi commit

- Quét độc lập toàn bộ 7.057 ảnh trước áp dụng nhật ký và 6.992 ảnh sau áp dụng;
  khớp SHA-256 và hash pixel RGBA sau EXIF orientation với inventory lịch sử.
- Đối chiếu đủ 82 file dedup từ checkout chính tới worktree và bản sao lưu.
- 83 kiểm thử offline được chọn đạt: QA v2 (31), contract/prompt/patch/runner
  DermNet (36), DeepSeek (7), Huatuo (1), adapter cũ (5), vendor patch (3).
- 40 file Python phục hồi qua parse AST; năm script Bash qua `bash -n`.
- 39 file Python từ commit main mới qua parse AST. `Phase_1/` và
  `list_images.txt` khớp main mới; `Phase_2/` và `final_canonical_vi/`
  không đổi so với nhánh phục hồi.
- Không chạy lại toàn bộ suite legacy có lỗi tham chiếu ảnh/API Windows đã
  ghi trong biên bản phục hồi; không tuyên bố toàn bộ suite đó đã đạt.
- Không inference, tải trọng số, cài dependency hoặc chạy job HPC.
