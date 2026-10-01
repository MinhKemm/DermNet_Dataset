# Biên bản phục hồi và hợp nhất DermNet — 02/10/2026

## Kết luận và phạm vi

Đây là bản hợp nhất phục hồi trên nhánh `codex/reconcile-main-20261002`, không
phải xác nhận dataset đã sẵn sàng inference. Chưa cập nhật main khi còn lỗi
tham chiếu ảnh. Không force-push, không reset checkout chính, không xóa kết
quả mới, không thay đổi bệnh danh hay chọn nhãn cho ảnh đang chờ duyệt.

## Hai lịch sử được giữ lại

- Main mới trước hợp nhất: `2255b768c3f75f8408793054766052038afed92a`.
- Code trước lần cập nhật lại lịch sử: `codex/preserve-main-before-sync-20260930`
  tại `d57b5ace`; 30 commit chưa có trong ancestry của main.
- Merge-base: `3544df70`. Reflog ghi nhận origin/main forced-update từ
  `d57b5ace` sang `2736c270` ngày 30/09/2026.
- Fetch ngày 02/10 xác nhận main remote vẫn ở `2255b768`.
- Bản hợp nhất giữ cả hai cha lịch sử, thay vì chép code rồi bỏ mất ancestry.
- Giữ nguyên tree `dermnet-output/` và `final_canonical_vi/` của main;
  không đưa lại 35 ảnh bổ sung của nhánh cũ.
- Giữ 5 TSV tên gốc từ main và thêm 4 TSV VI/EN mà runner cũ sử dụng.
  Đây là các snapshot khác nhau, không phải các tên thay thế tương đương.

## Code và tài liệu đã gom

Runner `run_phase2.sh`, manifest 16 lượt, script chuẩn bị/sửa environment,
backend DeepSeek/vLLM/Huatuo, các adapter và test liên quan được phục hồi.
README giữ hướng dẫn HPC cũ và nối đến chuẩn QA v2, đồng thời ghi rõ giới hạn
dữ liệu và hướng dẫn environment có sẵn. Phần prompt v2 chưa tự thay thế
pipeline QA cũ hoặc khởi chạy subagent/lịch xử lý ảnh.

Sửa một lỗi được test tái hiện: `selected_patch_rows` lấy mask bằng
`to_numpy(copy=True)` trước khi sửa mask, tránh lỗi mảng chỉ đọc trên Pandas 3.
Có test kiểm tra lựa chọn dòng và việc không sửa DataFrame nguồn.
Cơ sở hành vi: [hướng dẫn Copy-on-Write của Pandas](https://pandas.pydata.org/docs/dev/user_guide/migration.html#copy-on-write-cow).

## Công việc mới và đang làm

43 file ổn định trong 6 thư mục `outputs/` được gom nguyên trạng:

- `dermnet-lexical-tree-20260925/`: cây thuật ngữ, bảng song ngữ và audit.
- `dermnet-qa-prompts-20261001/`: bản nháp 10 prompt và catalog 40 biến thể v2.
- `dermnet-vqa-cleaned-20260923/`: nguồn cleaned và báo cáo.
- `dermnet-vqa-quality-audit-20260923/`: báo cáo chất lượng.
- `dermnet-vqa-reviewed-20260923/`: reviewed, legacy-6col, ledger và script.
- `image-history-audit-20261001/`: báo cáo lịch sử ảnh.

SHA-256 của từng file khớp bản trong checkout chính tại thời điểm kiểm kê.
Các bản dịch/chuẩn hóa đề xuất vẫn giữ trạng thái nguồn; không nâng thành
thuật ngữ chính thức hoặc kết quả bác sĩ xác nhận. Script nghiên cứu có đường
dẫn máy cá nhân và một số thao tác ghi đầu ra khi chạy: chúng được giữ như
snapshot, không tự chạy/import để tái sinh hoặc ghi đè dữ liệu.

Công việc đang làm được **sao lưu và kiểm kê, chưa commit như công việc hoàn tất**:

- `outputs/image-dedup-20261002/` và `tasks/` nằm nguyên trong checkout chính.
- Snapshot ghi nhận 65 đường dẫn ảnh bị bỏ khỏi active set; lượt xử lý ảnh
  khác đã giữ archive gốc và một ảnh chờ xác nhận G050.
- Không chạm vào 65 thay đổi này; không tự suy ra nhãn hoặc chuyển ảnh G050.
- `.env` và output runtime/log ngoài phạm vi không được đọc, sửa hay publish.
- Bản sao lưu riêng giữ cả snapshot đầu và snapshot mới nhất của lượt đang làm;
  vị trí cụ thể được thông báo trong chat, không phải một nguồn Git công khai.

Chi tiết đường dẫn, dung lượng, SHA-256 và trạng thái đang làm:
[recovery/20261002_inventory.json](recovery/20261002_inventory.json).
Đây là snapshot, không tự nhận biết chỉnh sửa xảy ra sau thời điểm kiểm kê.

## Điểm chặn dữ liệu: không tự đổi đáp án hoặc loại dòng

Số dòng tham chiếu ảnh không tồn tại, đối chiếu Unicode NFC không phân biệt
hoa/thường. “Main” là bộ 7.057 ảnh đã commit tại `2255b768`; “live” là
bộ ảnh tại checkout chính sau lượt xử lý ảnh đang làm.

| TSV | Tổng dòng | Thiếu ảnh trên main | Thiếu ảnh trong live |
|---|---:|---:|---:|
| DermNet_Val_VI.tsv | 4000 | 943 | 978 |
| DermNet_Val_EN.tsv | 4000 | 943 | 978 |
| DermNet_Test_VI.tsv | 19133 | 4382 | 4501 |
| DermNet_Test_EN.tsv | 19133 | 4382 | 4501 |
| DermNet_Test.reviewed.tsv | 23681 | 5318 | 5457 |
| DermNet_Test_1of3.reviewed.tsv | 7891 | 1818 | 1858 |
| DermNet_Val_4k.reviewed.tsv | 2721 | 626 | 647 |

Các số này là **số dòng**, không phải số ảnh riêng, không cộng các split hoặc
hai ngôn ngữ thành số ảnh. Báo cáo reviewed 23/09 từng ghi không thiếu ảnh,
nhưng chỉ đúng với snapshot khi đó; không đúng với bộ ảnh được tuyển chọn sau.

Giữ `MISSING_IMAGE_POLICY=fail` mặc định. Không dùng `skip`, tự đổi đường
dẫn qua thư mục bệnh khác, khôi phục mọi ảnh cũ, hoặc sửa index chỉ để test đạt.
Việc chọn bộ QA chạy chính thức cần đối chiếu dữ liệu mới, chứng cứ ảnh,
split/index và Excel source; thay index làm mất khả năng vá kết quả cũ.

## Xác minh

- Baseline QA v2: 31 test đạt trước hợp nhất.
- Sau sửa Pandas: 11 test patch reasoning đạt, gồm TSV/XLSX và bảo toàn nguồn.
- Kiểm tra runner bằng Bash với `DRY_RUN=1`: không gọi model, không tải trọng số.
- Không thực hiện inference GPU hoặc cài môi trường trên HPC.
- Suite vendor chạy đúng thư mục kit, trong môi trường test riêng với Pandas
  3.0.1, pytest, tabulate và requests: **78 đạt, 4 không đạt, 3 subtest đạt**.
  Không cài dependency vào Python làm việc của người dùng hoặc environment HPC.
- Một test không đạt là `test_all_dataset_images_exist`: 15.975 tham chiếu
  thiếu ảnh trong các TSV tên gốc và canonical mà test quét; con số này
  đếm nhiều snapshot/ngôn ngữ, không phải số ảnh riêng.
- Ba test vendor API không đạt trên Windows:
  `test_async_recv_process_message_raises_when_process_exits`
  (BrokenPipeError khi pipe kết thúc),
  `test_eval_process_sends_result` và
  `test_prompt_process_builds_prompt_and_exits`
  (multiprocessing spawn không pickle được module stub).
  Các file API/mp_util và test này không bị thay đổi trong lượt phục hồi.
  Không tuyên bố đã kiểm tra GPU/Linux chỉ từ các test trên Windows.
- Lỗi dữ liệu giữ nguyên và phải được báo, không xóa test hay cho skip im lặng.

## Rà soát trước khi lưu nhánh

- Đã rà diff, resolve README, không còn unmerged entry và không có lỗi
  whitespace trong staged diff.
- 35 file Python được phục hồi/chỉnh sửa đã qua parse AST.
- `dermnet-output/` và `final_canonical_vi/` không có staged diff so với main.
- Code điều phối, cấu hình model và dữ liệu sửa cũ được giữ theo lịch sử;
  thay đổi hành vi mới chỉ là bản sao mask cho lỗi Pandas đã được test đỏ/xanh.
- 43 snapshot ổn định được đối chiếu SHA-256; không thực thi script có
  đường dẫn máy cá nhân, không chứng nhận lại nội dung lâm sàng.
- Kết luận: có thể lưu/push **nhánh phục hồi**, chưa đủ điều kiện nhập main
  hoặc chạy benchmark thật. Không sửa test để che các điểm chặn.

## Việc tiếp theo trước khi đưa vào main

1. Đối chiếu tham chiếu ảnh của legacy và reviewed với bộ ảnh tuyển chọn
   và nhật ký dedup mới; chỉ sửa đường dẫn khi có chứng cứ tương đương phù hợp.
2. Cách ly câu không còn ảnh/không đủ bằng chứng theo một quyết định dữ liệu
   được duyệt; giữ source_index và bản nguồn, không tự đoán đáp án.
3. Chốt bộ QA/split chạy chính thức và quan hệ với Excel kết quả cũ.
4. Chạy lại toàn bộ kiểm tra; fetch trước khi merge/push thường vào main.
