# Kiến trúc Phase 1: prompt và nền tảng VQA kiểm thử được

Ngày: 2026-10-03. Trạng thái: đề xuất lịch sử, không triển khai theo phạm vi hiện tại.
Người dùng đã chốt: Phase 1 chỉ lưu prompt; sửa prompt và đặt vào Phase 1 trước,
không tái cấu trúc runner/CLI. Hướng dẫn đang dùng là
[Phase_1/prompts/README_VI.md](../../../Phase_1/prompts/README_VI.md).
Thay thế `2026-10-03-phase1-prompt-redesign-design.md`.
Đây là thiết kế và kết quả rà soát, không phải biên bản hoàn tất triển khai.

## 1. Mục tiêu và ranh giới

Xây dựng một phần Phase 1 gọn, chạy được các bước chuẩn bị và kiểm tra offline,
dùng 40 prompt đã phân tích làm nền, không sinh QA bằng ghép câu cố định.
Cho phép tích hợp model và nhiều worker sau này nhưng không làm hệ thống chạy
model, scheduler, dịch tên bệnh hay tái tạo dữ liệu trong đợt thay đổi này.

“Toàn bộ dự án phù hợp Phase 1” được hiểu là tổ chức trách nhiệm và điểm vào của
repo quanh Phase 1, không di chuyển/xóa dataset, Phase 2 hay dữ liệu lịch sử.
Không biến Phase 2 benchmark thành thư viện phụ thuộc bắt buộc của Phase 1.
Không cam kết thời hạn thực tế khi chưa biết nhân lực và hạn chót của người dùng.

Giữ nguyên ảnh/dedup, QA cũ, progress/registry cũ, taxonomy nguồn, DOCX và few-shot.
Không đọc `.env` vào báo cáo, không cài thư viện model, không gọi API.
`tasks/` untracked ở gốc là nhật ký dedup, không phải module sinh QA; giữ nguyên.

## 2. Bằng chứng rà soát hiện tại

Đã fetch `origin/main`: `03e1fd53`. Nhánh công việc hiện tại
`codex/phase1-prompt-redesign-20261003` mới có commit tài liệu `ec693e80` trước lượt này.
Không chạy bất kỳ script sinh model hoặc script batch nào để kiểm tra.

| Mức | Điểm không phù hợp | Bằng chứng | Điều chỉnh đề xuất |
|---|---|---|---|
| P0 | QA từ đặc điểm giả định được ghi như dữ liệu thật | `scripts/run_full_pipeline.py:243,271-283,315`: Color mặc định, facts mock, đánh dấu completed | Ngừng đường chạy này; không thay mock bằng mock khác |
| P1 | Điểm chạy import module không có | `run_pipeline.py:14`, `run_test_demo.py:19`, `run_gold_standard.py:42`; không có `core/vlm_engine.py` hoặc `core/gemma_comparison.py` | Chặn entrypoint cũ bằng thông báo chuyển đổi; không tự xây adapter API trong đợt offline |
| P1 | Hai luồng/schema cũ khác nhau đang cùng được mô tả là pipeline | `config/prompts.yaml` dùng 5 dòng; `tasks/` dùng 20 hàm `generate_vqa()` | Một CLI Phase 1 rõ chức năng; bỏ các bộ ghép QA lỗi thời |
| P1 | Trộn ý nghĩa Location/Distribution và Shape/Boundary | `config/prompts.yaml:26-27,91-94`; `utils/json_handler.py:51-58` | Hợp đồng dữ liệu tách trường, không dùng bộ canonicalize cũ |
| P1 | Loại nhãn thật và nhận dữ liệu thiếu | `core/data_gate.py:10-36` | Trạng thái quan sát tách riêng giá trị; thiếu label Diagnose phải chặn |
| P1 | Nhiễu Reasoning chứa cơ chế bệnh không nhìn thấy | `tasks/task4_lesion_reasoning/prompt_4_1_multi_choice.py:36`, `prompt_4_2_judgement.py:5` | Nhiễu cùng mục tiêu thị giác, bỏ qua nếu không chứng minh được một đáp án duy nhất |
| P1 | Lưu dữ liệu bỏ mất trường mới | `utils/json_handler.py:62-71` chỉ giữ canonical keys cũ | Validate trước lưu, không âm thầm bỏ key; không sửa JSON cũ |
| P1 | Đường dẫn cá nhân và nhiều cách dò root | `config/settings.yaml:3-4`, `scripts/export_tsv.py:7`, `loaders/` | Dùng `pathlib`, root cấu hình được, đường dẫn tương đối |
| P1 | Pool Quantity và Distribution giống nhau | `taxonomy/taxonomy_loader.py:171-172` | View riêng từ mapping đã duyệt, không tự phân loại thuật ngữ chưa rõ |
| P1 | Vị trí rộng không có trong kết quả loader | `get_location_labels()` chỉ lấy `label`, không lấy nhãn level rộng | Cho dùng nhãn rộng có trong nguồn, không tự suy cây từ ô trống |
| P1 | Registry CSV ghi lại toàn bộ file, không khóa | `loaders/registry.py:53-58,115-119,138,163` | Không dùng registry này cho nhiều worker; hiện chỉ chuẩn bị job offline |
| P1 | Lỗi lưu có thể bị che và vẫn báo hoàn thành | `json_handler.py:142-145` bắt lỗi và chỉ print; `run_pipeline.py:212-220` vẫn ghi P2_OK | Lỗi lưu phải được báo lên caller; artifact mới ghi atomic, không đánh dấu thành công giả |
| P2 | Tên Phase 1/2/3 nội bộ gây nhầm với Phase_1/Phase_2 của repo | README và scripts | Gọi các bước là quan sát, chuẩn bị, kiểm tra, chọn, xuất |
| P2 | Kiểm thử Phase 1 chưa thành suite offline | Có demo gọi engine, chưa có `Phase_1/tests` | `unittest` với fixture nhỏ; không dùng demo API làm kiểm thử bắt buộc |

### Kết quả kiểm tra đọc-only / offline

- Parse AST thành công 48 file Python Phase 1; tìm thấy 20 hàm `generate_vqa()`.
- Gate Boundary=`Không rõ` trả False; Color/Lesion=`Không xác định được trên ảnh`
  trả True; Diagnose trống trả True; Reasoning cả hai trường unknown trả True.
- `canonicalize_fields()` nhận Location, Boundary, Size, Quantity, Diagnosis
  kiểu mới trả `{}`: các trường đó không được giữ.
- Pool Quantity và Distribution hiện bằng nhau.
- `Lưng`, `Cánh tay`, `Vùng quanh mắt` không nằm trong danh sách do
  `get_location_labels()` trả về; không có nghĩa những nhãn này vắng trong nguồn.
- Runtime Python bundled hiện thiếu `yaml`; không cài thêm để che hạn chế.
  Khi tách riêng hàm parser có sẵn khỏi import YAML, một dòng không theo mẫu gây
  `UnboundLocalError` vì `value_str` chưa được gán. Đây là probe hàm cô lập,
  không phải kiểm tra trọn module thành công.
- 12 test `Phase_2.tests.test_qa_prompts` và 19 test
  `Phase_2.tests.test_qa_validation` chạy thành công: tổng 31 test.
  Chúng chỉ chứng minh hợp đồng cũ còn hoạt động, không chứng nhận yêu cầu mới.

Hai lần phân tích ảnh trước là Codex xem ảnh và soạn QA, không gọi model API.
Không dùng các số candidate/review đó như chỉ số chính xác lâm sàng.

## 3. Lựa chọn kiến trúc

1. Vá 20 module và giữ cả hai runner: ít thay file nhưng giữ trùng logic và schema;
   không chọn vì đã được yêu cầu bỏ `generate_vqa()`.
2. Phase 1 độc lập, dùng pack 40 prompt đã duyệt, CLI offline và core nhỏ:
   chọn vì giải quyết trực tiếp lỗi, không cần GPU/API và kiểm thử được.
3. Xây dịch vụ phân tán, database queue, UI duyệt, nhiều adapter model ngay:
   không chọn cho lượt này vì quá phạm vi và chưa có bằng chứng cần mức đó.

Không thêm nguồn quy tắc chung tại `Phase_2/config` hoặc bộ luật nền để composer
ghép lại. Pack mới chứa 40 prompt đầy đủ, mỗi prompt có thể đọc và dùng độc lập.
Code chỉ chọn prompt theo metadata, không sáng tác câu hỏi/đáp án thay model.

## 4. Bố cục mục tiêu

```text
DermNet_Dataset/
  README.md                         hướng dẫn repo, chỉ rõ Phase 1 đang làm được gì
  Phase_1/
    __init__.py
    __main__.py                     python -m Phase_1
    cli.py                          catalog / render / validate / select
    README.md                       quickstart và giới hạn offline
    config/
      settings.json                 paths + prompt version + giới hạn QA; không API key
    core/
      contracts.py                  schema, nhiệm vụ, dạng trả lời, trạng thái
      eligibility.py                đủ/thiếu căn cứ theo trường
      selection.py                  chọn QA đa dạng, ít trùng, tối đa 10
      validation.py                 kiểm tra cấu trúc/nhãn, không kiểm chứng ảnh
    io/
      config.py                     root và đường dẫn, không phụ thuộc cwd
      prompt_catalog.py             nạp/render 40 prompt, không import Phase_2
      taxonomy.py                   đọc/view taxonomy, giữ raw nguyên trạng
      artifacts.py                  nạp response, ghi artifact atomic, chống ghi đè
    tests/
      test_prompts.py
      test_contracts.py
      test_eligibility.py
      test_selection.py
      test_cli.py
      fixtures/                     dữ liệu tổng hợp nhỏ, không QA cũ
    scripts/                        thông báo chuyển đổi tại entrypoint cũ
    taxonomy_data.json              nguồn hiện tại, giữ nguyên
    assets/, test_demo/, *.docx      giữ nguyên; demo cũ không phải suite mới
  outputs/
    dermnet-qa-prompts-20261001/      40 prompt v2 giữ nguyên làm bằng chứng gốc
    dermnet-qa-prompts-20261003/
      DermNet_40_Prompts_VI_v3.json   pack đã sửa từ v2, mỗi mục đủ nội dung
      README_VI.md                  40 ánh xạ, thay đổi và cách dùng
  docs/
    PHASE1_ARCHITECTURE.md           trách nhiệm các module và luồng
    PHASE1_MIGRATION.md              các lệnh cũ bị ngừng và lệnh thay thế
  dermnet-output/, final_canonical_vi/, Phase_2/  giữ nguyên
```

Pack ở `outputs` là artifact prompt đã duyệt có phiên bản, không phải kết quả
model. Phase 1 dùng đúng pack được settings chỉ định; không thêm một bản copy
vào `Phase_1/tasks` rồi sửa hai bản. Pack v2 không bị ghi đè. Không di chuyển
ảnh để làm cây thư mục trông gọn hơn. Root requirements phục vụ benchmark được
giữ; lõi CLI offline mới dùng standard library, không bắt cài torch hay SDK.

Các module/schema cũ chỉ được bỏ hoặc thay sau kiểm tra caller. Giữ entrypoint
cũ để báo chuyển đổi có chủ đích, không giữ một cây code còn được gọi ngầm.
Không di chuyển DOCX/assets hoặc xóa tiện ích gốc `handle_json.py`,
`convert_server.py` khi chưa có yêu cầu riêng.

## 5. Luồng dữ liệu và giao diện

### Lượt này triển khai đến đâu

```text
Taxonomy + pack 40 prompt + context ảnh
                 -> kiểm tra đầu vào -> render prompt (chưa đọc ảnh)

Response JSON được cấp tường minh
                 -> validate -> chọn QA đa dạng -> artifact candidate mới
```

Lượt này không có mũi tên tự gọi model giữa hai dòng. Fixture kiểm thử không
được mô tả là kết quả đọc ảnh. Mọi artifact response mới có trạng thái candidate,
không được tự gọi gold standard hoặc được duyệt lâm sàng.

Luồng tương lai: runner đính kèm ảnh thật -> quan sát toàn ảnh -> sinh trực tiếp
QA có căn cứ -> validate -> chọn 5–10 QA -> người duyệt khi cần. Model/worker
adapter chỉ tích hợp khi có yêu cầu mới. Không tự dịch disease label từ kiến thức
bệnh; Diagnose lấy nhãn nguồn được truyền riêng, giữ nguyên từng ký tự.

### Hợp đồng tối thiểu

- Task: Lesion_Recognition, Attribute_Recognition, Location, Lesion_Reasoning,
  Diagnosis. Format: Short_answer, Multi_choice, Judgement, Fill_in_blank.
- Trường tổn thương: Category, Location, Size, Color, Boundary, Shape, Quantity,
  Distribution. Diagnose là nguồn bệnh danh riêng, không Category hình thái.
- Observation có `values` (mảng), `state` (`observed`, `unknown`, `not_applicable`),
  `evidence` (mảng), `target`, `scope`; missing state không được coi là observed.
  `Không rõ` thuộc Boundary có thể là một giá trị observed; không so chuỗi để
  biến nó thành trạng thái unknown.
- Context có schema version, image_id, taxonomy version/fields, observations,
  target_scope, used_questions và giới hạn từng yêu cầu. Diagnose nhận riêng
  diagnosis_label/candidates; nhiệm vụ thị giác không nhận các trường bệnh danh.
- Response có schema version, image_id, taxonomy_version, `qas`, `skipped`.
  QA có task/type/sub_category, question, answer, answer_label, source_labels
  (mảng), evidence, rationale, target, scope, status=candidate.
- MCQ có options A–D, answer là khóa, answer_label là nội dung đáp án đúng.
  Judgement có claim_label khi cần, answer/answer_label là Có hoặc Không.
  Diagnose SA/FB giữ tên bệnh nguyên văn; MCQ giữ tên bệnh ở lựa chọn đúng;
  Judgement giữ bệnh danh trong claim/source_labels, không ép answer thành tên bệnh.
- Unknown là trạng thái quan sát; thiếu căn cứ thì skipped có lý do, không ép
  đủ câu có đáp án Không xác định. Không âm thầm làm rơi key không được hỗ trợ.
- Rendering thêm JSON serialized, không format toàn prompt bằng dấu ngoặc.
  Allowlist context; ID không chứa tên bệnh. Không hứa lọc được mọi rò rỉ ngữ
  nghĩa trong ghi chú tự do. Đường dẫn không thay cho ảnh được đính kèm.

Taxonomy adapter không sửa file nguồn, không tự suy quan hệ cây từ ô trống,
không tự chốt mapping Quantity/Distribution còn chưa duyệt. Nhãn rộng có sẵn
trong nguồn được cho phép; mô tả `Hai bàn chân` có thể dùng nhãn nguồn Bàn chân
và định tố hai bên dựa trên quan sát, không thêm một bệnh hay nhãn hình thái mới.
Định tố và nguồn bằng chứng được giữ riêng, không ép mọi câu trả lời phải bằng
một phần tử whitelist duy nhất.

### Ngân sách và độ đa dạng

- Tổng ảnh: mục tiêu 5–10, tối đa 10; ảnh thiếu căn cứ có thể dưới 5, có lý do.
- Mỗi prompt nhận phần ngân sách còn lại; không chạy mặc định cả 40 prompt/ảnh.
- Selector chỉ nhận candidate đã qua validator; ưu tiên phủ task/thuộc tính
  khác nhau, sau đó mới đa dạng format. Không lấy một fact x 4 format để lấp đủ.
- Thứ tự ổn định; khóa fact gồm task, sub_category, target, scope và tập nhãn.
  Reasoning không được tính như nhận diện nếu chỉ lặp tên tổn thương.
- Selection không tự đánh giá độ đúng lâm sàng hoặc chọn theo confidence do AI
  tự khai. Giữ số QA bị loại và lý do; không sinh thêm câu để bù.
- Hiện chưa có job scheduler. Khi thêm nhiều worker, mỗi job có run/image/prompt
  ID và version/hash; single-writer hoặc cơ chế transaction để ghi tiến độ.
  Không dùng CSV ghi đè đồng thời, không xem chờ N giây là chứng cứ đọc ảnh kỹ.

## 6. Những sửa đổi bắt buộc trong 40 prompt

Giữ mục tiêu của từng cặp task/format đã phân tích; sửa nội dung ngay từ pack
v2, không viết một bộ khác rồi gọi là bộ đã test. Ghi ID/version mới và nguồn v2.

1. Mục tiêu 5–10 toàn ảnh, không ép đạt số, không cứng câu chữ hoặc tỷ lệ format.
2. Nhận nhiều màu/vị trí khi có căn cứ; bỏ hạn chế đơn nhãn ở SA và dạng phù hợp.
3. Location phủ đúng toàn ảnh/phạm vi hỏi; cấp rộng đủ chắc được chấp nhận.
4. Boundary phân biệt độ rõ và độ đều; khác khía cạnh không tự thành đáp án sai.
5. Patch: dùng vùng đổi màu rộng, phẳng và định nghĩa chuẩn; không đòi mọi ảnh
   phải có thước cho Lesion Type, không bịa đo >1cm hoặc kết quả sờ nắn.
6. Crust/Vảy tiết và dấu hiệu đóng mài nhận diện từ bề mặt nhìn thấy, không ép
   lấy loại nền khác khi ảnh không đủ căn cứ.
7. Bất thường móng ngoài bộ nhãn hoặc cận cảnh không có mốc: unknown/skip,
   không ép thành Sẩn/Mảng hoặc đoán vùng cơ thể.
8. Quantity: đếm trực tiếp khi nhìn đủ mục tiêu; Vài/Nhiều không bịa ngưỡng.
   Size chỉ hỏi đo thực khi có căn cứ. Không suy cả cơ thể từ một ảnh cận cảnh.
9. Reasoning chỉ dấu hiệu nhìn thấy; không chép kích thước/độ sâu/độ chắc từ định
   nghĩa như thể đã quan sát, không hỏi cơ chế sinh bệnh.
10. Diagnose giữ nhãn nguồn; không được dùng bệnh danh suy màu, hình thái hoặc
    vị trí. Nhiễu thiếu hoặc có hơn một đáp án đúng thì bỏ qua dạng đó.
11. Câu hỏi ví dụ chỉ minh họa. MCQ/Judgement phải chứng minh mục tiêu phân biệt,
    không tạo Không bằng random chọn một label khác.
12. Paper/runtime có mục tiêu chung nhưng khác mức vận hành; runtime kiểm tra
    phạm vi, ngân sách và trùng fact, paper mô tả quy trình tái lập. Không đổi
    tên file/tiêu đề rồi xem như đã có hai cấu hình riêng.

15 mục review trước được phân loại lại theo góp ý người dùng, không coi tất cả
là lỗi hay 15 ảnh. Không ghi đè kết quả thử cũ, không công bố accuracy từ chúng.

## 7. Khả năng triển khai trong Phase 1

Chia thành ba đợt có thể dừng và kiểm tra độc lập:

- A: sửa pack 40 prompt, nội dung + bảng ánh xạ + ghi nhận thay đổi.
- B: CLI, hợp đồng và phần nối offline; ngừng generator/entrypoint cũ và sửa paths.
- C: validate/select, regression tests, hướng dẫn triển khai portable và bàn giao.

Không cần GPU, database server, UI hoặc mua API để hoàn thành A–C. Một môi trường
Python có standard library là đủ cho core và unittest; không đổi môi trường HPC.
Ước lượng sơ bộ 2–4 ngày công cho một người phát triển, gồm duyệt nội dung 40
prompt và regression. Đây là ước lượng, không lịch cam kết; chưa gồm bác sĩ duyệt,
tích hợp API hoặc đánh giá model/chi phí batch. Nếu Phase 1 yêu cầu chạy model
thật ngay, cần điều chỉnh phạm vi, nguồn lực và tiêu chí nghiệm thu riêng.

## 8. Tiêu chuẩn kỹ thuật và nghiệm thu

- Python >=3.10 (đồng nhất kiểu union sẵn có); kiểm tra offline trên Python 3.12
  hiện có. Không nói đã test 3.10 nếu chưa chạy môi trường đó.
- Module import không tạo file, quét dataset, đọc `.env` hay gọi mạng.
- Không hardcode home cá nhân; CLI hoạt động từ root và cwd khác trên Windows.
  Tính portable Linux phải có kiểm thử phù hợp, không chỉ suy từ pathlib.
- Đủ 40 prompt, 20/profile; mỗi ID có đúng task/type/profile và nội dung đầy đủ.
- Test render bảo toàn Unicode/ngoặc; sai cấu hình hoặc schema báo lỗi rõ.
- Test gate bảo toàn Boundary Không rõ và chặn unknown/empty Diagnosis đúng cách.
- Test multi-color/location, phạm vi ảnh, đa dạng task/attributes và ngân sách
  tối đa 10; dưới 5 không bị ép sinh thêm.
- Test từng dạng giữ bệnh danh nguồn; Reasoning có evidence/rationale. Kiểm tra
  cấu trúc không thể chứng minh evidence có thật trong ảnh, cần ghi giới hạn đó.
- Test Size cần căn cứ đo, Quantity không bịa ngưỡng; không tái áp validator
  đơn nhãn Phase 2 khiến các đáp án hợp lệ bị loại.
- Test atomic write, từ chối ghi đè mặc định, lỗi ghi trả thất bại; không âm thầm
  đánh dấu image completed. Lượt này không sửa progress cũ.
- Test entrypoint cũ dừng trước thao tác ảnh/QA/API. Không chạy script model thật.
- Fixtures mới là dữ liệu tổng hợp; không sửa, nạp tự động hoặc xuất lại QA cũ.
- Regression 31 test Phase 2 giữ nguyên; check diff không chạm protected paths.
- Bàn giao lệnh chạy catalog/render/validate/select và unittest, mô tả rõ chỉ
  triển khai offline, không gọi đó là production model pipeline.

## 9. Những quyết định cần người dùng duyệt ở bản này

Thiết kế đề xuất: giữ bộ prompt ở `outputs`, lấy v2 làm nền tạo v3; Phase 1
độc lập khỏi Phase 2; một CLI offline và các module nhỏ như cây trên; ngừng
mọi entrypoint sinh giả định/lỗi thời; không model, không scheduler trong lượt này.
Giữ cả source pack v2, data và công việc untracked nguyên trạng.

Theo quy trình thiết kế, sau khi người dùng duyệt tài liệu này mới lập kế hoạch
triển khai chi tiết và chọn cách thực hiện. Chưa có thay đổi code/prompt sản phẩm
ở giai đoạn rà soát này, chưa push main và chưa chứng nhận kiến trúc mới hoàn tất.
