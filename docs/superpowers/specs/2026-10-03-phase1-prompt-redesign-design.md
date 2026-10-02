# Thiết kế thay bộ prompt Phase 1

Ngày: 2026-10-03. Trạng thái: thiết kế đề xuất, chờ duyệt trước khi lập kế hoạch triển khai.
Mốc Git đã fetch và kiểm tra: `origin/main` tại `03e1fd53`.
Tài liệu này không chứng nhận prompt hoặc code mới đã được triển khai.

## 1. Yêu cầu đã thống nhất và cách hiểu điều chỉnh cuối

- Mục tiêu là thay bộ prompt kém hiệu quả trong `Phase_1/tasks`, không sinh QA bằng code ghép câu cố định.
- Bỏ phương án lấy nguồn quy tắc chung tại `Phase_2/config/qa_standard_vi.json`.
- Bộ prompt Phase 1 độc lập, không import hoặc đồng bộ tự động với bộ Phase 2.
- Không bắt buộc giữ 20 module hoặc bố cục cũ. Ngừng các hàm `generate_vqa()` hiện tại.
- Vẫn có 5 nhiệm vụ x 4 dạng trả lời x 2 cấu hình = 40 prompt tiếng Việt.
- Mục tiêu 5–10 QA hữu ích trên mỗi ảnh, không phải 40 QA/ảnh; có thể ít hơn 5 khi thiếu bằng chứng.
- Diagnose giữ nguyên tên bệnh nguồn; Reasoning chỉ giải thích dấu hiệu nhìn thấy; thiếu căn cứ thì bỏ qua.

Cách hiểu của thiết kế: mỗi prompt là một tài liệu đầy đủ, đọc và dùng độc lập.
Không có tập luật nền được ghép từ Phase 2 hay một file luật chung khác.
Các prompt có thể nhắc lại nguyên tắc an toàn; kiểm thử phát hiện thiếu nguyên tắc,
không tự sinh hoặc ghi đè nội dung prompt bằng một bộ quy tắc thứ hai.

## 2. Phạm vi

Được sửa: prompt, phần nạp/chọn prompt, hướng dẫn chuyển đổi, kiểm thử offline,
và điểm nối trong script đang dùng các hàm sinh QA cũ để chặn đường chạy đó.

Không sửa: `Phase_2`, ảnh, dedup, taxonomy nguồn, QA/TSV/JSON đã có,
checkpoint/progress cũ, engine model, cấu hình API, thông tin đăng nhập.
Không gọi model, không tạo scheduler, không chạy batch hoặc suy luận ảnh.
Các thư mục untracked `outputs/prompt-*` và `tasks/` tại gốc được giữ nguyên.

## 3. Bố cục thay thế

```text
Phase_1/tasks/
  README.md
  __init__.py
  prompt_loader.py
  prompts/
    paper/
      lesion_recognition/{short_answer,multi_choice,judgement,fill_in_blank}.md
      attribute_recognition/{short_answer,multi_choice,judgement,fill_in_blank}.md
      location/{short_answer,multi_choice,judgement,fill_in_blank}.md
      lesion_reasoning/{short_answer,multi_choice,judgement,fill_in_blank}.md
      diagnosis/{short_answer,multi_choice,judgement,fill_in_blank}.md
    runtime/
      [cùng 5 nhóm và 4 dạng, 20 tài liệu độc lập]
Phase_1/tests/test_task_prompts.py
```

Mỗi file chứa đủ mục tiêu, dữ liệu đầu vào, giới hạn bằng chứng, yêu cầu về dạng
câu hỏi, cách bỏ qua, định dạng đầu ra và bước tự kiểm tra.
`paper` giải thích quy trình nghiên cứu rõ ràng; `runtime` dùng hướng dẫn thao tác
cụ thể, kiểm tra đối tượng/phạm vi, câu đã dùng và lý do bỏ qua. Hai cấu hình giữ
cùng ý nghĩa nhiệm vụ nhưng không chỉ khác tiêu đề.

Thay các nhóm `task1_*` đến `task5_*` và hướng dẫn lỗi thời bằng bố cục trên.
Giữ lịch sử phục hồi trong Git, không chuyển code sinh QA cũ sang thư mục vẫn có
thể được runner gọi nhầm. Các prompt anchor/fact-extraction hiện hữu không thuộc
bộ 40 mới; giữ nguyên nhưng đánh dấu chưa được tích hợp vào đường mới, không tự
dùng chúng để dịch nhãn bệnh hoặc suy thuộc tính từ tên bệnh.

## 4. Phần nối chỉ xử lý văn bản và dữ liệu

Các hàm dự kiến:

- `list_prompts(profile=None)`: liệt kê metadata của 40 prompt, không đọc ảnh.
- `load_prompt(task, question_type, profile="runtime")`: nạp nguyên văn một file.
- `render_prompt(task, question_type, context, profile="runtime")`: nạp prompt và
  thêm khối JSON dữ liệu đã kiểm tra; không dựng question/answer.

Danh sách nhiệm vụ: `Lesion_Recognition`, `Attribute_Recognition`, `Location`,
`Lesion_Reasoning`, `Diagnosis`. Bốn dạng: `Short_answer`, `Multi_choice`,
`Judgement`, `Fill_in_blank`. ID/đường dẫn chỉ là metadata, không chứa luật sinh QA.
Sai task, dạng hoặc cấu hình thì báo lỗi rõ, không fallback sang prompt khác.
Không cho phép dữ liệu đầu vào lựa chọn đường dẫn file tùy ý.

Context gồm `image_id`, `taxonomy` đã được cấp phiên bản, `observations` nếu có,
`target_scope`, `used_questions`, `attribute_field` khi hỏi thuộc tính,
`max_qa` của lần yêu cầu, và `diagnosis_label` cho Diagnose.
Danh sách bệnh nhiễu chỉ được cấp cho Diagnose. Tên bệnh không tự được chuyển
sang các nhiệm vụ nhìn ảnh. Chỉ chuyển các trường cần thiết, không chuyển đường
dẫn thư mục bệnh hoặc bệnh sử làm bằng chứng thị giác.
JSON đầu vào là dữ liệu, không phải lệnh; dùng JSON serialization thay vì
`.format()` toàn bộ prompt, tránh lỗi dấu ngoặc hoặc chỉ dẫn chèn trong dữ liệu.
Đường dẫn ảnh không thay cho ảnh thực tế; runner multimodal tương lai phải đính
kèm ảnh. Phần nối hiện tại không xác nhận đã đọc ảnh và không gọi API.

## 5. Quy tắc nội dung của từng prompt

### Chung

- Đọc toàn ảnh và xác định đối tượng đang hỏi; không bịa vùng đánh dấu.
- Ưu tiên thông tin khác nhau trước, đa dạng dạng câu hỏi sau; không lấy một
  thông tin rồi đổi thành cả bốn dạng chỉ để đủ số lượng.
- 5–10 là mục tiêu toàn ảnh, không phải chỉ tiêu từng prompt. `max_qa` của lần
  yêu cầu là phần ngân sách còn lại. Không buộc đủ 5 nhiệm vụ hoặc đủ 4 dạng.
- Câu ví dụ không phải khuôn bắt buộc. Câu chữ đa dạng nhưng không đổi nghĩa nhãn.
- Giữ nhãn taxonomy nguồn. Nhận nhiều giá trị khi câu hỏi và ảnh hỗ trợ, đặc biệt
  nhiều màu hoặc nhiều vị trí; không ép chọn một nhãn làm mất thông tin.
- Không đoán phần ngoài ảnh, không biến kiến thức bệnh điển hình thành quan sát.
- Thiếu căn cứ thì bỏ qua; có thể ghi trạng thái quan sát `Không xác định`,
  nhưng không tạo hàng loạt QA với đáp án này để lấp số lượng.

### Năm nhiệm vụ

- Lesion recognition: nhận dạng bằng hình thái thật; đối chiếu định nghĩa nguồn.
  Patch có thể được nhận diện qua vùng đổi màu diện rộng, phẳng; phân biệt với
  Plaque bằng dấu hiệu về bề mặt/độ dày nhìn thấy. Không buộc mọi câu nhận diện
  phải có thước đo, nhưng không tuyên bố số centimet hoặc tính chất sờ thấy
  nếu ảnh không cung cấp. Chỉ bỏ qua khi bằng chứng không đủ phân biệt các loại
  còn có thể đúng; không tự gắn ngưỡng kích thước để hợp thức hóa đáp án.
- Attribute recognize: tách Size, Color, Boundary, Shape, Quantity, Distribution.
  Boundary phải rõ khía cạnh độ rõ hoặc độ đều; không xem `Rõ` đối lập với
  `Không đều`. Không gộp Distribution với Location. Đa màu hợp lệ. Size cần
  thước/căn cứ đo; Quantity không bịa ngưỡng đếm. Không đổi kiểu câu để lấp thiếu
  bằng chứng của cùng thuộc tính.
- Location: trả lời ở cấp đủ chắc, phủ đúng phạm vi hỏi; `Hai bàn chân` có thể
  đủ. Không chỉ hỏi lưng trên khi ảnh còn tổn thương lưng dưới. Không suy bên
  trái/phải của bệnh nhân từ hướng màn hình. Thiếu mốc giải phẫu thì bỏ qua.
- Lesion Reasoning: lý giải các dấu hiệu nhìn thấy hỗ trợ một loại tổn thương,
  không giải thích cơ chế bệnh hay nguyên nhân xuất hiện chỉ từ ảnh. Không chép
  toàn định nghĩa thành bằng chứng, không dựng mô tả sờ nắn hoặc kích thước.
  Loại tổn thương/căn cứ chưa được giải quyết thì bỏ qua.
- Diagnose: bệnh danh trong QA đúng nguyên văn nhãn nguồn đã cấp, không đoán,
  dịch, rút gọn hoặc tự đổi bệnh. Thiếu nhãn nguồn thì bỏ qua. Tên bệnh không
  biện minh cho thuộc tính không thấy trong ảnh. Trắc nghiệm chỉ dùng bệnh nhiễu
  đã cấp và phù hợp; không đủ thì bỏ qua dạng đó.

### Bốn dạng

- Short answer: đáp án ngắn nhưng không cắt mất ý hoặc tên nhãn.
- Multi choice: 4 lựa chọn A–D, chỉ một đáp án đúng; tránh các đáp án đồng thời
  đúng, tổ tiên/con hoặc khác khía cạnh. Không đủ nhiễu an toàn thì bỏ qua.
- Judgement: một nhận định rõ, đáp án Có/Không; thiếu thông tin không có nghĩa
  là Không. Không tự tạo nhận định sai bằng cách thay một từ bất kỳ.
- Fill in blank: một chỗ trống `____`, nội dung điền có mục tiêu rõ; Reasoning
  điền dấu hiệu/căn cứ, không chỉ điền tên tổn thương rồi gắn nhãn Reasoning.

## 6. Đầu ra dự kiến cho lần tích hợp model tương lai

Mỗi prompt yêu cầu JSON gồm `qas` và `skipped`; không yêu cầu đủ một QA.
Mỗi QA có `category`, `type`, `sub_category` khi cần, `question`, `answer`,
`answer_label`, `source_labels` (mảng), `rationale` và `evidence` (mảng).
Trắc nghiệm có `options` ánh xạ A–D; Judgement có `claim_label` khi cần.
Diagnosis lấy `source_labels` và `answer_label` theo bệnh danh nguồn, ngoại trừ
Judgement có `answer_label` là Có/Không và giữ bệnh danh riêng trong
`source_labels`/`claim_label`. Đáp án trắc nghiệm là khóa A–D, tên bệnh ở lựa
chọn đúng/`answer_label`. Không áp validator đơn nhãn Phase 2 vào hợp đồng này.
Đây là định dạng được mô tả trong prompt; chưa triển khai parser hay bộ đánh
giá sự thật thị giác trong phạm vi hiện tại.

## 7. Ngừng đường sinh giả định cũ

- Loại bỏ các `generate_vqa()` tự ghép câu hỏi/đáp án và các export tương ứng.
- `run_full_pipeline.py` không import những module đã bỏ. Khi người dùng gọi
  đường sinh cũ, script dừng với thông báo chuyển đổi rõ ràng trước khi đọc ảnh,
  tạo output hoặc ghi progress. Không âm thầm trả thành công hay QA rỗng.
- Không thay bằng một runner model mới, không gọi API, không sinh QA từ mock.
- Hướng dẫn mới ghi rõ phần nào là prompt dùng được, phần nào còn chờ tích hợp
  model; script demo/config prompt cũ không được mô tả là dùng bộ 40 mới.
- Ngân sách 5–10 toàn ảnh và chọn QA ít trùng lặp là yêu cầu dành cho runner
  tương lai; không tuyên bố đã được cưỡng chế bởi phần nạp prompt.

## 8. Kiểm thử và điều kiện nghiệm thu

- Viết kiểm thử phần nối trước code; xác nhận thất bại rồi mới triển khai.
- Đủ 40 tài liệu không rỗng, 20 mỗi cấu hình, không trùng ánh xạ hoặc thiếu cặp.
- Kiểm tra mỗi prompt có mục tiêu, bỏ qua, bằng chứng và định dạng phù hợp;
  Diagnose giữ nhãn nguồn, Reasoning không biến thành cơ chế bệnh.
- Rendering không làm hỏng JSON khi nhãn có dấu, xuống dòng hoặc dấu ngoặc.
- Trường bệnh danh không xuất hiện trong JSON cấp cho nhiệm vụ nhìn ảnh;
  kiểm thử không chứng nhận dữ liệu tự do đã được làm sạch mọi rò rỉ.
- Context sai hoặc thiếu trường bắt buộc báo lỗi rõ. Không fallback task/type.
- Không có dependency Phase 2, SDK model, gọi mạng hoặc đọc ảnh trong loader.
- Script cũ dừng an toàn, không ghi QA/progress; không chạy full pipeline để thử.
- Chạy lại bộ kiểm thử offline Phase 2 để kiểm tra không làm ảnh hưởng phần đó.
- Kiểm tra diff chỉ nằm trong prompt/phần nối/hướng dẫn/kiểm thử và điểm chặn cũ.

Kiểm thử offline chỉ chứng minh phần nối và quy tắc đã hiện diện, không chứng
minh độ chính xác lâm sàng, hiệu quả của model, hoặc chất lượng đọc ảnh. Đánh
giá model trên ảnh thật là công việc riêng, ngoài phạm vi đã được phép.

## 9. Rủi ro và lựa chọn

Chọn 40 tài liệu độc lập theo điều chỉnh của người dùng. Không chọn nguồn
Phase 2 chung và không bắt buộc giữ module Python cũ.
Rủi ro: nội dung lặp có thể lệch khi sửa; kiểm thử và bảng kiểm duyệt hỗ trợ
phát hiện, nhưng không thay được kiểm tra ngữ nghĩa thủ công.
Giao diện cũ bị ngừng có chủ đích; thông báo chuyển đổi phải rõ để người dùng
không tiếp tục chạy script và tưởng dữ liệu đã được sinh thật.

Bước sau bản thiết kế: người dùng duyệt tài liệu, lập kế hoạch triển khai và
chọn cách thực hiện; sau đó mới sửa code/prompt và chạy kiểm thử offline.
