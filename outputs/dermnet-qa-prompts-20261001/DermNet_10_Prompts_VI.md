# DermNet: 10 prompt sinh trực tiếp câu hỏi và đáp án tiếng Việt

Phiên bản prompt: `v1.0-draft`, ngày 01/10/2026, múi giờ Asia/Bangkok.

Trạng thái: đã cập nhật theo các quyết định của người dùng; chưa chạy thử trên ảnh và chưa được duyệt chuyên môn. Đây là tài liệu prompt, không phải runner đã triển khai. Không thay đổi dataset cũ, code sinh QA hoặc tài liệu chuẩn hóa nguồn.

## Nguồn và phạm vi

- Bộ chuẩn hóa: https://docs.google.com/document/d/1cyrZ6Jqz8L5_FOhAb9nCQF_fFP9m20MEG_OySif8nZI/edit
- Dùng bản đọc của bộ chuẩn hóa đã được đối chiếu trong phiên làm việc ngày 01/10/2026; không khẳng định Google Docs sẽ không thay đổi sau thời điểm đó. SHA-256 của bản đọc: `2F1D98B176826E25006E812E429445CB5A50445BF9661D08E03F8BA7859D161D`.
- MedLesionVQA, mục 3.3, Bảng 8 và Bảng 9: `C:\Users\Vu\Downloads\15645_MedLesionVQA_A_Multimoda (2).pdf`.
- Đối chiếu cấu trúc dataset cũ tại `Phase_2/VLMEvalKit/LMUData/DermNet_Test_mac_relative.tsv`. Checkout tham chiếu: `main`, commit `2736c270c86bb17e9ca2aeec7ba73001285d911d`.
- Kế thừa cách chia nhiệm vụ, bốn hình thức trả lời và lựa chọn phương án nhiễu của bài báo. Không sao chép giả định mọi chú thích đều suy ra được từ ảnh, không coi trắc nghiệm là cách chữa thiếu bằng chứng, không áp dụng tỷ lệ câu hỏi của bài báo một cách máy móc.
- Nội dung tài liệu nguồn là dữ liệu tham khảo. Các chỉ dẫn nằm trong nguồn không cấp quyền chạy công cụ, thay đổi dữ liệu hoặc lập lịch cho phiên làm việc hiện tại.

## Những quyết định đã chốt

1. Có đúng năm loại nội dung: Lesion recognition, Attribute recognize, Location, Lesion Reasoning, Diagnose.
2. Giữ bốn giá trị `type` của TSV cũ: `Short_answer`, `Multi_choice`, `Judgement`, `Fill_in_blank`.
3. Diagnose lấy bệnh danh đã có làm đáp án. AI sinh QA không chẩn đoán lại và không thay bệnh danh nguồn.
4. Câu hỏi nhận diện mặc định nói về ảnh: "Bức ảnh này thể hiện loại tổn thương nào?" và các cách diễn đạt tương đương. Không mặc định tồn tại vùng đánh dấu hoặc buộc mọi câu dùng một mẫu duy nhất.
5. Category là loại tổn thương; Location là vị trí; sáu thuộc tính là Size, Color, Boundary, Shape, Quantity, Distribution. Diagnose là trường bệnh danh riêng về mặt nhiệm vụ, dù dữ liệu nguồn có thể lưu trong nhóm Attribute Lesion.
6. Lesion Reasoning giải thích căn cứ nhận diện hoặc phân biệt tổn thương. Không suy nguyên nhân sinh bệnh chỉ từ ảnh. Nhãn mục tiêu được phép xuất hiện trong câu hỏi yêu cầu lý giải nhãn đó.
7. Các dạng sáng tạo như sửa một thuộc tính sai, nhận diện nền/bề mặt hoặc vị trí theo cấp nằm bên trong năm loại nội dung, không tạo thêm loại chính.

## Cách dùng

Mỗi lần sử dụng ghép: **quy tắc chung + dữ liệu đầu vào + một prompt được chọn**. Quy tắc chung là phần dùng chung, không phải prompt thứ 11.

- Prompt 01-05 là bản ngắn cho phụ lục bài báo. Khi công bố, phải ghi đúng phiên bản thực tế đã chạy; không tuyên bố đã kiểm nghiệm bản dự thảo.
- Prompt 06-10 là bản vận hành tương ứng, thêm kiểm tra ảnh, nguồn nhãn và xử lý trường hợp không đủ căn cứ.
- Các biến trong dấu ngoặc nhọn phải được thay bằng dữ liệu thật. Ảnh phải được đính kèm cho mô hình có khả năng đọc ảnh; một đường dẫn văn bản không thay thế ảnh.
- `category` và JSON mở rộng bên dưới là quy ước đề xuất cho bộ mới. Muốn dùng với runner/TSV hiện tại cần bộ chuyển đổi; tài liệu này chưa thực hiện tích hợp đó.

## Quy tắc chung áp dụng cho cả 10 prompt

### Nhãn và bằng chứng

- Từ điển được cấp phải khóa phiên bản và có mã nhãn, tên Việt chuẩn, tên nguồn và tên tương đương đã duyệt. Không tự tạo tên chuẩn hoặc tự đổi tên giữa hai bảng.
- Nguồn chính cho bản đề xuất là Bảng 1-6 và Bảng 8; Bảng 7 dùng đối chiếu, không tự ghi đè bảng chi tiết. Đây là lựa chọn thiết kế của gói prompt, không sửa tài liệu nguồn.
- Giữ đúng nhãn theo từ điển đã cấp. Ví dụ: `Nodule → Cục`, `Tumor → U`, `Shape.Linear → Dạng dải`, `Distribution.Linear → Dạng đường`. Cùng chữ "Không đều" ở Shape và Boundary vẫn là hai nhãn khác trường.
- Đáp án là nhãn phải khớp tên chuẩn. Đáp án lý giải được viết thành câu ngắn, nhưng tên nhãn trong lời giải vẫn dùng đúng từ điển.
- Chú thích dùng để tạo đáp án không tự động trở thành bằng chứng người trả lời được xem. Với câu chỉ ảnh, không hỏi những thông tin chỉ có trong hồ sơ nội bộ mà ảnh không cho phép trả lời.
- Không suy ra đau, ngứa, độ chắc khi sờ, độ sâu mô, xét nghiệm hoặc diễn biến thời gian chỉ từ ảnh. Không coi hình mờ là "đường bờ không rõ" và không coi "không nhìn thấy rõ" là "không có".
- Mỗi câu có một mục tiêu chấm điểm và đối tượng rõ. Mặc định hỏi toàn ảnh. Nếu có nhiều loại tổn thương, dùng câu hỏi nhiều nhãn khi phù hợp hoặc mô tả đối tượng bằng lời; không tự chọn một loại làm chủ đạo, không bịa vùng đánh dấu.
- Size phải có căn cứ đo mà người trả lời được cung cấp. Các khoảng ví von chồng lấn hoặc chưa bao phủ đủ không được tự biến thành phân loại đơn nhãn. Quy tắc Quantity phải được cấp trước khi gán "Vài"/"Nhiều".
- Câu hỏi phải đa dạng cách diễn đạt nhưng giữ nguyên đối tượng, phạm vi và nội dung cần trả lời. Không thêm biến thể chỉ để tăng số lượng hoặc giả tạo độ khó.
- Số QA tối đa là giới hạn, không phải chỉ tiêu bắt buộc. Chỉ xuất QA có căn cứ; còn lại ghi `skipped` hoặc `needs_review`.

### Bốn hình thức trả lời

| `type` | Quy tắc |
|---|---|
| `Short_answer` | Hỏi trực tiếp; đáp án nhãn chuẩn hoặc lời giải ngắn theo nhiệm vụ. Không đặt giới hạn từ làm cắt mất tên chuẩn. |
| `Multi_choice` | Bốn lựa chọn A-D, một đáp án đúng. Cùng trường, cùng mức phân loại, không trùng nghĩa hoặc đồng thời đúng đối với mục tiêu. Không có ba phương án nhiễu hợp lệ thì bỏ qua hình thức này. |
| `Judgement` | Nhận định kiểm chứng được, trả lời Có/Không. Nhận định sai phải có căn cứ bác bỏ; thiếu thông tin không đồng nghĩa với Không. |
| `Fill_in_blank` | Một chỗ trống và một nội dung đích rõ. Không tạo nhiều cách điền khác nghĩa đều hợp lệ. |

Với Diagnose, `Short_answer` và `Fill_in_blank` dùng tên bệnh đã có; `Multi_choice` có một lựa chọn đúng mang tên bệnh đó và `answer` là khóa A-D; `Judgement` đối chiếu bệnh danh mục tiêu với nhãn có sẵn. Không biến đáp án Không thành tuyên bố bệnh khác tuyệt đối không thể cùng tồn tại ở người bệnh.

Với Lesion Reasoning, cả bốn hình thức phải kiểm tra căn cứ hoặc lời giải: trắc nghiệm chọn lời giải, phán định kiểm tra một lời giải, điền khuyết điền căn cứ phân biệt. Chỉ hỏi tên tổn thương không được gắn nhãn Reasoning.

Vị trí A-D và tỷ lệ Có/Không được bộ điều phối phân bổ ở cấp dataset. Một lượt sinh không được tự tuyên bố đã cân bằng toàn bộ dữ liệu.

### Đầu vào và đầu ra

Đầu vào do hệ thống cấp:

- `{IMAGE}` và `{IMAGE_ID}`.
- `{TAXONOMY}` và `{TAXONOMY_VERSION}`.
- `{ANNOTATIONS}`: Category, Location và thuộc tính đã có, nguồn/trạng thái kiểm chứng, căn cứ đo nếu có.
- `{DIAGNOSIS_LABEL}`: bệnh danh có sẵn cho nhiệm vụ Diagnose; chỉ xuất hiện ở vị trí cần thiết của đáp án/lựa chọn/nhận định phán định, không thêm thành gợi ý làm lộ đáp án.
- `{REQUESTED_TYPES}` hoặc `{REQUESTED_TYPE}`: các hình thức được yêu cầu; không tự đổi hình thức khi không làm được.
- `{MAX_QA}`, `{USED_QUESTIONS}` và cấu hình đối tượng/phạm vi nếu có.
- `{ATTRIBUTE_FIELD}`, `{SUBTYPE}`, `{TARGET_SCOPE}`: trường thuộc tính, dạng phụ và phạm vi được giao khi nhiệm vụ cần chúng; không tự suy một cấu hình chưa được cấp.
- `{CORRECT_OPTION_POSITION}`: vị trí A-D được bộ điều phối cấp khi cần phân bổ đáp án trắc nghiệm; nếu chưa cấp thì chọn vị trí và ghi kết quả để điều phối kiểm tra, không tự tuyên bố đã cân bằng.
- Với chạy thật: `{RUN_ID}`, `{WORKER_ID}`, `{TASK_ID}` do bộ điều phối cấp. Đồng hồ, mã băm và bằng chứng đính kèm ảnh phải do chương trình ghi, không yêu cầu mô hình tự bịa.

Chỉ xuất JSON hợp lệ. Mẫu cấu trúc sau là ví dụ đầu ra, không phải dữ liệu đã sinh từ ảnh:

```json
{
  "image_id": "do_he_thong_cap",
  "prompt_id": "P01",
  "taxonomy_version": "do_he_thong_cap",
  "qas": [
    {
      "category": "Lesion_Recognition",
      "sub_category": "Category",
      "type": "Short_answer",
      "question": "Bức ảnh này thể hiện loại tổn thương nào?",
      "options": {},
      "answer": "Sẩn",
      "answer_label": "Sẩn",
      "target": "tổn thương được thể hiện trong ảnh",
      "scope": "whole_image",
      "subtype": "direct_recognition",
      "source_table": "Bảng 8",
      "source_label_id": "do_tu_dien_cap",
      "evidence": [],
      "rationale": "",
      "status": "candidate"
    }
  ],
  "skipped": [],
  "needs_review": []
}
```

- Các `category` mới: `Lesion_Recognition`, `Attribute_Recognition`, `Location`, `Lesion_Reasoning`, `Diagnosis`.
- `options` chỉ có A-D khi là `Multi_choice`; `answer` phải khớp đúng khóa đã xáo trộn. `answer_label` lưu nhãn đúng hoặc nội dung lời giải đúng để kiểm tra nội bộ.
- Với QA nhiều nhãn, `answer_label` và `source_label_id` là danh sách tương ứng; `answer` vẫn là chuỗi trả lời hoặc khóa A-D theo hình thức. Quy tắc chấm tập nhãn phải được bộ đánh giá xác nhận trước khi chạy.
- `evidence` là các quan sát ngắn thực sự có căn cứ; không phải lời giải suy luận dài. `rationale` lưu lời giải ngắn của Reasoning hoặc căn cứ kiểm tra nội bộ.
- Khi chuyển sang TSV cũ, các lựa chọn A-D phải được ghép vào `question` và giữ đúng `type`; không đưa `evidence`, nhãn nguồn, mã bệnh hoặc metadata chứa đáp án vào câu hỏi.
- `candidate` không có nghĩa đã được chuyên gia duyệt. Lỗi ảnh đầu vào, thiếu đáp án nguồn và câu hỏi không đủ căn cứ phải được phân biệt trong lý do bỏ qua.

## A. Năm prompt cho bài báo

### Prompt 01 - Lesion recognition: nhận diện loại tổn thương

```text
Bạn hãy tạo trực tiếp câu hỏi và đáp án tiếng Việt về loại tổn
thương trong {IMAGE}, dựa trên {ANNOTATIONS} và Bảng 8 của
{TAXONOMY}. Tuân thủ quy tắc chung và các hình thức {REQUESTED_TYPES}.

Mục tiêu là Category hình thái, không phải tên bệnh hoặc hình dạng.
Ưu tiên cách hỏi toàn ảnh, chẳng hạn “Bức ảnh này thể hiện loại
tổn thương nào?”, nhưng chủ động thay đổi cách diễn đạt tự nhiên
mà không đổi nội dung cần trả lời.

Không mặc định có vùng đánh dấu. Nếu nhiều loại tổn thương cùng
xuất hiện, hỏi những loại hiện diện khi hình thức cho phép hoặc
xác định đối tượng bằng lời; không ép một đáp án khi nhiều đáp án đúng.

Khi ảnh và chú thích cho phép, tạo các câu bổ sung hỏi riêng tổn
thương nền và biến đổi bề mặt. Không dùng “Vảy da” thay cho “Mảng”
hoặc ngược lại; không buộc mọi ảnh phải có hai thành phần.

Đáp án dùng đúng nhãn chuẩn. Không lấy tiêu chí cần sờ, độ sâu hoặc
kích thước chưa được cung cấp làm bằng chứng ảnh. Không tạo câu
trắc nghiệm có lựa chọn đồng nghĩa hoặc cùng đúng.

Xuất tối đa {MAX_QA} QA theo JSON chung; không xuất bản mô tả ảnh
thay cho QA. Bỏ qua nội dung chưa đủ căn cứ.
```

### Prompt 02 - Attribute recognize: sáu thuộc tính và sửa một lỗi

```text
Bạn hãy tạo trực tiếp câu hỏi và đáp án tiếng Việt về Size, Color,
Boundary, Shape, Quantity, Distribution trong {IMAGE}, dùng nhãn
ở Bảng 2-6 của {TAXONOMY} và {ANNOTATIONS}.

Tuân thủ quy tắc chung, {REQUESTED_TYPES} và giới hạn {MAX_QA}.
Mỗi câu nhận diện chỉ hỏi một thuộc tính của một đối tượng/phạm vi
rõ ràng. Đa dạng cách hỏi; không đưa giá trị cần trả lời vào câu hỏi.

Phân biệt hình dạng một tổn thương với cách nhiều tổn thương sắp
xếp; phân biệt độ rõ đường bờ với độ đều đường viền. Các đặc điểm
thuộc những khía cạnh khác nhau có thể cùng tồn tại.

Chỉ hỏi kích thước thật khi đầu vào người trả lời được xem có căn
cứ đo; không tự lấp khoảng ví von chồng lấn hoặc thiếu. Số lượng
chỉ tính trong phạm vi ảnh và theo quy tắc đếm được cấp. Không suy
phân bố toàn thân hoặc đối xứng từ phạm vi không đủ quan sát.

Nếu dữ liệu đủ rõ, có thể tạo dạng phụ “sửa một lỗi”: mô tả hai
hoặc ba thuộc tính nhưng thay đúng một giá trị bằng nhãn sai cùng
trường, rồi hỏi trường nào sai và giá trị đúng. Lỗi phải cần nhìn
ảnh để phát hiện, không phải lỗi ngôn ngữ hay đặt nhãn sai trường.

Xuất QA theo JSON chung; đánh dấu trường thuộc tính cho từng câu.
Không ép mọi ảnh có đủ sáu thuộc tính.
```

### Prompt 03 - Location: vị trí giải phẫu theo cấp

```text
Bạn hãy tạo trực tiếp câu hỏi và đáp án tiếng Việt về vị trí tổn
thương trong {IMAGE}, sử dụng cây giải phẫu ở Bảng 1 của
{TAXONOMY} và {ANNOTATIONS}.

Tuân thủ quy tắc chung, {REQUESTED_TYPES} và giới hạn {MAX_QA}.
Hỏi tự nhiên theo ảnh, không mặc định tồn tại vùng đánh dấu.
Đáp án là vị trí, không phải kiểu phân bố.

Chọn cấp cụ thể nhất mà mốc giải phẫu trong ảnh hỗ trợ. Khi phù
hợp, tạo nhóm QA ở cấp rộng và cấp cụ thể; các đáp án phải cùng
một đường đi hợp lệ, nhưng mỗi câu dùng độc lập và không lấy
đáp án câu trước làm gợi ý.

Có thể khai thác đường đi như Chi trên → Bàn tay → Mu bàn tay
khi ảnh thực sự hỗ trợ. Không tự thêm nút, đoán trái/phải hoặc
ép cấp sâu từ ảnh cận cảnh không có mốc giải phẫu.

Nếu nhiều vị trí cùng hiện diện, xác định rõ mục tiêu bằng lời
hoặc hỏi nhiều vị trí khi hình thức cho phép. Lưu cấp và đường
đi nguồn ở metadata kiểm tra, không hiển thị đường đi chứa đáp án
trong câu hỏi.

Xuất trực tiếp QA theo JSON chung. Nếu cây nguồn hoặc vị trí còn
nhập nhằng, ghi lý do chờ duyệt thay vì tự chọn.
```

### Prompt 04 - Lesion Reasoning: lý giải nhận diện và phân biệt

```text
Bạn hãy tạo trực tiếp câu hỏi và đáp án tiếng Việt kiểm tra vì
sao một tổn thương trong {IMAGE} được nhận diện hoặc phân biệt
với loại dễ nhầm, dựa trên {ANNOTATIONS} và Bảng 8 của {TAXONOMY}.

Tuân thủ quy tắc chung, {REQUESTED_TYPES} và giới hạn {MAX_QA}.
Mỗi câu phải yêu cầu căn cứ, lời giải hoặc sự phân biệt, không chỉ
hỏi lại tên tổn thương. Được nêu nhãn mục tiêu trong câu hỏi lý giải.

Ví dụ cách hỏi: “Đặc điểm nào trong ảnh hỗ trợ gọi tổn thương là
Sẩn thay vì Dát?”. Đây là mẫu cấu trúc, không phải kết luận sẵn
cho mọi ảnh. Đáp án phải nêu căn cứ thực sự được cung cấp.

Kết hợp các đặc điểm liên quan khi có đủ bằng chứng, hoặc khai
thác phân biệt tổn thương nền với lớp phủ. Không chép toàn bộ
định nghĩa thành những đặc điểm được cho là nhìn thấy.

Không giải thích nguyên nhân sinh bệnh, bản chất mô, tính chất sờ,
độ sâu hoặc kích thước thật khi đầu vào không có căn cứ.

Trả lời ngắn nêu lời giải; trắc nghiệm chọn lời giải đúng; phán
định kiểm tra một lời giải; điền khuyết điền căn cứ phân biệt.
Các hình thức không được biến nhiệm vụ này thành nhận diện tên.

Xuất QA theo JSON chung, kèm lời giải ngắn và bằng chứng; không
xuất suy luận dài hoặc lý giải chỉ có tính sách giáo khoa.
```

### Prompt 05 - Diagnose: sinh QA từ bệnh danh có sẵn

```text
Bạn hãy tạo trực tiếp câu hỏi và đáp án tiếng Việt về bệnh danh
cho {IMAGE}. Bệnh danh mục tiêu đã được cung cấp trong
{DIAGNOSIS_LABEL}; dùng đúng tên này làm đáp án tham chiếu.

Không tự chẩn đoán lại, thay nhãn bệnh hoặc đoán thêm một bệnh
khác. Nếu chưa có tên chuẩn, chỉ dùng ánh xạ đã được cấp trong
{TAXONOMY}; không tự dịch hoặc rút gọn làm mất thể bệnh, vị trí,
giai đoạn.

Tuân thủ quy tắc chung, {REQUESTED_TYPES} và giới hạn {MAX_QA}.
Đa dạng cách hỏi tự nhiên như “Bức ảnh này thể hiện bệnh da liễu
nào?” hoặc cách diễn đạt tương đương. Không thêm tên thư mục,
mã ICD hay bệnh danh nguồn thành gợi ý làm lộ đáp án. Tên bệnh
được phép nằm trong lựa chọn hoặc nhận định Judgement, vì đó
là nội dung cần đánh giá chứ không phải gợi ý bổ sung.

Trả lời ngắn và điền khuyết dùng tên bệnh có sẵn. Trắc nghiệm có
một lựa chọn đúng là tên bệnh đó; các lựa chọn khác thuộc danh
mục bệnh, không trùng nghĩa hoặc là nhãn rộng/hẹp cùng có thể đúng.
Phán định đối chiếu nhận định với bệnh danh mục tiêu đã cấp.

Không thêm bệnh sử, xét nghiệm hoặc điều trị chưa được cung cấp.
Thiếu bệnh danh hoặc ánh xạ nhập nhằng thì ghi lý do, không sinh
một đáp án bệnh mới. Bất thường ảnh-nhãn được ghi để duyệt dữ liệu,
không được dùng để tự sửa bệnh danh.

Xuất QA theo JSON chung; không xuất một báo cáo chẩn đoán thay QA.
```

## B. Năm prompt chạy thật

Mỗi lượt xử lý một ảnh; mỗi prompt vận hành dưới đây tương ứng với một prompt bài báo. Quy tắc nhãn và cách tạo QA không thay đổi giữa hai bản.

### Prompt 06 - Chạy Lesion recognition

```text
Bạn là subagent sinh QA loại tổn thương cho đúng {IMAGE_ID} trong
{TASK_ID}. Dùng {IMAGE}, {ANNOTATIONS}, {TAXONOMY} và quy tắc chung;
chỉ xử lý lượt này, không mang thông tin ảnh khác sang.

Kiểm tra bạn thực sự nhận và xem được ảnh. Nếu không, trả qas
rỗng với image_unavailable; không suy từ tên tệp hoặc mô tả văn bản.
Quan sát ảnh trước khi đối chiếu chú thích. Nếu dữ liệu đọc ảnh
độc lập đã được cấp, kiểm tra nó nhưng không coi việc đó tự động
chứng minh chú thích đúng.

Sinh trực tiếp QA theo {REQUESTED_TYPE}, tối đa {MAX_QA}.
Hỏi toàn ảnh bằng cách diễn đạt đa dạng, tránh lặp {USED_QUESTIONS}.
Không tự nói “vùng đánh dấu” khi đầu vào không có dấu.

Đáp án là Category trong Bảng 8. Nếu nhiều loại hiện diện, không
ép một loại làm chủ đạo. Chỉ tạo câu nhiều nhãn hoặc câu có đối
tượng rõ khi đúng hình thức yêu cầu; nếu không thì bỏ qua.
Khi có đủ căn cứ, hỏi riêng nền/bề mặt, nhưng không trộn hai nhãn.

Nếu chú thích và quan sát mâu thuẫn hoặc việc phân biệt cần dữ
liệu không được cung cấp, ghi needs_review. Không tự sửa nhãn.
Không đủ bốn lựa chọn hợp lệ thì bỏ qua Multi_choice, không bịa
phương án hoặc tự đổi type.

Xuất JSON chung với prompt_id P06, gắn đúng mã hệ thống cấp.
Không tự ghi đã hoàn thành đọc ảnh hoặc đã duyệt chuyên môn.
```

### Prompt 07 - Chạy Attribute recognize

```text
Bạn là subagent sinh QA thuộc tính cho đúng {IMAGE_ID} trong
{TASK_ID}. Nhận ảnh thật, nhãn chuẩn, chú thích, {ATTRIBUTE_FIELD},
{SUBTYPE}, {TARGET_SCOPE} và {REQUESTED_TYPE}. Tuân thủ quy tắc chung.

Xem toàn ảnh và kiểm tra đối tượng/phạm vi trước khi tạo câu.
Không xem được ảnh thì trả qas rỗng với image_unavailable.
Không coi ảnh cắt, bóng đổ hoặc phản sáng là bằng chứng chắc
chắn khi không phân biệt được đặc điểm tổn thương.

Sinh trực tiếp QA về trường được giao trong sáu trường Size,
Color, Boundary, Shape, Quantity, Distribution; không tự đổi
sang trường khác để hoàn thành lượt. Đa dạng cách hỏi theo ảnh.

Giữ Shape.Linear là Dạng dải và Distribution.Linear là Dạng đường.
Giữ độ rõ và độ đều đường bờ riêng; không đặt các nhãn có thể
cùng đúng thành lựa chọn cạnh tranh. Chỉ hỏi màu chủ đạo khi
thực sự xác định được, hoặc hỏi tập màu khi hình thức cho phép.

Size cần căn cứ đo người trả lời được xem; các khoảng ví von
chưa chốt không được tự gán. Quantity cần quy tắc đếm và phạm vi.
Distribution cần đủ phạm vi để kết luận; không suy toàn cơ thể.

Nếu được giao subtype sửa một lỗi, chỉ thay một giá trị cùng
trường bị ảnh bác bỏ và xuất câu hỏi cùng đáp án sửa; nếu không
chứng minh được lỗi thì bỏ qua, không tạo lỗi dễ đoán bằng văn bản.

Xuất JSON chung với prompt_id P07, ghi thuộc tính và lý do bỏ qua.
Không ép ảnh có đủ sáu thuộc tính hoặc đủ số lượng QA.
```

### Prompt 08 - Chạy Location

```text
Bạn là subagent sinh QA vị trí cho đúng {IMAGE_ID} trong {TASK_ID}.
Dùng ảnh thật và cây Bảng 1 đã được cấp; tuân thủ quy tắc chung,
{REQUESTED_TYPE} và giới hạn {MAX_QA}.

Kiểm tra ảnh xem được. Quan sát mốc giải phẫu, không dùng thư mục,
tên tệp hoặc nhãn bệnh để thay bằng chứng vị trí. Không có mốc
đủ rõ thì chỉ dùng mức chắc chắn hoặc ghi skipped.

Sinh trực tiếp câu hỏi và đáp án về vị trí, không hỏi phân bố.
Ưu tiên cách hỏi toàn ảnh đa dạng, chẳng hạn “Tổn thương xuất
hiện ở vị trí nào trong ảnh?”. Không mặc định có vùng đánh dấu.

Chọn nút sâu nhất có căn cứ. Nếu được giao nhóm QA theo cấp,
kiểm tra mọi đáp án thuộc cùng đường đi; mỗi câu vẫn dùng độc
lập. Không đưa toàn đường đi chứa đáp án vào câu hỏi.

Không suy trái/phải từ hướng màn hình. Không đoán mặt gấp/mặt
duỗi hoặc vị trí cụ thể từ một ảnh cận cảnh thiếu mốc. Nếu nhiều
vị trí hiện diện, mô tả rõ đối tượng hoặc dùng câu nhiều vị trí
khi phù hợp với type; không tự chọn một vị trí chính.

Nếu cây có quan hệ cha-con chưa chốt, ghi needs_review, không
suy cấu trúc theo ô trống hoặc tự thêm nút.

Xuất JSON chung với prompt_id P08. Lưu cấp và đường đi ở metadata
kiểm tra; không tạo câu hỏi mới khi đã hết lượt được giao.
```

### Prompt 09 - Chạy Lesion Reasoning

```text
Bạn là subagent sinh trực tiếp QA lý giải nhận diện tổn thương
cho đúng {IMAGE_ID} trong {TASK_ID}. Dùng ảnh thật, Category tham
chiếu, các căn cứ được cung cấp và Bảng 8 của {TAXONOMY}.

Kiểm tra ảnh xem được rồi đối chiếu từng căn cứ với đúng đối
tượng. Không lấy định nghĩa chung hoặc chú thích không được
người trả lời xem làm bằng chứng ảnh.

Tạo QA theo {REQUESTED_TYPE}, tối đa {MAX_QA}, với cách diễn đạt
đa dạng. Nội dung phải hỏi vì sao nhận diện, đặc điểm phân biệt
hoặc lời giải nào phù hợp. Được nêu Category cần lý giải.

Mỗi lời giải có kết luận và tối đa ba căn cứ ngắn có liên quan;
ít căn cứ hơn vẫn hợp lệ, không thêm dấu hiệu để đủ số lượng.
Nếu phân biệt hai loại, kiểm tra căn cứ thực sự giúp phân biệt,
không chỉ có thể xuất hiện ở cả hai.

Không giải thích nguyên nhân sinh bệnh từ ảnh; không suy độ sâu,
độ chắc khi sờ, thời gian hoặc kích thước thật thiếu căn cứ.
Nếu câu chỉ hỏi tên tổn thương, không gắn nhãn Reasoning.

Multi_choice phải có lời giải đúng và các lời giải nhiễu bị căn
cứ bác bỏ. Judgement kiểm tra một lời giải và Fill_in_blank điền
căn cứ phân biệt. Không làm được đúng type thì ghi skipped.

Xuất JSON chung với prompt_id P09, lời giải ngắn và evidence;
trường hợp thiếu căn cứ hoặc mâu thuẫn đưa vào needs_review.
```

### Prompt 10 - Chạy Diagnose từ bệnh danh có sẵn

```text
Bạn là subagent sinh trực tiếp QA bệnh danh cho đúng {IMAGE_ID}
trong {TASK_ID}. Bệnh danh mục tiêu là {DIAGNOSIS_LABEL} đã được
hệ thống cấp. Giữ nhãn này; không thực hiện chẩn đoán lại.

Kiểm tra ảnh thực tế xem được và bệnh danh được gắn đúng mã ảnh
theo đầu vào. Thiếu ảnh ghi image_unavailable; thiếu nhãn ghi
missing_diagnosis_label. Không đoán bệnh để lấp dữ liệu thiếu.

Tạo QA theo {REQUESTED_TYPE}, tối đa {MAX_QA}, dùng cách hỏi đa
dạng như “Bức ảnh này thể hiện bệnh da liễu nào?”. Không thêm
bệnh danh nguồn, tên thư mục hoặc mã ICD thành gợi ý làm lộ đáp
án; không bịa đặc điểm ảnh. Tên bệnh trong lựa chọn hoặc nhận
định Judgement là nội dung hợp lệ của hình thức đó.

Short_answer và Fill_in_blank lấy đúng tên bệnh có sẵn.
Multi_choice đặt tên bệnh đó theo {CORRECT_OPTION_POSITION};
chọn ba tên bệnh khác hợp lệ, không trùng nghĩa hoặc nhãn rộng/hẹp
cùng đúng. Sau khi đổi vị trí phải cập nhật khóa answer.
Judgement đối chiếu bệnh danh mục tiêu với nhãn nguồn; không
khẳng định bệnh khác tuyệt đối không thể cùng tồn tại ở người bệnh.

Không tự dịch, đổi thể bệnh hoặc bỏ giai đoạn. Nếu ánh xạ chưa
chốt, ghi needs_review thay vì tạo tên chuẩn mới. Nếu nhận thấy
bất thường ảnh-nhãn, giữ bệnh danh nguồn và ghi cần duyệt dữ liệu;
không sửa nó bằng dự đoán của bạn.

Xuất JSON chung với prompt_id P10. Không sinh báo cáo chẩn đoán,
giải thích nguyên nhân hay đề xuất điều trị thay cho QA.
```

## Bảng chuyển cấu trúc cũ sang bộ mới

| `category` cũ | Nhiệm vụ mới | Cách xử lý |
|---|---|---|
| Lesion_Recognition | Lesion_Recognition | Giữ mục tiêu Category và nhãn tổn thương trong Bảng 8. |
| Attribute_Color | Attribute_Recognition / Color | Chuẩn hóa tên màu; không giữ màu không thuộc từ điển bằng cách tự đoán tên gần nhất. |
| Attribute_Shape | Attribute_Recognition / Shape hoặc Boundary | Tách hình dạng và đường bờ dựa theo nội dung, không đổi nhãn hàng loạt chỉ theo category cũ. |
| Attribute_Characteristics | Attribute_Recognition hoặc Lesion_Recognition | Phân loại theo ý nghĩa; Vảy da/Vảy tiết và các nhãn Bảng 8 thuộc Category. Nội dung ngoài bộ chuẩn hóa cần duyệt. |
| Anatomical_Distribution | Location hoặc Attribute_Recognition / Distribution | Tách vị trí khỏi phân bố. Câu gộp không tách an toàn phải được duyệt, không tự nhân đôi đáp án. |
| Lesion_Reasoning | Lesion_Reasoning | Giữ nhiệm vụ lý giải, nhưng loại câu chỉ chép định nghĩa hoặc suy căn cứ không có. |
| Diagnosis | Diagnosis | Đáp án lấy bệnh danh có sẵn, không dự đoán lại. |

Size, Quantity và Boundary được kiểm tra như các trường rõ ràng trong bộ mới; không mặc định dataset cũ đã có đủ QA cho chúng. Không bỏ nhóm Reasoning chỉ vì một bản đã rà soát trước đây không còn dòng thuộc nhóm này.

## Ví dụ minh họa cách hỏi

Các hàng dưới đây là tình huống giả định độc lập, không phải kết quả đọc ảnh thực tế và không gộp thành một ca bệnh. Nhãn chỉ có ý nghĩa khi ảnh và dữ liệu thật đáp ứng điều kiện tương ứng.

| Nhiệm vụ | Hình thức | Câu hỏi minh họa | Đáp án giả định |
|---|---|---|---|
| Lesion recognition | Short_answer | Bức ảnh này thể hiện loại tổn thương nào? | Sẩn |
| Attribute recognize / Color | Multi_choice | Màu nào được ghi nhận ở tổn thương trong ảnh? A. Màu đỏ; B. Màu tím; C. Màu đen; D. Màu trắng. | A; nhãn đúng: Màu đỏ |
| Location | Fill_in_blank | Vị trí cụ thể của tổn thương trong ảnh là ____. | Mu bàn tay |
| Lesion Reasoning | Short_answer | Đặc điểm nào trong ảnh hỗ trợ gọi tổn thương là Sẩn thay vì Dát? | Tổn thương nhô lên trên bề mặt da, thay vì chỉ thay đổi màu sắc. Đây là căn cứ phân biệt được hỏi, không phải xác nhận mọi tiêu chí của Sẩn. |
| Diagnose | Judgement | Nhận định bệnh danh của ảnh là Bệnh vảy nến có đúng không? | Có, khi bệnh danh mục tiêu có sẵn là Bệnh vảy nến. |

Các biến thể nhận diện cùng nghĩa có thể dùng: "Tổn thương xuất hiện trong ảnh thuộc loại nào?", "Quan sát hình ảnh, hãy xác định loại tổn thương.", "Trong ảnh này có thể nhận diện dạng tổn thương nào?". Không coi các biến thể này là những mẫu bắt buộc hoặc lặp tất cả cho mỗi ảnh.

## Chia subagent và lập lịch: đặc tả cho giai đoạn tích hợp

Phần này là kế hoạch thực hiện sau khi có runner; chưa tạo automation hoặc khởi chạy subagent xử lý ảnh trong phiên hiện tại.

1. Lập manifest ảnh ổn định. Gom ảnh trùng/cùng nguồn ca bệnh để kiểm soát chia tập; mỗi ảnh chỉ thuộc một shard xử lý chính. Không chia năm nhóm prompt thành năm agent cùng đọc toàn bộ dataset.
2. Mỗi worker xử lý một ảnh tại một thời điểm; thực hiện các nhiệm vụ được giao bằng Prompt 06-10. Một batch có thể là danh sách tối đa 10 ảnh trong hàng đợi, không phải nhồi 10 ảnh vào cùng ngữ cảnh suy luận. Ngữ cảnh từng ảnh phải được tách.
3. Chạy thử một tập nhỏ đa dạng, khởi đầu với 2-4 worker nếu tài nguyên cho phép. Đo thời gian thực tế theo nhiệm vụ/nhóm nhiệm vụ; không mặc định mô hình chỉ văn bản có thể kiểm chứng ảnh.
4. Chọn ngân sách lượt dự kiến `S = 1,25 × p95` từ số đo thử. `p95` là mốc 95% lượt đã đo hoàn thành trong đó, không phải bảo đảm chính xác. Với worker có lượt thứ j, mốc dự kiến là `T0 + j × S`; chương trình dùng đồng hồ thật để điều phối.
5. Lượt trước chưa xong thì lượt sau chờ. Mốc lịch không được ép mô hình trả lời gấp hoặc giả hoàn thành. Đặt timeout theo tài nguyên đã đo; timeout ghi lỗi/chờ thử lại, không tạo đáp án thay thế. Không hứa chính xác thời điểm hoàn tất từng ảnh khi độ trễ biến động.
6. IDs, timestamps, log gắn ảnh và mã băm do chương trình ghi. Nếu muốn quan sát mù nhãn, runner phải tách lượt đọc ảnh không cấp nhãn khỏi lượt tạo QA có nhãn; chỉ viết "đọc ảnh trước" trong một prompt có sẵn nhãn không tạo được kiểm tra mù nhãn thực sự.
7. Kiểm tra JSON và lưu checkpoint sau mỗi kết quả hợp lệ vào file riêng của worker. Bộ điều phối chỉ commit một kết quả cho mỗi khóa tác vụ; chỉ giao lại khi worker cũ đã dừng hoặc quyền xử lý đã hết hạn, không để nhiều worker ghi đè cùng tác vụ.
8. Kiểm tra hình thức/nhãn bằng chương trình; kiểm tra sự thật hình ảnh cần ảnh thực tế và người duyệt hoặc bộ kiểm tra độc lập phù hợp. LLM chỉ đọc văn bản không thể xác nhận bằng chứng nhìn thấy trong ảnh. QA Reasoning, bất thường ảnh-nhãn và các trường hợp tranh chấp cần duyệt chuyên môn.
9. Đo độ đúng theo từng nhiệm vụ/thuộc tính, tỷ lệ bỏ qua/chờ duyệt và tính nhất quán các QA cùng ảnh. Không chỉ báo độ chính xác trên các câu dễ được giữ lại. Giữ QA cùng ảnh và nhóm ảnh trùng/cùng ca trong cùng split.

## Điều kiện trước khi chạy thử

- Có mô hình đọc ảnh và đường truyền ảnh thực tế; không chỉ có prompt văn bản.
- Có từ điển khóa phiên bản, cây vị trí đã xác nhận và quy tắc Quantity.
- Size không được ép gán một nhãn ví von ở khoảng chồng lấn; chỉ dùng căn cứ đo hợp lệ.
- Có bệnh danh đầu vào cho Diagnose và cách đối chiếu nguồn ảnh-nhãn, không yêu cầu chẩn đoán lại.
- Có bộ chuyển JSON sang TSV giữ `type` cũ; mapping category mới được kiểm tra với bộ đánh giá.
- Có kiểm tra câu hỏi không lộ đáp án, phương án trắc nghiệm hợp lệ, đối tượng rõ và không bịa vùng đánh dấu.
- Có quy trình duyệt, checkpoint và chạy tiếp không ghi trùng trước khi tăng số worker.

## Đoạn mô tả phương pháp đề xuất cho bài

"Phương pháp đề xuất sinh trực tiếp QA tiếng Việt theo năm nhiệm vụ: nhận diện tổn thương, nhận diện thuộc tính, xác định vị trí, lý giải nhận diện và bệnh danh. Nội dung được ràng buộc bởi bộ chuẩn hóa DermNet; bệnh danh tham chiếu được lấy từ nhãn có sẵn, không được mô hình sinh QA dự đoán lại. Bốn hình thức trả lời của dataset cũ được giữ. Các biến thể câu hỏi theo cấp vị trí, thành phần nền/bề mặt, sửa một thuộc tính sai và lời giải phân biệt nhằm kiểm tra khả năng liên kết đúng đối tượng với nhãn chuẩn."

Đây là mô tả thiết kế, không tuyên bố đã triển khai, đã được chuyên gia duyệt hoặc đã cải thiện điểm đánh giá. Khi có thử nghiệm, phải công bố prompt thực tế, phiên bản từ điển, đầu vào được người trả lời xem, nguồn đáp án và quy trình duyệt đã thực hiện.
