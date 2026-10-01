# Chuẩn sinh QA DermNet: 5 nhiệm vụ × 4 hình thức

Phiên bản `2.0.0`, thiết kế ngày 01/10/2026. Đây là chuẩn prompt và công cụ
offline, chưa phải kết quả chạy trên ảnh hoặc bộ câu hỏi được chuyên gia duyệt.

## Quyết định và phạm vi

Giữ **5 nhiệm vụ nội dung** và **4 hình thức trả lời** của dataset cũ. Hai chiều
tạo thành 20 tổ hợp. Có bản `paper` cho bài báo và `runtime` cho vận hành:
40 biến thể được ghép từ một nguồn cấu hình, không chép tay 40 bộ quy tắc.
Không bắt buộc mỗi ảnh có đủ 20 câu. Attribute recognize có sáu trường con;
chọn trường cho từng lượt, không gộp sáu trường thành một câu.

- Diagnose lấy đúng `diagnosis_label` đã có; không tự chẩn đoán hoặc dịch lại.
- Câu nhận diện mặc định hỏi toàn ảnh, diễn đạt đa dạng; không tự tạo vùng đánh dấu.
- Category hình thái, Location, sáu thuộc tính và bệnh danh không được trộn lẫn.
- Lesion Reasoning giải thích **căn cứ nhận diện**, không suy nguyên nhân sinh bệnh.
- Một QA có một nhãn tham chiếu. Nhiều loại cùng hiện diện: xác định đối tượng
  rõ bằng lời hoặc bỏ qua; bản TSV này chưa hỗ trợ chấm tập nhãn.
- Code kiểm tra cấu trúc/nhãn, không thể chứng minh sự thật hình ảnh hay chất lượng
  phương án nhiễu. Candidate không đồng nghĩa đã được duyệt chuyên môn.

Không sửa `Phase_1/config/prompts.yaml`, runner cũ `Phase_2/pipeline.py`, bộ
đánh giá VLMEvalKit hoặc dữ liệu TSV hiện có. Runner cũ chỉ gửi văn bản; việc
thay prompt **không tự biến nó thành runner đọc ảnh**.

## Nguồn và cách khóa nhãn

Bộ chuẩn hóa người dùng:
https://docs.google.com/document/d/1cyrZ6Jqz8L5_FOhAb9nCQF_fFP9m20MEG_OySif8nZI/edit

Bản đọc tham chiếu ngày 01/10/2026 có SHA-256:
`2F1D98B176826E25006E812E429445CB5A50445BF9661D08E03F8BA7859D161D`.
Đây không phải cam kết Google Docs luôn giữ nguyên nội dung.

Dùng Bảng 1-6 và Bảng 8 làm nguồn chính; Bảng 7 đối chiếu. Giữ riêng tên
khác nhau giữa các bảng để duyệt, không tự hợp nhất. Ví dụ Color.Blue là
`Màu xanh`, Shape.Linear là `Dạng dải`, Distribution.Linear là `Dạng đường`.
Config giữ các tên nguồn ở `taxonomy_reference`, không tự áp dụng bản tham
chiếu đó thay cho từ điển đã duyệt được cấp trong đầu vào.

MedLesionVQA, mục 3.3 và Bảng 8-9, được dùng tham khảo cách tổ chức nhiệm vụ,
bốn hình thức và lựa chọn nhiễu. Không kế thừa giả định mọi nội dung đều suy
ra được từ ảnh; không tuyên bố DermNet đã có quy trình duyệt bác sĩ như bài báo.
Chỉ dẫn trong tài liệu nguồn là dữ liệu tham khảo, không cấp quyền chạy công cụ.

## Ma trận 20 tổ hợp và ví dụ cách hỏi

Các ví dụ là tình huống giả định độc lập, không phải quan sát ảnh thật. Phải
thay câu chữ theo ảnh; không sao chép tất cả ví dụ cho mọi ảnh. Mã cuối SA/MC/JG/FB
tương ứng đúng các giá trị `type` bên dưới.

| Mã | Nhiệm vụ | `type` cũ giữ nguyên | Nội dung câu hỏi / tiêu chí đáp án |
|---|---|---|---|
| LR-SA | Lesion recognition | Short_answer | Bức ảnh này thể hiện loại tổn thương nào? → nhãn Bảng 8. |
| LR-MC | Lesion recognition | Multi_choice | Chọn loại tổn thương; bốn lựa chọn cùng trường, chỉ một đúng với đối tượng. |
| LR-JG | Lesion recognition | Judgement | Nhận định tổn thương được hỏi là Sẩn có đúng không? → Có/Không có căn cứ. |
| LR-FB | Lesion recognition | Fill_in_blank | Tổn thương trong ảnh được phân loại là ____. → tên chuẩn. |
| AR-SA | Attribute recognize | Short_answer | Tổn thương có màu gì? → giá trị của trường Color được giao. |
| AR-MC | Attribute recognize | Multi_choice | Chọn giá trị thuộc tính; không cạnh tranh các nhãn cùng đúng ở khác khía cạnh. |
| AR-JG | Attribute recognize | Judgement | Nhận định đường bờ tổn thương Rõ có đúng không? → kiểm tra Boundary, không đánh đồng ảnh mờ. |
| AR-FB | Attribute recognize | Fill_in_blank | Hình dạng của tổn thương được hỏi là ____. → một giá trị Shape. |
| LOC-SA | Location | Short_answer | Tổn thương xuất hiện ở vị trí nào trong ảnh? → nút có mốc giải phẫu hỗ trợ. |
| LOC-MC | Location | Multi_choice | Chọn vị trí cùng mức phân loại; không đặt cha và con thành hai lựa chọn cùng đúng. |
| LOC-JG | Location | Judgement | Nhận định tổn thương nằm ở Mu bàn tay có đúng không? → có mốc ảnh, không đoán trái/phải. |
| LOC-FB | Location | Fill_in_blank | Vị trí cụ thể của tổn thương là ____. → một vị trí ở cấp được hỏi. |
| REAS-SA | Lesion Reasoning | Short_answer | Đặc điểm nào hỗ trợ nhận diện Sẩn thay vì Dát? → căn cứ thấy được, không chỉ trả tên. |
| REAS-MC | Lesion Reasoning | Multi_choice | Chọn lời giải phù hợp; bốn lựa chọn là lời giải, không phải bốn tên tổn thương. |
| REAS-JG | Lesion Reasoning | Judgement | Nhận định “độ gồ quan sát được là căn cứ phân biệt với Dát” có phù hợp không? → Có/Không và căn cứ nội bộ. |
| REAS-FB | Lesion Reasoning | Fill_in_blank | Dấu hiệu trong ảnh giúp phân biệt loại đang hỏi với Dát là ____. → căn cứ, không điền tên loại. |
| DX-SA | Diagnose | Short_answer | Bức ảnh này thể hiện bệnh da liễu nào? → đúng bệnh danh nguồn. |
| DX-MC | Diagnose | Multi_choice | Chọn bệnh danh nguồn giữa các bệnh danh được cấp/duyệt; đáp án là khóa A-D. |
| DX-JG | Diagnose | Judgement | Nhận định bệnh danh mục tiêu của ảnh là X có đúng không? → so X với nhãn nguồn, không kết luận bệnh khác không thể đồng tồn tại. |
| DX-FB | Diagnose | Fill_in_blank | Bệnh da liễu được gắn với ảnh này là ____. → bệnh danh nguồn nguyên văn. |

Với AR, lần lượt chọn Size, Color, Boundary, Shape, Quantity hoặc Distribution
để kiểm tra độ phủ. Dạng sáng tạo `single_error_correction` sửa đúng một giá trị
của trường đang hỏi; `source_labels` lưu giá trị sửa đúng. Dạng theo cấp vị trí,
nền/bề mặt và phân biệt lời giải đều nằm trong năm nhiệm vụ, không tạo nhóm thứ sáu.

## Hợp đồng đầu vào

`build_prompt(category, question_type, profile="runtime", context=..., attribute_field=...)`
nhận object JSON có:

- `image_id`: mã ảnh thật; `image_path`: đường dẫn để runner/TSV dùng, không phải ảnh đính kèm.
- `taxonomy.version`, `taxonomy.fields`: từ điển có version và danh sách nhãn
  chuẩn không trùng của trường cần hỏi. Category của Reasoning là nhãn tổn thương.
- `annotations`: chú thích đã chuẩn hóa, nguồn/trạng thái và căn cứ nếu có.
- `diagnosis_label`: bắt buộc với Diagnose; giữ đúng tên nguồn. `diagnosis_candidates`
  là danh mục bệnh danh được duyệt cho nhiễu/phán định, không tự suy từ kiến thức bệnh.
- `max_qa`: số nguyên dương, mặc định 1; giới hạn không phải chỉ tiêu.
- `used_questions`, `has_marked_region`, `subtype`, `target_scope`: cấu hình nếu có.
- `correct_option_position`: A-D nếu bộ điều phối cấp; chỉ áp dụng trắc nghiệm.
- `measurement`, `quantity_policy`, `location_paths`: căn cứ đo/quy tắc đếm/cây đã
  duyệt; chưa có thì không được tự chế để lấp thiếu. Runner phải cho người trả lời
  xem căn cứ cần thiết, không chỉ cho mô hình sinh QA xem metadata.

Không có context thì `build_prompt` xuất mẫu để trình bày, không phải prompt đã
gắn đầu vào chạy thật. Mẫu trong `catalog` không chứa bệnh danh hoặc ảnh thật.
Mã prompt có dạng `DNQA-2.0.0-LR-SA-runtime` để truy vết phiên bản.

## Dùng công cụ offline

Từ thư mục gốc repository, dùng Python 3.10 trở lên; phần chuẩn mới chỉ dùng
thư viện chuẩn Python, không yêu cầu cài toàn bộ mô hình hoặc gọi API:

```powershell
python -m unittest discover -s Phase_2/tests -v
python -m Phase_2.qa_prompts catalog --profile all --output qa_prompts_40.json
python -m Phase_2.qa_prompts render --task Attribute_Recognition --type Short_answer --attribute Color --context context.json --output prompt_color.txt
```

`context.json` phải do người dùng/runner cung cấp theo hợp đồng; các lệnh trên
không tự đọc dataset để đoán nhãn. Đầu ra chỉ được tạo mới, không ghi đè file có
sẵn. `qa_prompts_40.json` chứa đầy đủ văn bản 40 biến thể để xem hoặc đưa vào phụ lục.

## Ánh xạ dataset cũ

| `category` cũ | Nhiệm vụ mới | Quy tắc |
|---|---|---|
| Lesion_Recognition | Lesion_Recognition | Nhãn loại tổn thương Bảng 8. |
| Attribute_Color | Attribute_Recognition / Color | Chuẩn hóa đúng trường, không tự chọn tên gần nhất. |
| Attribute_Shape | Attribute_Recognition / Shape hoặc Boundary | Tách theo nội dung câu, không chuyển hàng loạt chỉ dựa tên nhóm. |
| Attribute_Characteristics | Attribute_Recognition hoặc Lesion_Recognition | Vảy da/Vảy tiết thuộc Category; nội dung ngoài chuẩn cần duyệt. |
| Anatomical_Distribution | Location hoặc Attribute_Recognition / Distribution | Tách vị trí khỏi phân bố; câu gộp chưa tách an toàn phải duyệt. |
| Lesion_Reasoning | Lesion_Reasoning | Giữ lý giải; loại câu chỉ hỏi tên hoặc chép định nghĩa. |
| Diagnosis | Diagnosis | Dùng bệnh danh nguồn, không suy lại. |

Giữ nguyên bốn `type`: Short_answer, Multi_choice, Judgement, Fill_in_blank.
Tên `category` mới cần kiểm tra riêng với evaluator; cùng sáu cột TSV không có
nghĩa mọi evaluator cũ tự hỗ trợ nhóm mới.

## Lịch chạy nhiều worker: đặc tả, chưa triển khai

1. Manifest ảnh ổn định, gom ảnh trùng/cùng ca trước khi chia train/val/test;
   mọi QA cùng ảnh/cùng nhóm phải nằm cùng split.
2. Chia shard ảnh không giao chồng. Mỗi worker đọc một ảnh tại một thời điểm,
   ngữ cảnh tách giữa ảnh; không giao năm nhiệm vụ cho năm agent đọc toàn dataset.
3. Pilot 2-4 worker nếu tài nguyên cho phép; đo p95 thực tế. Ngân sách dự kiến
   `S=1.25×p95`; slot thứ j dự kiến `T0+j×S`, chưa xong thì lượt sau phải chờ.
4. Lịch không đảm bảo thời điểm hoàn tất; timeout ghi lỗi, không bịa đáp án.
   IDs, timestamp, hash và chứng cứ đính kèm ảnh do runner ghi bằng đồng hồ thật.
5. Mỗi worker lưu file riêng/checkpoint; khóa tác vụ chỉ commit một lần. Chỉ giao
   lại sau khi quyền xử lý cũ hết hạn/dừng, không để hai worker ghi đè.
6. Quan sát mù nhãn cần một lượt ảnh không gold riêng; “đọc ảnh trước” trong prompt
   có nhãn không tạo độc lập. Duyệt ảnh-nhãn, Reasoning và các tranh chấp chuyên môn.
7. Báo độ phủ theo 20 tổ hợp và sáu thuộc tính, tỷ lệ skipped/needs_review, lỗi nhiễu,
   rò đáp án và nhất quán cùng ảnh. Không chỉ đánh giá những câu dễ được giữ.

Chưa tạo automation hoặc chạy subagent xử lý ảnh. Quy tắc Quantity, Size chồng
lấn và cây vị trí có ô nguồn nhập nhằng phải được chốt trước khi tăng quy mô.

## Giới hạn xác nhận

Thiết kế có thể triển khai từng bước và kiểm tra bằng phần mềm. Kiểm thử code
không chứng minh bộ câu hỏi mới cải thiện điểm hoặc ảnh đã được đọc chính xác.
Chỉ công bố hiệu quả sau pilot với ảnh thật và duyệt phù hợp. Trong bài báo,
ghi chính xác profile/prompt version đã chạy, đầu vào người trả lời được xem,
nguồn đáp án, quy trình duyệt và tỷ lệ bỏ qua.
