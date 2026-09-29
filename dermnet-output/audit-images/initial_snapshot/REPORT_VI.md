# Kiểm tra dermnet-output/images

Ngày kiểm tra: 28/09/2026. Không di chuyển, đổi tên hoặc xoá ảnh/thư mục gốc. Báo cáo phục vụ chuẩn hoá bộ dữ liệu, không xác nhận chẩn đoán từ ảnh.

## 1. Kết quả toàn bộ dữ liệu

| Chỉ số | Kết quả |
|---|---|
| Thư mục nhãn | 338 |
| Ảnh được đọc/giải mã | 7694 |
| Ảnh lỗi đọc / thư mục không có ảnh | 0 / 0 |
| Tệp ngoài ảnh | 88 tệp .DS_Store |
| Nhóm trùng byte SHA-256 | 347 |
| Bản dư theo byte | 358 |
| Nhóm trùng pixel | 366 |
| Bản dư theo pixel trên toàn bộ tập | 380 |
| Ảnh pixel duy nhất | 7314 |
| Bản dư trong cùng thư mục | 146 |
| Nhóm pixel xuất hiện ở nhiều thư mục | 228 |
| Cặp thư mục có chung ảnh pixel | 103 |
| Cặp nghi gần trùng, cần duyệt | 175 |

358 bản dư theo byte nằm trong 380 bản dư theo pixel: không cộng hai số này. 380 = 146 bản dư nội bộ + 234 bản dư còn lại giữa nhãn sau khi khử trùng nội bộ. Nếu chỉ khử trùng nội bộ, còn 7.548 tệp; nếu mỗi nội dung pixel chỉ giữ một bản toàn bộ tập, còn 7.314 ảnh. Đây là phép đếm, chưa phải khuyến nghị xoá 380 ảnh ngay.

## 2. Cách kiểm tra và giới hạn

- Đọc đệ quy toàn bộ images; lập danh mục đường dẫn, kích thước, định dạng, SHA-256. Không dùng tên file để kết luận trùng.
- Trùng byte: toàn bộ nội dung tệp giống nhau. Trùng pixel: cùng kích thước và cùng RGBA sau khi áp dụng EXIF orientation, bất kể metadata/container. Ảnh động được loại khỏi so sánh pixel/pHash; bộ hiện tại không có ảnh động.
- Gần trùng: pHash 63 bit từ DCT ảnh xám 32×32, khoảng cách Hamming ≤6; tìm toàn bộ tập bằng BK-tree, mỗi nhóm pixel chỉ lấy một đại diện. 175 là số cặp ứng viên, không phải 175 ảnh có thể xoá và không cộng vào 380.
- Đã xem trực quan 32 cặp mẫu: 20 cặp gần trùng và 12 cặp trùng pixel đáng ngờ ở VIN; cả mẫu đều cho thấy cùng nội dung ảnh chụp. 155 cặp gần trùng còn lại chưa được duyệt trực quan. Danh sách mẫu nằm trong visual_review_sample.json.
- pHash có thể bỏ sót crop lớn, xoay/lật, watermark lớn hoặc góc chụp khác; không phát hiện đầy đủ cùng bệnh nhân. Không bảo đảm mọi trùng ảnh biến đổi đã được tìm thấy.
- Pixel hash không chuẩn hoá ICC profile; không khẳng định hiển thị giống hệt trong mọi phần mềm quản lý màu. Phát hiện trùng ảnh không xác định nhãn nào đúng.
- Sàng lọc toàn bộ 338 tên thư mục; đối chiếu bài nguồn cục bộ cho các trường hợp được gắn cờ và DermNet gốc cho các quan hệ chính. Không phải thẩm định lâm sàng từng ảnh của cả 338 lớp. Bản dịch nguồn có lỗi, ví dụ một số chỗ dịch Hydroa vacciniforme thành thủy đậu.

## 3. Gộp tên đồng nghĩa và vấn đề cần xử lý trước

| Thư mục | Số ảnh | Đề xuất | Lý do / bằng chứng |
|---|---|---|---|
| Papular mucinosis | 2 | merge_synonym → Lichen myxoedematosus | Tên đồng nghĩa; giữ nguồn và phân nhóm của từng ảnh. 2 ảnh ở nhãn nguồn không trùng pixel với 6 ảnh ở nhãn đích. [DermNet](https://dermnetnz.org/topics/lichen-myxoedematosus) |
| Pyoderma | 76 | quarantine_label_review → Cutaneous abscess (chỉ ảnh xác minh đúng áp xe) | Hai tệp contents giống hệt, nhưng Pyoderma có 76 ảnh, Cutaneous abscess có 6 và không trùng pixel giữa hai thư mục. Không thể suy ra cả 76 ảnh đều là áp xe. Bỏ nhãn Pyoderma khỏi tập đơn nhãn tạm thời; kiểm tra từng ảnh trước khi gộp. [DermNet](https://dermnetnz.org/topics/cutaneous-abscess) |
| Vulval intraepithelial neoplasia | 62 | quarantine_image_review → rà soát | 14 ảnh pixel duy nhất trùng với 9 nhãn khác; đã thấy ảnh bàn tay, cằm/râu và tổn thương ngoài vùng âm hộ trong mẫu. Ưu tiên rà soát ảnh số thứ tự; giữ riêng ảnh VIN đã xác minh. Không xoá bệnh VIN. [DermNet](https://dermnetnz.org/topics/vulval-intraepithelial-neoplasia) |
| Non-albicans candida infections | 71 | quarantine_label_review → Candidiasis + vị trí + bằng chứng loài | 36/69 ảnh pixel duy nhất trùng với 6 nhãn khác. Toàn bộ 15 ảnh của Vulvovaginal candidiasis nằm trong nhóm này. Hình lâm sàng không chứng minh được loài non-albicans; chỉ giữ nhãn loài khi có nguồn xét nghiệm. Không chuyển toàn bộ onychomycosis thành Candida. [DermNet](https://dermnetnz.org/topics/non-albicans-candida-infections) |
| Atypical solar lentigo | 3 | quarantine_label_review → rà soát | Thuật ngữ dùng cho tổn thương không chắc là solar lentigo lành tính hay melanoma in situ; không gộp mù vào nhóm lành tính. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Atypical solar lentigo.txt>) |

Chỉ cặp Papular mucinosis → Lichen myxoedematosus đủ rõ để đề xuất hợp nhất tên lớp ngay về mặt taxonomy. Pyoderma có bài nguồn trùng nhưng tập ảnh khác nên chưa được coi là thư mục thừa có thể xoá. Ảnh số thứ tự không có thông tin nguồn đủ để tự gán lại nhãn.

## 4. Các nhóm cha–con có thể gộp tùy cấp phân loại

Khuyến nghị lưu `canonical_disease`, `subtype`, `site`, `stage`, `source_label`, `source_url`, `image_group_id`. Với bộ đơn nhãn, chọn cùng cấp độ phân loại; tránh cho nhãn cha và nhãn con cạnh tranh nhau. Gộp là giữ ảnh không trùng và metadata, không xoá ảnh chỉ vì thuộc subclass.

| Thư mục | Quan hệ / đích | Đề xuất và nguồn |
|---|---|---|
| Facial psoriasis | Psoriasis | Nhãn vị trí/biểu hiện của psoriasis. Nếu phân loại cấp bệnh, gộp và lưu vị trí làm metadata. Nếu phân loại chi tiết, giữ nhãn con và rà soát ảnh trong nhãn cha. [DermNet](https://dermnetnz.org/topics/psoriasis) |
| Flexural psoriasis | Psoriasis | Nhãn vị trí/biểu hiện của psoriasis. Nếu phân loại cấp bệnh, gộp và lưu vị trí làm metadata. Nếu phân loại chi tiết, giữ nhãn con và rà soát ảnh trong nhãn cha. [DermNet](https://dermnetnz.org/topics/psoriasis) |
| Genital psoriasis | Psoriasis | Nhãn vị trí/biểu hiện của psoriasis. Nếu phân loại cấp bệnh, gộp và lưu vị trí làm metadata. Nếu phân loại chi tiết, giữ nhãn con và rà soát ảnh trong nhãn cha. [DermNet](https://dermnetnz.org/topics/psoriasis) |
| Erythrodermic psoriasis | Psoriasis | Cùng nhóm nhưng thể bệnh có ý nghĩa lâm sàng; ưu tiên cấu trúc cha–con, chỉ gộp khi bài toán dùng cấp bệnh rộng. [DermNet](https://dermnetnz.org/topics/psoriasis) |
| Generalised pustular psoriasis | Psoriasis | Cùng nhóm nhưng thể bệnh có ý nghĩa lâm sàng; ưu tiên cấu trúc cha–con, chỉ gộp khi bài toán dùng cấp bệnh rộng. [DermNet](https://dermnetnz.org/topics/psoriasis) |
| Discoid lupus erythematosus | Cutaneous lupus erythematosus | DLE là thể lupus da; không đồng nhất với systemic lupus hoặc neonatal lupus. [DermNet](https://dermnetnz.org/topics/cutaneous-lupus-erythematosus) |
| Macular amyloidosis | Cutaneous amyloidosis | Các thể của amyloidosis da; lưu subtype. [DermNet](https://dermnetnz.org/topics/cutaneous-amyloidosis) |
| Amyloidosis cutis dyschromica | Cutaneous amyloidosis | Các thể của amyloidosis da; lưu subtype. [DermNet](https://dermnetnz.org/topics/cutaneous-amyloidosis) |
| Erythema dyschromicum perstans | Acquired dermal macular hyperpigmentation | Nhãn con trong nhóm tăng sắc tố dát trung bì mắc phải; không tự gộp melasma/postinflammatory hyperpigmentation. [DermNet](https://dermnetnz.org/topics/erythema-dyschromicum-perstans) |
| Cold urticaria | Urticaria | Các dạng mày đay cảm ứng; lưu yếu tố khởi phát. [DermNet](https://dermnetnz.org/topics/urticaria-an-overview) |
| Delayed pressure urticaria | Urticaria | Các dạng mày đay cảm ứng; lưu yếu tố khởi phát. [DermNet](https://dermnetnz.org/topics/urticaria-an-overview) |
| Solar urticaria | Urticaria | Các dạng mày đay cảm ứng; lưu yếu tố khởi phát. [DermNet](https://dermnetnz.org/topics/urticaria-an-overview) |
| Rhinophyma | Rosacea | Biểu hiện phymatous ở mũi thuộc phổ rosacea; lưu phenotype. [DermNet](https://dermnetnz.org/topics/rosacea) |
| Hyperkeratotic palmar dermatitis | Hand dermatitis | Thể chàm bàn tay tăng sừng. Không đồng nghĩa mọi hand dermatitis đều là atopic dermatitis. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Hyperkeratotic palmar dermatitis.txt>) |
| Acute localised exanthematous pustulosis | AGEP/ALEP spectrum | Nguồn mô tả ALEP là thể khu trú của AGEP. Nếu gộp nên dùng nhãn phổ chung, tránh gọi ảnh khu trú là generalised. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Acute localised exanthematous pustulosis.txt>) |
| Acral lentiginous melanoma | Melanoma | Các trục subtype, sắc tố và giai đoạn chồng lấp nhau. Có thể gộp ở cấp melanoma nhưng phải giữ stage/phenotype; không dùng chúng làm lớp đơn nhãn loại trừ nhau mà chưa rà soát. [DermNet](https://dermnetnz.org/topics/melanoma) |
| Amelanotic melanoma | Melanoma | Các trục subtype, sắc tố và giai đoạn chồng lấp nhau. Có thể gộp ở cấp melanoma nhưng phải giữ stage/phenotype; không dùng chúng làm lớp đơn nhãn loại trừ nhau mà chưa rà soát. [DermNet](https://dermnetnz.org/topics/melanoma) |
| Nodular melanoma | Melanoma | Các trục subtype, sắc tố và giai đoạn chồng lấp nhau. Có thể gộp ở cấp melanoma nhưng phải giữ stage/phenotype; không dùng chúng làm lớp đơn nhãn loại trừ nhau mà chưa rà soát. [DermNet](https://dermnetnz.org/topics/melanoma) |
| Melanoma in situ | Melanoma | Các trục subtype, sắc tố và giai đoạn chồng lấp nhau. Có thể gộp ở cấp melanoma nhưng phải giữ stage/phenotype; không dùng chúng làm lớp đơn nhãn loại trừ nhau mà chưa rà soát. [DermNet](https://dermnetnz.org/topics/melanoma) |
| Metastatic melanoma | Melanoma | Các trục subtype, sắc tố và giai đoạn chồng lấp nhau. Có thể gộp ở cấp melanoma nhưng phải giữ stage/phenotype; không dùng chúng làm lớp đơn nhãn loại trừ nhau mà chưa rà soát. [DermNet](https://dermnetnz.org/topics/melanoma) |
| Atypical melanocytic naevus | Melanocytic naevus | Có thể tổ chức phân cấp nhóm tổn thương tế bào hắc tố, nhưng đặc điểm và vị trí khác nhau. Ưu tiên giữ chi tiết; Ota/Ito/Hori còn là nhãn chứa nhiều thực thể. [DermNet](https://dermnetnz.org/topics/melanocytic-naevus) |
| Blue nevus | Melanocytic naevus | Có thể tổ chức phân cấp nhóm tổn thương tế bào hắc tố, nhưng đặc điểm và vị trí khác nhau. Ưu tiên giữ chi tiết; Ota/Ito/Hori còn là nhãn chứa nhiều thực thể. [DermNet](https://dermnetnz.org/topics/melanocytic-naevus) |
| Halo naevus | Melanocytic naevus | Có thể tổ chức phân cấp nhóm tổn thương tế bào hắc tố, nhưng đặc điểm và vị trí khác nhau. Ưu tiên giữ chi tiết; Ota/Ito/Hori còn là nhãn chứa nhiều thực thể. [DermNet](https://dermnetnz.org/topics/melanocytic-naevus) |
| Meyerson naevus | Melanocytic naevus | Có thể tổ chức phân cấp nhóm tổn thương tế bào hắc tố, nhưng đặc điểm và vị trí khác nhau. Ưu tiên giữ chi tiết; Ota/Ito/Hori còn là nhãn chứa nhiều thực thể. [DermNet](https://dermnetnz.org/topics/melanocytic-naevus) |
| Naevus of Ota, naevus of Ito and naevus of Hori | Melanocytic naevus | Có thể tổ chức phân cấp nhóm tổn thương tế bào hắc tố, nhưng đặc điểm và vị trí khác nhau. Ưu tiên giữ chi tiết; Ota/Ito/Hori còn là nhãn chứa nhiều thực thể. [DermNet](https://dermnetnz.org/topics/melanocytic-naevus) |
| Anogenital squamous cell carcinoma | Squamous cell carcinoma | Phân theo vị trí; giữ miền ảnh niêm mạc/sinh dục, chỉ gộp nếu scope bao gồm các vị trí này. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Anogenital squamous cell carcinoma.txt>) |
| Oral squamous cell carcinoma | Squamous cell carcinoma | Phân theo vị trí; giữ miền ảnh niêm mạc/sinh dục, chỉ gộp nếu scope bao gồm các vị trí này. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Oral squamous cell carcinoma.txt>) |
| Marjolin ulcer | Squamous cell carcinoma (khi nguồn xác nhận) | Ung thư phát sinh trên sẹo/loét, thường là SCC. Cần lịch sử/nguồn chẩn đoán; không coi là loét chân thông thường. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Marjolin ulcer.txt>) |
| Flea bite | Arthropod bites and stings | Nhóm con theo tác nhân; lưu tác nhân nếu có nguồn xác minh. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Flea bite.txt>) |
| Exercise-induced vasculitis | Cutaneous vasculitis (cấp biểu hiện da) | Có liên hệ nhóm viêm mạch nhưng khác căn nguyên/bối cảnh; giữ nhãn cụ thể. HSP là IgA vasculitis có thể có tổn thương hệ thống; không gộp vào Purpura như một bệnh đồng nghĩa. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Exercise-induced vasculitis.txt>) |
| Henoch–Schönlein purpura | Cutaneous vasculitis (cấp biểu hiện da) | Có liên hệ nhóm viêm mạch nhưng khác căn nguyên/bối cảnh; giữ nhãn cụ thể. HSP là IgA vasculitis có thể có tổn thương hệ thống; không gộp vào Purpura như một bệnh đồng nghĩa. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Henoch–Schönlein purpura.txt>) |
| Urticarial vasculitis | Cutaneous vasculitis (cấp biểu hiện da) | Có liên hệ nhóm viêm mạch nhưng khác căn nguyên/bối cảnh; giữ nhãn cụ thể. HSP là IgA vasculitis có thể có tổn thương hệ thống; không gộp vào Purpura như một bệnh đồng nghĩa. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Urticarial vasculitis.txt>) |
| Erythema elevatum diutinum | Cutaneous vasculitis (cấp biểu hiện da) | Có liên hệ nhóm viêm mạch nhưng khác căn nguyên/bối cảnh; giữ nhãn cụ thể. HSP là IgA vasculitis có thể có tổn thương hệ thống; không gộp vào Purpura như một bệnh đồng nghĩa. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Erythema elevatum diutinum.txt>) |
| Boil | Bacterial folliculitis / Cutaneous abscess | Viêm nang lông sâu và có thể tạo áp xe; quan hệ chồng lấp, không phải ba tên đồng nghĩa. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Boil.txt>) |
| Kerion | Tinea capitis / Tinea faciei / Tinea corporis | Biểu hiện viêm mạnh do dermatophyte; chỉ chuyển vào Tinea capitis nếu xác nhận ở da đầu. [DermNet](https://dermnetnz.org/topics/kerion) |
| Tinea barbae | Dermatophytosis (cha mới nếu cần) | Giữ site/depth làm metadata. Đích chung chưa có trong images; không lấy Tinea corporis làm cha cho tất cả. Majocchi là nhiễm sâu. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Tinea barbae.txt>) |
| Tinea capitis | Dermatophytosis (cha mới nếu cần) | Giữ site/depth làm metadata. Đích chung chưa có trong images; không lấy Tinea corporis làm cha cho tất cả. Majocchi là nhiễm sâu. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Tinea capitis.txt>) |
| Tinea corporis | Dermatophytosis (cha mới nếu cần) | Giữ site/depth làm metadata. Đích chung chưa có trong images; không lấy Tinea corporis làm cha cho tất cả. Majocchi là nhiễm sâu. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Tinea corporis.txt>) |
| Tinea cruris | Dermatophytosis (cha mới nếu cần) | Giữ site/depth làm metadata. Đích chung chưa có trong images; không lấy Tinea corporis làm cha cho tất cả. Majocchi là nhiễm sâu. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Tinea cruris.txt>) |
| Tinea faciei | Dermatophytosis (cha mới nếu cần) | Giữ site/depth làm metadata. Đích chung chưa có trong images; không lấy Tinea corporis làm cha cho tất cả. Majocchi là nhiễm sâu. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Tinea faciei.txt>) |
| Tinea manuum | Dermatophytosis (cha mới nếu cần) | Giữ site/depth làm metadata. Đích chung chưa có trong images; không lấy Tinea corporis làm cha cho tất cả. Majocchi là nhiễm sâu. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Tinea manuum.txt>) |
| Tinea pedis | Dermatophytosis (cha mới nếu cần) | Giữ site/depth làm metadata. Đích chung chưa có trong images; không lấy Tinea corporis làm cha cho tất cả. Majocchi là nhiễm sâu. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Tinea pedis.txt>) |
| Majocchi granuloma | Dermatophytosis (cha mới nếu cần) | Giữ site/depth làm metadata. Đích chung chưa có trong images; không lấy Tinea corporis làm cha cho tất cả. Majocchi là nhiễm sâu. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Majocchi granuloma.txt>) |

## 5. Thuật ngữ, dấu hiệu và nhóm quá rộng

Nếu mục tiêu là nhận diện một bệnh cụ thể từ một ảnh, đề xuất tách các nhãn sau khỏi tập chẩn đoán đơn nhãn sang nhóm dấu hiệu/nhãn tổng quát hoặc rà từng ảnh để gán bệnh nền. Không đề xuất xoá vĩnh viễn thư mục: các nhãn này vẫn hữu ích cho bài toán nhận diện hình thái hoặc đa nhãn.

| Thư mục | Lý do / hành động |
|---|---|
| Pruritus | Triệu chứng ngứa, nhiều nguyên nhân; ảnh trùng với Notalgia paraesthetica và Prurigo không làm ba nhãn đồng nghĩa. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Pruritus.txt>) |
| Purpura | Dấu hiệu xuất huyết da, chứa nhiều căn nguyên; 27 ảnh chung với Pigmented purpura và toàn bộ 6 ảnh HSP. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Purpura.txt>) |
| Exanthems | Thuật ngữ phát ban lan toả, không chỉ một bệnh hay một tác nhân. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Exanthems.txt>) |
| Cutaneous horn | Hình thái sừng da, có nhiều tổn thương nền; không tự gộp vào Actinic keratosis hoặc SCC. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Cutaneous horn.txt>) |
| Pityriasis amiantacea | Kiểu phản ứng da đầu, có nhiều bệnh nền; không tự gộp vào psoriasis hoặc tinea. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Pityriasis amiantacea.txt>) |
| Elastosis | Thuật ngữ thay đổi mô đàn hồi bao gồm nhiều tình trạng, không đồng nghĩa Elastosis perforans serpiginosa. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Elastosis.txt>) |
| Leg ulcer | Loét chân là biểu hiện với nhiều nguyên nhân. Tách theo căn nguyên nếu nguồn cho phép. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Leg ulcer.txt>) |
| Madarosis | Rụng lông mi/lông mày do nhiều bệnh; không tự gộp vào Alopecia areata. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Madarosis.txt>) |
| Poliosis | Lông/tóc trắng khu trú, nhiều nguyên nhân; không tự gộp vào Vitiligo. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Poliosis.txt>) |
| Melanonychia | Sắc tố móng, có nhiều nguyên nhân lành/ác tính; không tự chuyển thành melanoma. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Melanonychia.txt>) |
| Macroglossia | Lưỡi to là dấu hiệu với nhiều căn nguyên. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Macroglossia.txt>) |
| Dry gangrene | Biểu hiện hoại tử khô, nhiều nguyên nhân mạch máu/hệ thống. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Dry gangrene.txt>) |
| Amputation stump dermatoses | Nhóm nhiều bệnh ở mỏm cụt; cần nhãn bệnh cụ thể cho từng ảnh. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Amputation stump dermatoses.txt>) |
| Granulomatous dermatitis | Nhóm kiểu phản ứng/mô bệnh học; không tự gộp vào sarcoidosis hay granuloma annulare. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Granulomatous dermatitis.txt>) |
| Panniculitis | Nhóm viêm mô mỡ dưới da; có thể giữ làm cha, không coi mọi ảnh là erythema nodosum. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Panniculitis.txt>) |
| Palmoplantar keratoderma | Nhóm dày sừng lòng bàn tay/bàn chân, có dạng di truyền và mắc phải. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Palmoplantar keratoderma.txt>) |
| Ichthyosis | Nhóm bệnh nhiều thể; giữ làm cha hoặc gán subtype từ nguồn. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Ichthyosis.txt>) |
| Oral mucositis - Stomatitis | Viêm niêm mạc miệng do nhiều nguyên nhân; không đồng nghĩa Aphthous ulcer. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Oral mucositis - Stomatitis.txt>) |
| Wound infection | Nhiễm trùng vết thương nhiều tác nhân; không đồng nhất với Pyoderma hoặc áp xe. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Wound infection.txt>) |
| Malignant histiocytoses | Nhãn nhóm rộng/thuật ngữ lịch sử; cần rà từng thực thể, không gộp toàn bộ vào Leukaemia cutis. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Malignant histiocytoses.txt>) |

## 6. Các tên gần nhau nhưng không nên gộp tự động

| Thư mục | Lưu ý |
|---|---|
| Ocular melanoma | Giữ riêng modality/vị trí mắt; không tự trộn vào bài toán ảnh da ngoài. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Ocular melanoma.txt>) |
| Papular urticaria | Phản ứng dạng sẩn do côn trùng; không gộp vào Urticaria chỉ vì tên có urticaria. [DermNet](https://dermnetnz.org/topics/papular-urticaria) |
| Atopic dermatitis | Các nhãn theo căn nguyên, vị trí, cơ chế khác nhau; có thể chồng lấp. Không gộp tất cả vào Atopic dermatitis. Với bài toán rộng tạo nhóm eczema mới và lưu các trục nhãn. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Atopic dermatitis.txt>) |
| Asteatotic eczema | Các nhãn theo căn nguyên, vị trí, cơ chế khác nhau; có thể chồng lấp. Không gộp tất cả vào Atopic dermatitis. Với bài toán rộng tạo nhóm eczema mới và lưu các trục nhãn. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Asteatotic eczema.txt>) |
| Nummular eczematous dermatitis | Các nhãn theo căn nguyên, vị trí, cơ chế khác nhau; có thể chồng lấp. Không gộp tất cả vào Atopic dermatitis. Với bài toán rộng tạo nhóm eczema mới và lưu các trục nhãn. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Nummular eczematous dermatitis.txt>) |
| Dyshidrotic eczema (pompholyx) | Các nhãn theo căn nguyên, vị trí, cơ chế khác nhau; có thể chồng lấp. Không gộp tất cả vào Atopic dermatitis. Với bài toán rộng tạo nhóm eczema mới và lưu các trục nhãn. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Dyshidrotic eczema (pompholyx).txt>) |
| Venous eczema | Các nhãn theo căn nguyên, vị trí, cơ chế khác nhau; có thể chồng lấp. Không gộp tất cả vào Atopic dermatitis. Với bài toán rộng tạo nhóm eczema mới và lưu các trục nhãn. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Venous eczema.txt>) |
| Disseminated secondary eczema | Các nhãn theo căn nguyên, vị trí, cơ chế khác nhau; có thể chồng lấp. Không gộp tất cả vào Atopic dermatitis. Với bài toán rộng tạo nhóm eczema mới và lưu các trục nhãn. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Disseminated secondary eczema.txt>) |
| Napkin dermatitis | Các nhãn theo căn nguyên, vị trí, cơ chế khác nhau; có thể chồng lấp. Không gộp tất cả vào Atopic dermatitis. Với bài toán rộng tạo nhóm eczema mới và lưu các trục nhãn. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Napkin dermatitis.txt>) |
| Nipple eczema | Các nhãn theo căn nguyên, vị trí, cơ chế khác nhau; có thể chồng lấp. Không gộp tất cả vào Atopic dermatitis. Với bài toán rộng tạo nhóm eczema mới và lưu các trục nhãn. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Nipple eczema.txt>) |
| Airborne contact dermatitis | Các nhãn theo căn nguyên, vị trí, cơ chế khác nhau; có thể chồng lấp. Không gộp tất cả vào Atopic dermatitis. Với bài toán rộng tạo nhóm eczema mới và lưu các trục nhãn. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Airborne contact dermatitis.txt>) |
| Eyelid contact dermatitis | Các nhãn theo căn nguyên, vị trí, cơ chế khác nhau; có thể chồng lấp. Không gộp tất cả vào Atopic dermatitis. Với bài toán rộng tạo nhóm eczema mới và lưu các trục nhãn. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Eyelid contact dermatitis.txt>) |
| Nickel allergy | Các nhãn theo căn nguyên, vị trí, cơ chế khác nhau; có thể chồng lấp. Không gộp tất cả vào Atopic dermatitis. Với bài toán rộng tạo nhóm eczema mới và lưu các trục nhãn. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Nickel allergy.txt>) |
| Palmoplantar pustulosis | Giữ subtype; palmoplantar pustulosis có thể là thực thể riêng dù liên hệ psoriasis. Không tự gộp vào Generalised pustular psoriasis. [DermNet](https://dermnetnz.org/topics/palmoplantar-pustulosis) |
| Acrodermatitis continua of Hallopeau | Giữ subtype; palmoplantar pustulosis có thể là thực thể riêng dù liên hệ psoriasis. Không tự gộp vào Generalised pustular psoriasis. [DermNet](https://dermnetnz.org/topics/palmoplantar-pustulosis) |
| Psoriatic arthritis | Giữ chẩn đoán riêng; có quan hệ hoặc biểu hiện giống nhãn khác nhưng không đủ cơ sở coi là đồng nghĩa hay xoá lớp. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Psoriatic arthritis.txt>) |
| Systemic lupus erythematosus | Giữ chẩn đoán riêng; có quan hệ hoặc biểu hiện giống nhãn khác nhưng không đủ cơ sở coi là đồng nghĩa hay xoá lớp. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Systemic lupus erythematosus.txt>) |
| Neonatal lupus | Giữ chẩn đoán riêng; có quan hệ hoặc biểu hiện giống nhãn khác nhưng không đủ cơ sở coi là đồng nghĩa hay xoá lớp. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Neonatal lupus.txt>) |
| Epidermolysis bullosa acquisita | Giữ chẩn đoán riêng; có quan hệ hoặc biểu hiện giống nhãn khác nhưng không đủ cơ sở coi là đồng nghĩa hay xoá lớp. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Epidermolysis bullosa acquisita.txt>) |
| Pemphigoid gestationis | Giữ chẩn đoán riêng; có quan hệ hoặc biểu hiện giống nhãn khác nhưng không đủ cơ sở coi là đồng nghĩa hay xoá lớp. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Pemphigoid gestationis.txt>) |
| Pemphigus foliaceus | Giữ chẩn đoán riêng; có quan hệ hoặc biểu hiện giống nhãn khác nhưng không đủ cơ sở coi là đồng nghĩa hay xoá lớp. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Pemphigus foliaceus.txt>) |
| Pemphigus vulgaris | Giữ chẩn đoán riêng; có quan hệ hoặc biểu hiện giống nhãn khác nhưng không đủ cơ sở coi là đồng nghĩa hay xoá lớp. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Pemphigus vulgaris.txt>) |
| Sézary syndrome | Giữ chẩn đoán riêng; có quan hệ hoặc biểu hiện giống nhãn khác nhưng không đủ cơ sở coi là đồng nghĩa hay xoá lớp. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Sézary syndrome.txt>) |
| Pseudoporphyria | Giữ chẩn đoán riêng; có quan hệ hoặc biểu hiện giống nhãn khác nhưng không đủ cơ sở coi là đồng nghĩa hay xoá lớp. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Pseudoporphyria.txt>) |
| Acquired lymphangiectasia | Giữ chẩn đoán riêng; có quan hệ hoặc biểu hiện giống nhãn khác nhưng không đủ cơ sở coi là đồng nghĩa hay xoá lớp. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Acquired lymphangiectasia.txt>) |
| Lymphatic malformation | Giữ chẩn đoán riêng; có quan hệ hoặc biểu hiện giống nhãn khác nhưng không đủ cơ sở coi là đồng nghĩa hay xoá lớp. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Lymphatic malformation.txt>) |
| Erysipelas | Giữ chẩn đoán riêng; có quan hệ hoặc biểu hiện giống nhãn khác nhưng không đủ cơ sở coi là đồng nghĩa hay xoá lớp. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Erysipelas.txt>) |
| Cellulitis | Giữ chẩn đoán riêng; có quan hệ hoặc biểu hiện giống nhãn khác nhưng không đủ cơ sở coi là đồng nghĩa hay xoá lớp. [Bài nguồn lưu cục bộ](</Users/binhminh/Desktop/DermNet_Dataset/dermnet-output/contents/Toàn bộ nội dung - Cellulitis.txt>) |

Đặc biệt: Epidermolysis bullosa acquisita không phải chỉ là tên khác của nhóm EB di truyền; Hailey-Hailey không phải pemphigus tự miễn; Sézary syndrome và Mycosis fungoides cần giữ riêng; Pseudoporphyria không đồng nghĩa Porphyria cutanea tarda. Giữ VIN/penile intraepithelial neoplasia riêng với SCC xâm lấn. Trùng ảnh có thể do đồng bệnh, chẩn đoán phân biệt hoặc lỗi thu thập.

## 7. Chuẩn hoá tên, không xoá lớp

| Tên hiện tại | Tên đề xuất | Lý do |
|---|---|---|
| Benign familial pemphigus | Hailey-Hailey disease | Tên đồng nghĩa, không gộp vào pemphigus tự miễn. |
| Bazin_s hydroa vacciniforme | Hydroa vacciniforme | Chuẩn hoá tên theo bài nguồn. |
| Blue nevus | Blue naevus | Thống nhất chính tả naevus/nevus; không bắt buộc. |
| Nevus comedonicus | Naevus comedonicus | Thống nhất chính tả; không phải melanocytic naevus. |
| Milker_s nodule | Milker’s nodule | Sửa dấu nháy bị thay bằng dấu gạch dưới. |
| Swimmer_s itch | Swimmer’s itch | Sửa dấu nháy bị thay bằng dấu gạch dưới. |
| Chloracne _ MADISH | Chloracne (MADISH) | Chuẩn hoá dấu phân cách. |
| Plasma cell balanitis_vulvitis | Plasma cell balanitis / vulvitis | Chuẩn hoá nhãn; giữ thông tin vị trí. |

## 8. Các cặp thư mục trùng ảnh nhiều nhất

Số chung tính theo ảnh pixel duy nhất, tỷ lệ có mẫu số là số ảnh pixel duy nhất trong từng thư mục. Trùng toàn bộ tập ảnh không đồng nghĩa hai bệnh đồng nghĩa.

| Thư mục A | Thư mục B | Ảnh chung | % A | % B |
|---|---|---|---|---|
| Pigmented purpura | Purpura | 27 | 52.9 | 24.1 |
| Non-albicans candida infections | Vulvovaginal candidiasis | 15 | 21.7 | 100.0 |
| Notalgia paraesthetica | Pruritus | 14 | 66.7 | 25.4 |
| Non-albicans candida infections | Onychomycosis | 11 | 15.9 | 17.7 |
| Non-albicans candida infections | Oral candidiasis | 7 | 10.1 | 38.9 |
| Henoch–Schönlein purpura | Purpura | 6 | 100.0 | 5.4 |
| Polymorphic light eruption | Prurigo | 6 | 21.4 | 6.1 |
| Alopecia areata | Madarosis | 5 | 10.0 | 29.4 |
| Atypical melanocytic naevus | Melanocytic naevus | 4 | 20.0 | 3.6 |
| Meningococcal disease | Purpura | 4 | 57.1 | 3.6 |
| Anagen effluvium | Kerion | 3 | 17.6 | 15.8 |
| Angular cheilitis | Non-albicans candida infections | 3 | 8.1 | 4.3 |
| Atopic dermatitis | Prurigo | 3 | 1.8 | 3.0 |
| Atopic dermatitis | Hand dermatitis | 3 | 1.8 | 2.3 |
| Bullous pemphigoid | Vulval intraepithelial neoplasia | 3 | 4.5 | 5.4 |
| Calciphylaxis | Purpura | 3 | 20.0 | 2.7 |
| Cutaneous amyloidosis | Macular amyloidosis | 3 | 8.1 | 37.5 |
| Cutaneous lupus erythematosus | Discoid lupus erythematosus | 3 | 3.0 | 4.0 |
| Elastosis | Elastosis perforans serpiginosa | 3 | 14.3 | 21.4 |
| Halo naevus | Melanocytic naevus | 3 | 11.5 | 2.7 |
| Lichen planus | Postinflammatory hyperpigmentation | 3 | 1.8 | 11.5 |
| Lichen planus | Vulval intraepithelial neoplasia | 3 | 1.8 | 5.4 |
| Postinflammatory hyperpigmentation | Vulval intraepithelial neoplasia | 3 | 11.5 | 5.4 |
| Acral lentiginous melanoma | Melanoma in situ | 2 | 5.0 | 3.6 |
| Alopecia areata | Poliosis | 2 | 4.0 | 15.4 |

## 9. Các thư mục có nhiều bản dư nội bộ

| Thư mục | Tệp ảnh | Ảnh pixel duy nhất | Bản dư |
|---|---|---|---|
| Pityriasis rubra pilaris | 43 | 30 | 13 |
| Pearly penile papules | 16 | 8 | 8 |
| Pemphigus foliaceus | 19 | 13 | 6 |
| Piebaldism | 13 | 7 | 6 |
| Pilomatricoma | 12 | 6 | 6 |
| Poikiloderma of Civatte | 12 | 6 | 6 |
| Vulval intraepithelial neoplasia | 62 | 56 | 6 |
| Neonatal lupus | 10 | 6 | 4 |
| Nickel allergy | 25 | 21 | 4 |
| Nummular eczematous dermatitis | 144 | 140 | 4 |
| Onychomycosis | 66 | 62 | 4 |
| Poliosis | 17 | 13 | 4 |
| Polymorphic eruption of pregnancy | 10 | 6 | 4 |
| Porphyria cutanea tarda | 16 | 12 | 4 |
| Cutaneous lupus erythematosus | 102 | 99 | 3 |
| Nipple eczema | 15 | 12 | 3 |
| Nodular chondrodermatitis | 11 | 8 | 3 |
| Noonan syndrome with multiple lentigines | 6 | 3 | 3 |
| Notalgia paraesthetica | 24 | 21 | 3 |
| Onychocryptosis | 11 | 8 | 3 |

## 10. Đề xuất xử lý theo thứ tự

1. Cách ly khỏi tập huấn luyện các ảnh/nhãn chưa rõ ở VIN, Pyoderma và Non-albicans candida infections; xác minh nguồn. Không gộp các bệnh không liên quan vì ảnh trùng.
2. Chuẩn hoá Papular mucinosis thành Lichen myxoedematosus; quyết định dùng nhãn cấp bệnh hay nhãn phân cấp trước khi gộp các subclass.
3. Rà manifest 146 bản dư nội bộ; giữ một ảnh đại diện và tất cả metadata/nguồn. Manifest chỉ là đề xuất, không chứa lệnh xoá.
4. Với 228 nhóm trùng xuyên thư mục, giữ một image_id và tập nhãn đã xác minh; nếu cần đơn nhãn thì giải quyết xung đột trước. Không chọn nhãn theo thứ tự tên thư mục.
5. Duyệt 175 cặp gần trùng trong gallery; ưu tiên bản có độ phân giải tốt sau khi kiểm tra nội dung. Không khử trùng chỉ dựa trên pHash.
6. Chia train/validation/test theo nhóm ảnh/case/patient nếu có, để ảnh trùng và biến thể của nó không rơi vào các split khác nhau.

## 11. Tệp kết quả

- `gallery.html`: xem từng nhóm trùng và cặp nghi gần trùng, lọc theo tên thư mục/đường dẫn, mở ảnh gốc.
- `within_folder_dedup_proposal.json`: 146 bản dư nội bộ và ảnh đại diện đề xuất.
- `cross_folder_label_review.json`: 228 nhóm trùng xuyên nhãn.
- `duplicates_bytes.json`, `duplicates_pixels.json`: toàn bộ nhóm trùng chính xác.
- `near_duplicate_candidates.json`: 175 cặp nghi gần trùng; đại diện pixel-group, không mở rộng thành mọi tổ hợp file.
- `folder_overlap.json`: toàn bộ 103 cặp thư mục có chung ảnh.
- `taxonomy_recommendations.json`: khuyến nghị có điều kiện cho từng thư mục được gắn cờ.
- `all_folder_assessments.json`: đủ 338 thư mục, bao gồm những thư mục chưa thấy vấn đề taxonomy rõ từ sàng lọc tên.
- `local_source_evidence.json`: đường dẫn và đoạn định nghĩa nguồn cục bộ; `inventory.json`, `folders.json`, `summary.json`: danh mục và thống kê.
- `audit.py`, `build_report.py`: mã tái lập. Chạy bằng `dermnet-output/.venv/bin/python`; chỉ ghi trong audit-images.
