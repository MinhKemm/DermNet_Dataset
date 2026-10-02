# Xử lý ảnh trùng — 02/10/2026

Phạm vi: thư mục `dermnet-output/images` tại commit `2255b768`. Chỉ xử lý ảnh trùng byte hoặc pixel chính xác. Đã xem đại diện cả 41 nhóm xuyên thư mục trong các bảng ảnh review-1.jpg đến review-6.jpg, xác minh nguồn bằng ảnh gốc DermNet; không suy ra bệnh chỉ từ ảnh.

## Kết quả

| Chỉ số | Trước | Sau |
|---|---:|---:|
| Ảnh trong thư mục dữ liệu | 7.057 | 6.992 |
| Nhóm trùng byte | 53 | 0 |
| Nhóm trùng pixel | 64 | 0 |
| Ảnh riêng chờ xác minh nhãn | 0 | 1 |

- Bỏ 23 bản dư cùng thư mục.
- Xử lý 41 nhóm xuyên thư mục: 40 nhóm giữ một bản có bằng chứng nguồn; một nhóm giữ riêng một bản chờ xác minh nhãn.
- Tổng cộng bỏ 64 bản dư. 65 đường dẫn gốc không còn vì có thêm một ảnh đại diện được chuyển ra khu chờ duyệt.
- 6.992 ảnh đang dùng + 1 ảnh chờ duyệt = 6.993 nội dung pixel duy nhất; không mất nội dung pixel nào của bộ trước xử lý.
- SHA-256 của cả 6.992 tệp giữ trong bộ không thay đổi. Sao lưu đủ 65 đường dẫn gốc, đã kiểm tra SHA-256 và CRC archive.

## Tiêu chí giữ bản

Trong cùng thư mục: cùng kích thước và pixel RGBA sau EXIF orientation; ưu tiên tên có trong nguồn, tên mô tả, bản không có hậu tố v2, rồi thông tin tệp gốc. Không thay đổi nội dung ảnh.

Giữa thư mục: ưu tiên nhãn được chú thích/metadata nguồn xác nhận cho chính ảnh đó. Giữ liên kết tới các nhãn/đường dẫn cũ; nhãn bị bỏ ở một vị trí không tự động được coi là sai về mặt lâm sàng. Các quan hệ bệnh và biểu hiện được lưu trong nhật ký.

## Các quyết định xuyên thư mục

| Nhóm | Bản giữ / vị trí chờ duyệt | Bản bỏ khỏi thư mục ảnh | Nguồn đã khớp pixel |
|---|---|---|---|
| G001 | `Atopic dermatitis/atopic-dermatitis-0049.jpg` | `Prurigo/05.jpg` | [DermNet](https://dermnetnz.org/images/atopic-dermatitis-images) |
| G002 | `Atopic dermatitis/eapsimplex1.jpg` | `Prurigo/09.jpg` | [DermNet](https://dermnetnz.org/topics/atopic-dermatitis); [DermNet](https://dermnetnz.org/images/atopic-dermatitis-images) |
| G003 | `Systemic lupus erythematosus/sle-13.jpg` | `Cutaneous lupus erythematosus/lupus-sc4.jpg` | [DermNet](https://dermnetnz.org/topics/cutaneous-lupus-erythematosus); [DermNet](https://dermnetnz.org/topics/systemic-lupus-erythematosus) |
| G004 | `Epidermolysis bullosa acquisita/epidermolysis-bullosa-acquisita-0002.jpg` | `Postinflammatory hyperpigmentation/10.jpg` | [DermNet](https://dermnetnz.org/images/epidermolysis-bullosa-acquisita-images) |
| G005 | `Epidermolysis bullosa acquisita/epidermolysis-bullosa-acquisita-0003.jpg` | `Postinflammatory hyperpigmentation/11.jpg` | [DermNet](https://dermnetnz.org/images/epidermolysis-bullosa-acquisita-images) |
| G007 | `Mycoplasma pneumoniae infection/mycoplasma1.jpg` | `Erythema multiforme/mycoplasma1.jpg` | [DermNet](https://dermnetnz.org/topics/mycoplasma-pneumoniae-infection) |
| G008 | `Mycoplasma pneumoniae infection/mycoplasma2.jpg` | `Erythema multiforme/mycoplasma2.jpg` | [DermNet](https://dermnetnz.org/topics/mycoplasma-pneumoniae-infection) |
| G009 | `Herpes simplex/herpes-simplex-labialis-38.jpg` | `Oral candidiasis/05.jpg` | [DermNet](https://dermnetnz.org/images/herpes-simplex-images) |
| G010 | `Herpes simplex/herpes-simplex-labialis-39.jpg` | `Oral candidiasis/06.jpg` | [DermNet](https://dermnetnz.org/images/herpes-simplex-images) |
| G011 | `Leg ulcer/3042.jpg` | `Pyoderma/01.jpg` | [DermNet](https://dermnetnz.org/topics/leg-ulcer-images) |
| G012 | `Lichen planus/annular-lichen-planus-0007.jpg` | `Postinflammatory hyperpigmentation/05.jpg` | [DermNet](https://dermnetnz.org/images/annular-lichen-planus-images); [DermNet](https://dermnetnz.org/imagedetail/3699-annular-lichen-planus) |
| G013 | `Lichen planus/papular-lichen-planus-0009.jpg` | `Postinflammatory hyperpigmentation/15.jpg` | [DermNet](https://dermnetnz.org/images/lichen-planus-images); [DermNet](https://dermnetnz.org/imagedetail/1527-papular-lichen-planus) |
| G014 | `Lichen planus/papular-lichen-planus-0010.jpg` | `Postinflammatory hyperpigmentation/16.jpg` | [DermNet](https://dermnetnz.org/images/lichen-planus-images); [DermNet](https://dermnetnz.org/imagedetail/1525-papular-lichen-planus) |
| G015 | `Porphyria cutanea tarda/03.jpg` | `Milium/milia08.jpg` | [DermNet](https://dermnetnz.org/topics/milia-images); [DermNet](https://dermnetnz.org/imagedetail/8550-milia) |
| G016 | `Oral mucositis - Stomatitis/81.jpg` | `Mycoplasma pneumoniae infection/reactive-infectious-mucocutaneous-eruption-0008.jpg` | [DermNet](https://dermnetnz.org/topics/reactive-infectious-mucocutaneous-eruption); [DermNet](https://dermnetnz.org/images/reactive-infectious-mucocutaneous-eruption-images) |
| G021 | `Paronychia/3296.jpg` | `Onychocryptosis/01.jpg` | [DermNet](https://dermnetnz.org/topics/paronychia-images) |
| G022 | `Tinea manuum/05.jpg` | `Onychomycosis/47.jpg` | [DermNet](https://dermnetnz.org/imagedetail/16071-tinea-manuum) |
| G023 | `Oral candidiasis/03.jpg` | `Oral mucositis - Stomatitis/18.jpg` | [DermNet](https://dermnetnz.org/topics/stomatitis) |
| G024 | `Pityriasis rubra pilaris/11.jpg` | `Palmoplantar keratoderma/35.jpg` | [DermNet](https://dermnetnz.org/images/pityriasis-rubra-pilaris-images) |
| G025 | `Pityriasis rubra pilaris/12.jpg` | `Palmoplantar keratoderma/36.jpg` | [DermNet](https://dermnetnz.org/images/pityriasis-rubra-pilaris-images) |
| G026 | `Panniculitis/03.jpg` | `Venous eczema/acute-lipodermatosclerosis-21.jpg` | [DermNet](https://dermnetnz.org/topics/lipodermatosclerosis-images) |
| G027 | `Panniculitis/04.jpg` | `Venous eczema/acute-lipodermatosclerosis-22.jpg` | [DermNet](https://dermnetnz.org/topics/lipodermatosclerosis-images) |
| G028 | `Pediculosis capitis/12.jpg` | `Pyoderma/08.jpg` | [DermNet](https://dermnetnz.org/imagedetail/2623-pediculosis) |
| G029 | `Pemphigoid gestationis/37.jpg` | `Postinflammatory hyperpigmentation/24.jpg` | [DermNet](https://dermnetnz.org/imagedetail/3395-pemphigoid-gestationis-00029) |
| G030 | `Pemphigoid gestationis/38.jpg` | `Postinflammatory hyperpigmentation/25.jpg` | [DermNet](https://dermnetnz.org/imagedetail/3385-pemphigoid-gestationis-00030) |
| G031 | `Pigmented purpura/capil1.jpg` | `Purpura/20.jpg` | [DermNet](https://dermnetnz.org/topics/capillaritis); [DermNet](https://dermnetnz.org/imagedetail/9593-capillaritis) |
| G032 | `Pigmented purpura/capillaritis-38.jpg` | `Varicose veins/01.jpg` | [DermNet](https://dermnetnz.org/images/capillaritis-images) |
| G033 | `Pigmented purpura/capillaritis-47.jpg` | `Purpura/21.jpg` | [DermNet](https://dermnetnz.org/images/capillaritis-images) |
| G034 | `Pigmented purpura/purpura1.jpg` | `Purpura/82.jpg` | [DermNet](https://dermnetnz.org/images/capillaritis-images) |
| G048 | `Purpura/101.jpg` | `Plasmacytoma/04.jpg` | [DermNet](https://dermnetnz.org/imagedetail/9637-senile-purpura) |
| G049 | `Porphyria cutanea tarda/04.jpg` | `Pyoderma/07.jpg` | [DermNet](https://dermnetnz.org/imagedetail/6832-porphyria-cutanea-tarda) |
| G050 | `outputs/image-dedup-20261002/pending_label_review/G050/21.jpg` | `Postinflammatory hyperpigmentation/21.jpg`<br>`Prurigo/64.jpg` | Chưa xác minh được nguồn |
| G051 | `Pretibial myxoedema/10.jpg` | `Postinflammatory hyperpigmentation/30.jpg` | [DermNet](https://dermnetnz.org/images/pretibial-myxoedema-images) |
| G052 | `Pretibial myxoedema/11.jpg` | `Postinflammatory hyperpigmentation/31.jpg` | [DermNet](https://dermnetnz.org/images/pretibial-myxoedema-images) |
| G053 | `Pruritus/07.jpg` | `Prurigo/06.jpg` | [DermNet](https://dermnetnz.org/imagedetail/5654-scratching-relating-to-dermatitis) |
| G054 | `Prurigo/98.jpg` | `Pruritus/42.jpg` | [DermNet](https://dermnetnz.org/imagedetail/3454-pruritus) |
| G055 | `Varicose veins/05.jpg` | `Prurigo/99.jpg` | [DermNet](https://dermnetnz.org/topics/varicose-veins) |
| G056 | `Pruritus/56.jpg` | `Purpura/98.jpg` | [DermNet](https://dermnetnz.org/imagedetail/6839-pruritus) |
| G060 | `Scurvy/01.jpg` | `Purpura/96.jpg` | [DermNet](https://dermnetnz.org/topics/scurvy); [DermNet](https://dermnetnz.org/imagedetail/7576-scurvy) |
| G061 | `Scurvy/02.jpg` | `Purpura/97.jpg` | [DermNet](https://dermnetnz.org/topics/scurvy); [DermNet](https://dermnetnz.org/imagedetail/7615-scurvy) |
| G062 | `Tinea capitis/01.jpg` | `Sarcoidosis/01.jpg` | [DermNet](https://dermnetnz.org/topics/tinea-capitis) |

## Những trường hợp cần đọc đúng

- G003: cả nguồn lupus ở da và lupus hệ thống đều có ảnh này. Giữ bản có chú thích lupus hệ thống rõ; giữ nhãn lupus ở da trong metadata nguồn, không kết luận nhãn này sai.
- G015: tên tệp milia08 nằm ở trang milia, nhưng chú thích chính ảnh ghi Porphyria cutanea tarda. Quyết định theo chú thích ảnh, không theo tên tệp/trang.
- G016: nguồn xác nhận viêm niêm mạc miệng do RIME, chưa xác nhận tác nhân Mycoplasma cho ca này. Giữ ở Oral mucositis - Stomatitis; lưu điều kiện nguồn RIME.
- G026–G027: nguồn là acute lipodermatosclerosis. Giữ trong nhóm cha Panniculitis hiện có và lưu tên nguồn chính xác; DermNet liệt kê lipodermatosclerosis trong nhóm panniculitis. Không thêm lớp bệnh mới.
- G034: hai URL có cùng tên purpura1 nhưng chỉ ảnh nguồn caption pigmented purpuric dermatitis khớp pixel. Không chọn nhãn chỉ theo tên tệp.
- G053: nguồn là gãi liên quan dermatitis; giữ ngữ cảnh Pruritus, không suy diễn thành prurigo nguyên phát.
- G054: tiêu đề nguồn là Pruritus nhưng mô tả ảnh cụ thể ghi Prurigo in HIV disease; giữ Prurigo và lưu ngữ cảnh nguồn.

## Một ảnh chờ xác minh nhãn — G050

Hai đường dẫn cũ: `Postinflammatory hyperpigmentation/21.jpg` và `Prurigo/64.jpg`. Hai tệp giống hoàn toàn. Chưa có nguồn đủ để chọn một trong hai nhãn. Đã giữ một ảnh đại diện ở `pending_label_review/G050/21.jpg`, ngoài tập ảnh gán nhãn đang dùng. Không tự chọn nhãn dựa trên hình ảnh.

![G050 — một bản duy nhất chờ xác minh nhãn](D:/VuLapTrinh2/DermNet_Dataset/outputs/image-dedup-20261002/pending_label_review/G050/21.jpg)

## Nhật ký và khôi phục

- `image_alias_map.json`: toàn bộ nhóm trùng, đường dẫn cũ, bản giữ và các nhãn nguồn được bảo toàn.
- `removed_to_kept.csv`: bảng bản bỏ → bản giữ, kèm lý do.
- `cross_folder_decisions.json`: nguồn, SHA/pixel nguồn, lý do chọn từng nhóm; không phải xác nhận bác sĩ duyệt nhãn.
- `removed_images_backup.zip`: toàn bộ 65 tệp gốc bị bỏ/chuyển khỏi images.
- `restore_images.py`: mặc định chỉ kiểm tra. Dùng `--apply` để khôi phục các đường dẫn gốc; không ghi đè tệp đã thay đổi.

```powershell
python outputs/image-dedup-20261002/restore_images.py
python outputs/image-dedup-20261002/restore_images.py --apply
```

## Kiểm chứng và giới hạn

Đã quét lại toàn bộ ảnh đang dùng: 0 nhóm trùng byte, 0 nhóm trùng pixel; toàn bộ nội dung pixel trước xử lý còn trong tập hoặc khu chờ duyệt. Không xoá ảnh chỉ vì trùng tên/mã; không xử lý các biến thể crop, watermark, đổi kích thước hoặc ảnh gần trùng trong lượt này. Không sửa JSON/TSV/câu hỏi trong lượt chỉ xử lý ảnh này.
