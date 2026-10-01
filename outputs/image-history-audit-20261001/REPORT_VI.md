# Kiểm tra ảnh trùng qua mã ảnh và commit — 01/10/2026

Mã ảnh trong báo cáo là tên tệp đầy đủ (kèm đuôi). Phạm vi: dermnet-output/images, 7 commit làm thay đổi thư mục này trong lịch sử Git cục bộ. Không fetch remote; không xoá hay sửa ảnh.

## Kết luận

Bản hiện tại có 7.057 ảnh, 53 nhóm trùng toàn bộ nội dung tệp, tương ứng 53 bản dư. Có 96 nhóm cùng mã ảnh; 94 nhóm chứa nội dung khác nhau, chỉ 2 nhóm có nội dung giống hoàn toàn. Không dùng mã số như 01.jpg để quyết định xoá.

## Đối chiếu commit

| Commit | Số ảnh | Nhóm cùng mã | Cùng mã nhưng khác nội dung | Nhóm trùng tệp | Bản dư |
|---|---:|---:|---:|---:|---:|
| 96a1a9f0 | 8804 | 322 | 120 | 1011 | 1064 |
| a68914a1 | 8805 | 322 | 120 | 1011 | 1064 |
| 0c6cadab | 8767 | 322 | 120 | 1008 | 1061 |
| 8fecb837 | 0 | 0 | 0 | 0 | 0 |
| 2f6baf91 | 8767 | 322 | 120 | 1008 | 1061 |
| 24177bb0 | 8802 | 322 | 120 | 1011 | 1064 |
| 2736c270 | 7057 | 96 | 94 | 53 | 53 |

Commit 8fecb837 không chứa ảnh tại đường dẫn này; số 0 không chứng minh đã khử trùng. Commit 2f6baf91 đưa thư mục trở lại.

## Các bộ còn trùng nội dung

| Bộ/thư mục chứa ảnh trùng | Số nhóm |
|---|---:|
| Pityriasis rubra pilaris | 13 |
| Lichen planus ↔ Postinflammatory hyperpigmentation | 3 |
| Pigmented purpura ↔ Purpura | 3 |
| Atopic dermatitis ↔ Prurigo | 2 |
| Epidermolysis bullosa acquisita ↔ Postinflammatory hyperpigmentation | 2 |
| Erythema multiforme ↔ Mycoplasma pneumoniae infection | 2 |
| Herpes simplex ↔ Oral candidiasis | 2 |
| Palmoplantar keratoderma ↔ Pityriasis rubra pilaris | 2 |
| Panniculitis ↔ Venous eczema | 2 |
| Pemphigoid gestationis ↔ Postinflammatory hyperpigmentation | 2 |
| Postinflammatory hyperpigmentation ↔ Pretibial myxoedema | 2 |
| Prurigo ↔ Pruritus | 2 |
| Purpura ↔ Scurvy | 2 |
| Leg ulcer ↔ Pyoderma | 1 |
| Milium ↔ Porphyria cutanea tarda | 1 |
| Mycoplasma pneumoniae infection ↔ Oral mucositis - Stomatitis | 1 |
| Onychocryptosis ↔ Paronychia | 1 |
| Onychomycosis ↔ Tinea manuum | 1 |
| Oral candidiasis ↔ Oral mucositis - Stomatitis | 1 |
| Pediculosis capitis ↔ Pyoderma | 1 |
| Pigmented purpura ↔ Varicose veins | 1 |
| Plasmacytoma ↔ Purpura | 1 |
| Porphyria cutanea tarda ↔ Pyoderma | 1 |
| Postinflammatory hyperpigmentation ↔ Prurigo | 1 |
| Prurigo ↔ Varicose veins | 1 |
| Pruritus ↔ Purpura | 1 |
| Sarcoidosis ↔ Tinea capitis | 1 |

## Toàn bộ nhóm trùng còn lại

1. `Atopic dermatitis/atopic-dermatitis-0049.jpg` ↔ `Prurigo/05.jpg`
2. `Atopic dermatitis/eapsimplex1.jpg` ↔ `Prurigo/09.jpg`
3. `Epidermolysis bullosa acquisita/epidermolysis-bullosa-acquisita-0002.jpg` ↔ `Postinflammatory hyperpigmentation/10.jpg`
4. `Epidermolysis bullosa acquisita/epidermolysis-bullosa-acquisita-0003.jpg` ↔ `Postinflammatory hyperpigmentation/11.jpg`
5. `Erythema multiforme/mycoplasma1.jpg` ↔ `Mycoplasma pneumoniae infection/mycoplasma1.jpg`
6. `Erythema multiforme/mycoplasma2.jpg` ↔ `Mycoplasma pneumoniae infection/mycoplasma2.jpg`
7. `Herpes simplex/herpes-simplex-labialis-38.jpg` ↔ `Oral candidiasis/05.jpg`
8. `Herpes simplex/herpes-simplex-labialis-39.jpg` ↔ `Oral candidiasis/06.jpg`
9. `Leg ulcer/3042.jpg` ↔ `Pyoderma/01.jpg`
10. `Lichen planus/annular-lichen-planus-0007.jpg` ↔ `Postinflammatory hyperpigmentation/05.jpg`
11. `Lichen planus/papular-lichen-planus-0009.jpg` ↔ `Postinflammatory hyperpigmentation/15.jpg`
12. `Lichen planus/papular-lichen-planus-0010.jpg` ↔ `Postinflammatory hyperpigmentation/16.jpg`
13. `Milium/milia08.jpg` ↔ `Porphyria cutanea tarda/03.jpg`
14. `Mycoplasma pneumoniae infection/reactive-infectious-mucocutaneous-eruption-0008.jpg` ↔ `Oral mucositis - Stomatitis/81.jpg`
15. `Onychocryptosis/01.jpg` ↔ `Paronychia/3296.jpg`
16. `Onychomycosis/47.jpg` ↔ `Tinea manuum/05.jpg`
17. `Oral candidiasis/03.jpg` ↔ `Oral mucositis - Stomatitis/18.jpg`
18. `Palmoplantar keratoderma/35.jpg` ↔ `Pityriasis rubra pilaris/11.jpg`
19. `Palmoplantar keratoderma/36.jpg` ↔ `Pityriasis rubra pilaris/12.jpg`
20. `Panniculitis/03.jpg` ↔ `Venous eczema/acute-lipodermatosclerosis-21.jpg`
21. `Panniculitis/04.jpg` ↔ `Venous eczema/acute-lipodermatosclerosis-22.jpg`
22. `Pediculosis capitis/12.jpg` ↔ `Pyoderma/08.jpg`
23. `Pemphigoid gestationis/37.jpg` ↔ `Postinflammatory hyperpigmentation/24.jpg`
24. `Pemphigoid gestationis/38.jpg` ↔ `Postinflammatory hyperpigmentation/25.jpg`
25. `Pigmented purpura/capil1.jpg` ↔ `Purpura/20.jpg`
26. `Pigmented purpura/capillaritis-38.jpg` ↔ `Varicose veins/01.jpg`
27. `Pigmented purpura/capillaritis-47.jpg` ↔ `Purpura/21.jpg`
28. `Pigmented purpura/purpura1.jpg` ↔ `Purpura/82.jpg`
29. `Pityriasis rubra pilaris/15.jpg` ↔ `Pityriasis rubra pilaris/prp1.jpg`
30. `Pityriasis rubra pilaris/16.jpg` ↔ `Pityriasis rubra pilaris/prp10.jpg`
31. `Pityriasis rubra pilaris/17.jpg` ↔ `Pityriasis rubra pilaris/prp11.jpg`
32. `Pityriasis rubra pilaris/18.jpg` ↔ `Pityriasis rubra pilaris/prp12.jpg`
33. `Pityriasis rubra pilaris/19.jpg` ↔ `Pityriasis rubra pilaris/prp13.jpg`
34. `Pityriasis rubra pilaris/20.jpg` ↔ `Pityriasis rubra pilaris/prp2.jpg`
35. `Pityriasis rubra pilaris/21.jpg` ↔ `Pityriasis rubra pilaris/prp3.jpg`
36. `Pityriasis rubra pilaris/23.jpg` ↔ `Pityriasis rubra pilaris/prp4.jpg`
37. `Pityriasis rubra pilaris/24.jpg` ↔ `Pityriasis rubra pilaris/prp5.jpg`
38. `Pityriasis rubra pilaris/25.jpg` ↔ `Pityriasis rubra pilaris/prp6.jpg`
39. `Pityriasis rubra pilaris/26.jpg` ↔ `Pityriasis rubra pilaris/prp7.jpg`
40. `Pityriasis rubra pilaris/27.jpg` ↔ `Pityriasis rubra pilaris/prp8.jpg`
41. `Pityriasis rubra pilaris/29.jpg` ↔ `Pityriasis rubra pilaris/prp9.jpg`
42. `Plasmacytoma/04.jpg` ↔ `Purpura/101.jpg`
43. `Porphyria cutanea tarda/04.jpg` ↔ `Pyoderma/07.jpg`
44. `Postinflammatory hyperpigmentation/21.jpg` ↔ `Prurigo/64.jpg`
45. `Postinflammatory hyperpigmentation/30.jpg` ↔ `Pretibial myxoedema/10.jpg`
46. `Postinflammatory hyperpigmentation/31.jpg` ↔ `Pretibial myxoedema/11.jpg`
47. `Prurigo/06.jpg` ↔ `Pruritus/07.jpg`
48. `Prurigo/98.jpg` ↔ `Pruritus/42.jpg`
49. `Prurigo/99.jpg` ↔ `Varicose veins/05.jpg`
50. `Pruritus/56.jpg` ↔ `Purpura/98.jpg`
51. `Purpura/96.jpg` ↔ `Scurvy/01.jpg`
52. `Purpura/97.jpg` ↔ `Scurvy/02.jpg`
53. `Sarcoidosis/01.jpg` ↔ `Tinea capitis/01.jpg`

## Giới hạn và hướng xử lý

Báo cáo audit-images cũ dựa trên 7.692 ảnh, khác danh mục hiện tại ở 943 đường dẫn (gồm thêm, mất hoặc thay đổi). Không áp dụng trực tiếp số 366 nhóm trùng pixel của báo cáo cũ cho bộ hiện tại. Lần kiểm tra này xác nhận trùng tệp bằng Git blob và đọc SHA-256 ảnh hiện tại đối chiếu danh mục cũ; chưa quét lại pixel/gần trùng cho toàn bộ ảnh.

Nên rà 53 nhóm trong danh sách, chọn bản giữ theo nhãn và nguồn, lưu liên kết mã cũ–mã giữ trước khi khử trùng. Cùng nội dung ở hai nhãn không tự chứng minh nhãn nào đúng.

Đã đọc lại và xác nhận SHA-256 giống nhau cho cả 53 nhóm (106 tệp) trong thư mục làm việc. Cả 53 nội dung trùng đều đã xuất hiện trong nhóm trùng ở commit 24177bb0.
