# DermNet VQA: cây thuật ngữ và danh sách nhãn

Ngày tạo: 2026-09-25

## Tóm tắt

Lập từ 2,721 dòng Val_4k và 23,681 dòng Test: tổng 26,402 QA, 7,870 ảnh không giao nhau. Test_1of3 có 7,891 dòng và là tập con của Test; không cộng thêm vào tổng chính.
Từ điển máy đọc được có 3,621 bản ghi theo tổ hợp nhóm/nhãn, gồm 3,344 chuỗi nhãn riêng biệt trong từng nhóm chính; cùng một nhãn có thể xuất hiện ở nhiều nhóm nguồn. Trong đó có 493 bản ghi vị trí cơ thể, 36 bản ghi khoang miệng/niêm mạc, 115 bản ghi kiểu phân bố, 207 bản ghi loại tổn thương, 354 bản ghi đáp án bệnh danh, 381 bản ghi bệnh danh nguồn và 1,944 bản ghi thuộc tính.

## Cách áp dụng cấu trúc bài báo

MedLesionVQA trình bày vị trí cơ thể theo cây nhiều cấp, tách cây khoang miệng, rồi liệt kê riêng tổn thương, bệnh và giá trị thuộc tính. Bản này áp dụng cùng cách tổ chức nhưng chỉ đưa vào thuật ngữ có trong các TSV đã rà. Trường Anatomical_Distribution của DermNet đang gộp vị trí, kiểu phân bố và một số loại thông tin khác; chỉ các cụm nhận diện an toàn mới được tách. Giá trị chưa phân loại chắc chắn được giữ trong nhánh riêng để rà soát, không ép vào cây vị trí.

Nguồn phương pháp: [MedLesionVQA, ICLR 2026, phần Annotation Protocol và Supplementary Tables 3–7](https://proceedings.iclr.cc/paper_files/paper/2026/file/d82c24b7a4237aa4283b38e12047dc38-Paper-Conference.pdf). PDF người dùng gửi: `15645_MedLesionVQA_A_Multimoda (2).pdf`.

## Cây vị trí cơ thể Level 1–4

Chỉ các nhãn quan sát được trong đầu vào mới được đưa vào cây. Bảng dùng bốn cột phân cấp như bài báo. Các nhãn có vị trí rộng hoặc chưa xác định được chi/vùng cụ thể được đánh dấu `requires_manual_body_region_mapping` trong TSV.

| Level 1 | Level 2 | Level 3 | Level 4 | Nhãn trong dữ liệu | QA dương Val | QA dương Test | Judgement phủ định Test |
|---|---|---|---|---|---:|---:|---:|
| Chi | Ngón chưa xác định chi | Các ngón |  | Các ngón | 0 | 2 | 0 |
| Chi | Ngón chưa xác định chi | Ngón |  | Ngón | 0 | 1 | 0 |
| Chi | Ngón chưa xác định chi | Ngón cái |  | Ngón cái | 0 | 4 | 0 |
| Chi | Ngón chưa xác định chi | Đầu ngón |  | Đầu ngón | 1 | 18 | 0 |
| Chi | Vị trí chi chưa xác định |  |  | Chi | 3 | 68 | 0 |
| Chi | Vị trí chi chưa xác định |  |  | Chi có lông | 0 | 1 | 0 |
| Chi | Vị trí chi chưa xác định |  |  | Da chi lông | 0 | 1 | 0 |
| Chi | Vị trí chi chưa xác định | Da vùng chi |  | Da vùng chi | 0 | 1 | 0 |
| Chi | Vị trí chi chưa xác định | Nếp gấp tay |  | Nếp gấp tay | 0 | 1 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Bản móng | Bản móng | 0 | 4 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Bờ bên móng | Bờ bên móng | 0 | 6 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Bờ móng bên | Bờ móng bên | 1 | 0 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Dưới móng | Dưới móng | 0 | 3 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Gần gốc móng | Gần gốc móng | 0 | 1 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Hai móng | Hai móng | 0 | 1 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Móng | Móng | 0 | 4 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Nếp móng | Nếp móng | 1 | 6 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Nếp móng bên | Nếp móng bên | 0 | 9 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Nếp móng gần | Nếp móng gần | 2 | 10 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Phiến móng | Phiến móng | 0 | 2 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Quanh bờ móng | Quanh bờ móng | 0 | 1 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Quanh gốc móng | Quanh gốc móng | 0 | 1 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Quanh móng | Quanh móng | 9 | 92 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Rãnh móng bên | Rãnh móng bên | 0 | 1 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | rìa bên móng | rìa bên móng | 1 | 0 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Rìa móng | Rìa móng | 0 | 2 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Rìa xa móng | Rìa xa móng | 0 | 1 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Sát móng | Sát móng | 0 | 2 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Toàn bộ móng | Toàn bộ móng | 0 | 1 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Đầu móng | Đầu móng | 0 | 5 | 0 |
| Chi chưa xác định | Móng | Vị trí móng chưa xác định | Đầu xa móng | Đầu xa móng | 1 | 3 | 0 |
| Chi chưa xác định | Ngón chưa xác định chi | Bên ngón |  | Bên ngón | 0 | 1 | 0 |
| Chi chưa xác định | Ngón chưa xác định chi | Gan ngón |  | Gan ngón | 0 | 1 | 0 |
| Chi chưa xác định | Ngón chưa xác định chi | gốc ngón cái |  | gốc ngón cái | 1 | 0 | 0 |
| Chi chưa xác định | Ngón chưa xác định chi | khớp ngón |  | khớp ngón | 1 | 1 | 0 |
| Chi chưa xác định | Ngón chưa xác định chi | Mu ngón |  | Mu ngón | 1 | 2 | 0 |
| Chi chưa xác định | Ngón chưa xác định chi | Mu đốt ngón |  | Mu đốt ngón | 0 | 1 | 0 |
| Chi chưa xác định | Ngón chưa xác định chi | Móng ngón cái |  | Móng ngón cái | 0 | 1 | 0 |
| Chi chưa xác định | Ngón chưa xác định chi | Ngón áp út |  | Ngón áp út | 0 | 1 | 0 |
| Chi chưa xác định | Ngón chưa xác định chi | Rìa ngón |  | Rìa ngón | 0 | 1 | 0 |
| Chi chưa xác định | Ngón chưa xác định chi | Đốt ngón |  | Đốt ngón | 0 | 1 | 0 |
| Chi dưới |  |  |  | Chi dưới | 6 | 29 | 71 |
| Chi dưới | Bàn chân |  |  | bàn chân | 5 | 36 | 67 |
| Chi dưới | Bàn chân | Gót chân |  | Gót chân | 5 | 48 | 0 |
| Chi dưới | Bàn chân | Gót chân |  | Vùng gót | 0 | 1 | 0 |
| Chi dưới | Bàn chân | Gót chân |  | Vùng gót chân | 1 | 1 | 0 |
| Chi dưới | Bàn chân | Lòng bàn chân |  | Lòng bàn chân | 12 | 97 | 0 |
| Chi dưới | Bàn chân | Mu bàn chân |  | Mu bàn chân | 6 | 56 | 0 |
| Chi dưới | Bàn chân | Ngón chân |  | Ngón chân | 7 | 44 | 0 |
| Chi dưới | Bàn chân | Ngón chân | Kẽ ngón chân | kẽ ngón chân | 2 | 13 | 0 |
| Chi dưới | Bàn chân | Ngón chân | Móng chân | Móng chân | 2 | 15 | 0 |
| Chi dưới | Bàn chân | Ngón chân | Móng chân | Móng chân cái | 0 | 3 | 0 |
| Chi dưới | Bàn chân | Ngón chân | Móng chân | Móng ngón chân | 0 | 11 | 0 |
| Chi dưới | Bàn chân | Ngón chân | Móng chân | Móng ngón chân cái | 0 | 3 | 0 |
| Chi dưới | Bàn chân | Ngón chân | Móng chân | Quanh móng chân | 0 | 1 | 0 |
| Chi dưới | Bàn chân | Ngón chân | Móng chân | Rìa móng chân | 0 | 1 | 0 |
| Chi dưới | Bàn chân | Ngón chân | Ngón chân cái | Ngón chân cái | 1 | 20 | 0 |
| Chi dưới | Bàn chân | Ngón chân | Đầu ngón chân | Đầu ngón chân | 2 | 13 | 0 |
| Chi dưới | Bàn chân | Vòm bàn chân |  | Vòm bàn chân | 1 | 3 | 0 |
| Chi dưới | Cẳng chân |  |  | Cẳng chân | 43 | 311 | 0 |
| Chi dưới | Cẳng chân | Cẳng chân dưới |  | Cẳng chân dưới | 1 | 5 | 0 |
| Chi dưới | Cổ chân |  |  | Cổ chân | 15 | 114 | 0 |
| Chi dưới | Cổ chân | Gân gót |  | Vùng gân gót | 0 | 1 | 0 |
| Chi dưới | Cổ chân | Mắt cá |  | Mắt cá | 0 | 1 | 0 |
| Chi dưới | Cổ chân | Mắt cá |  | Mắt cá chân | 1 | 1 | 0 |
| Chi dưới | Cổ chân | Mắt cá |  | quanh mắt cá | 0 | 2 | 0 |
| Chi dưới | Cổ chân | Quanh cổ chân |  | Quanh cổ chân | 1 | 4 | 0 |
| Chi dưới | Cổ chân | Vùng cổ chân |  | Vùng cổ chân | 1 | 1 | 0 |
| Chi dưới | Gối |  |  | Gối | 3 | 12 | 0 |
| Chi dưới | Gối | Hai gối |  | Hai gối | 0 | 2 | 0 |
| Chi dưới | Gối | Nếp khoeo |  | Khoeo chân | 0 | 4 | 0 |
| Chi dưới | Gối | Nếp khoeo |  | Nếp khoeo | 1 | 2 | 0 |
| Chi dưới | Gối | Quanh gối |  | Quanh gối | 3 | 7 | 0 |
| Chi dưới | Gối | Vùng gối |  | Vùng gối | 0 | 1 | 0 |
| Chi dưới | Mông |  |  | Mông | 4 | 29 | 0 |
| Chi dưới | Mông | Nếp liên mông |  | Nếp liên mông | 0 | 7 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Bờ bàn chân |  | Bờ bàn chân | 0 | 2 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Bờ ngoài bàn chân |  | Bờ ngoài bàn chân | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Bờ trong bàn chân |  | Bờ trong bàn chân | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Dưới ngón chân |  | Dưới ngón chân | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Gan ngón chân |  | Gan ngón chân | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | hai bàn chân |  | hai bàn chân | 0 | 2 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Hai đùi |  | Hai đùi | 1 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | hông đùi |  | hông đùi | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Khe liên mông |  | Khe liên mông | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Khe mông |  | Khe mông | 0 | 3 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Kẽ liên mông |  | Kẽ liên mông | 0 | 3 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Kẽ mông |  | Kẽ mông | 0 | 6 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Mông bẹn |  | Mông bẹn | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Mông trên |  | Mông trên | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Nếp gấp cổ chân |  | Nếp gấp cổ chân | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Nếp gấp gối |  | Nếp gấp gối | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Nếp gấp đùi |  | Nếp gấp đùi | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Nếp kẽ mông |  | Nếp kẽ mông | 0 | 5 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Nếp mông |  | Nếp mông | 1 | 3 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | quanh lỗ chân lông |  | quanh lỗ chân lông | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Rãnh liên mông |  | Rãnh liên mông | 0 | 4 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Rìa bàn chân |  | Rìa bàn chân | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Trước cẳng chân |  | Trước cẳng chân | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Vòm chân |  | Vòm chân | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Vòm gan chân |  | Vòm gan chân | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Vùng cẳng chân |  | Vùng cẳng chân | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Vùng mông |  | Vùng mông | 0 | 8 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Vùng đùi |  | Vùng đùi | 0 | 4 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Vùng đầu gối |  | Vùng đầu gối | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Đùi háng |  | Đùi háng | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Đùi trong |  | Đùi trong | 0 | 2 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Đùi trên |  | Đùi trên | 1 | 2 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Đùi trước |  | Đùi trước | 1 | 0 | 0 |
| Chi dưới | Vị trí chi dưới chưa phân nhóm | Đầu gối |  | Đầu gối | 0 | 1 | 0 |
| Chi dưới | Vị trí chi dưới chưa xác định | Hai chân |  | Hai chân | 0 | 3 | 0 |
| Chi dưới | Đùi |  |  | Đùi | 5 | 49 | 0 |
| Chi dưới | Đùi | Mặt sau đùi |  | Mặt sau đùi | 1 | 6 | 0 |
| Chi dưới | Đùi | Mặt trong đùi |  | Mặt trong đùi | 1 | 12 | 0 |
| Chi trên |  |  |  | Chi trên | 1 | 6 | 62 |
| Chi trên | Bàn tay |  |  | bàn tay | 1 | 14 | 59 |
| Chi trên | Bàn tay | Hai bàn tay |  | Hai bàn tay | 0 | 11 | 0 |
| Chi trên | Bàn tay | Lòng bàn tay |  | Lòng bàn tay | 22 | 131 | 0 |
| Chi trên | Bàn tay | Mu bàn tay |  | Mu bàn tay | 23 | 181 | 0 |
| Chi trên | Bàn tay | Mu hai bàn tay |  | Mu hai bàn tay | 0 | 3 | 0 |
| Chi trên | Bàn tay | Mô cái |  | Mô cái | 0 | 4 | 0 |
| Chi trên | Bàn tay | Mô cái |  | Vùng mô cái | 0 | 1 | 0 |
| Chi trên | Bàn tay | Ngón tay |  | ngón tay | 24 | 147 | 0 |
| Chi trên | Bàn tay | Ngón tay | Khớp ngón tay | Khớp ngón tay | 0 | 12 | 0 |
| Chi trên | Bàn tay | Ngón tay | Kẽ ngón tay | Kẽ ngón tay | 1 | 14 | 0 |
| Chi trên | Bàn tay | Ngón tay | Mu ngón tay | Mu ngón tay | 2 | 11 | 0 |
| Chi trên | Bàn tay | Ngón tay | Móng tay | Móng ngón tay | 0 | 1 | 0 |
| Chi trên | Bàn tay | Ngón tay | Móng tay | Móng tay | 5 | 47 | 0 |
| Chi trên | Bàn tay | Ngón tay | Ngón tay cái | Ngón tay cái | 0 | 4 | 0 |
| Chi trên | Bàn tay | Ngón tay | Đầu ngón tay | Đầu ngón tay | 3 | 29 | 0 |
| Chi trên | Cánh tay |  |  | Cánh tay | 5 | 59 | 0 |
| Chi trên | Cánh tay | Cánh tay trên |  | Cánh tay trên | 2 | 10 | 0 |
| Chi trên | Cánh tay | Mặt ngoài cánh tay |  | Mặt ngoài cánh tay | 0 | 2 | 0 |
| Chi trên | Cánh tay | Mặt trong cánh tay |  | Mặt trong cánh tay | 0 | 3 | 0 |
| Chi trên | Cẳng tay |  |  | Cẳng tay | 22 | 165 | 0 |
| Chi trên | Cổ tay |  |  | Cổ tay | 6 | 56 | 0 |
| Chi trên | Khuỷu tay |  |  | Khuỷu tay | 3 | 28 | 0 |
| Chi trên | Khuỷu tay | Nếp khuỷu |  | Gấp khuỷu tay | 0 | 2 | 0 |
| Chi trên | Khuỷu tay | Nếp khuỷu |  | Nếp gấp khuỷu tay | 0 | 4 | 0 |
| Chi trên | Khuỷu tay | Nếp khuỷu |  | Nếp khuỷu | 0 | 4 | 0 |
| Chi trên | Khuỷu tay | Nếp khuỷu |  | Nếp khuỷu tay | 0 | 5 | 0 |
| Chi trên | Vai |  |  | Vai | 8 | 102 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Bàn tay phải |  | Bàn tay phải | 1 | 0 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Bờ bàn tay |  | Bờ bàn tay | 0 | 1 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Bờ ngón tay |  | Bờ ngón tay | 0 | 1 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Các ngón tay |  | Các ngón tay | 0 | 1 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Cánh tay ngoài |  | Cánh tay ngoài | 0 | 1 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Cánh tay trong |  | Cánh tay trong | 0 | 2 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Cổ vai lưng |  | Cổ vai lưng | 0 | 1 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Da lòng bàn tay |  | Da lòng bàn tay | 0 | 1 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | gan bàn tay |  | gan bàn tay | 0 | 0 | 20 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Gốc ngón tay |  | Gốc ngón tay | 0 | 1 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Hai cánh tay |  | Hai cánh tay | 0 | 1 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Hai cẳng tay |  | Hai cẳng tay | 0 | 2 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Hai khuỷu tay |  | Hai khuỷu tay | 0 | 2 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Hai lòng bàn tay |  | Hai lòng bàn tay | 0 | 2 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Hai vai |  | Hai vai | 1 | 0 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | khuỷu |  | khuỷu | 1 | 3 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Một ngón tay |  | Một ngón tay | 0 | 1 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Ngực vai |  | Ngực vai | 0 | 2 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Nếp gấp cổ tay |  | Nếp gấp cổ tay | 0 | 1 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Nếp gấp khuỷu |  | Nếp gấp khuỷu | 0 | 3 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Rìa ngón tay |  | Rìa ngón tay | 0 | 1 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Vai cổ |  | Vai cổ | 1 | 0 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Vai gáy |  | Vai gáy | 0 | 1 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Vùng cánh tay |  | Vùng cánh tay | 0 | 1 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Vùng khuỷu tay |  | Vùng khuỷu tay | 1 | 2 | 0 |
| Chi trên | Vị trí chi trên chưa phân nhóm | Đốt ngón tay |  | Đốt ngón tay | 0 | 1 | 0 |
| Chi trên | Vị trí chi trên chưa xác định | Hai tay |  | Hai tay | 1 | 4 | 0 |
| Chi trên | Vị trí chi trên chưa xác định | tay |  | tay | 3 | 7 | 20 |
| Chưa phân nhánh giải phẫu | Gần khớp ngón cái |  |  | Gần khớp ngón cái | 0 | 1 | 0 |
| Chưa phân nhánh giải phẫu | Gốc ngón |  |  | Gốc ngón | 0 | 1 | 0 |
| Chưa phân nhánh giải phẫu | Hai khoeo |  |  | Hai khoeo | 0 | 2 | 0 |
| Chưa phân nhánh giải phẫu | Hai ngón |  |  | Hai ngón | 0 | 1 | 0 |
| Chưa phân nhánh giải phẫu | Khoeo |  |  | Khoeo | 0 | 1 | 0 |
| Chưa phân nhánh giải phẫu | Mu khớp ngón |  |  | Mu khớp ngón | 0 | 1 | 0 |
| Chưa phân nhánh giải phẫu | Quanh ngón |  |  | Quanh ngón | 1 | 0 | 0 |
| Chưa phân nhánh giải phẫu | Vùng khoeo |  |  | Vùng khoeo | 0 | 1 | 0 |
| Cổ |  |  |  | Cổ | 9 | 85 | 0 |
| Cổ | Cổ bên |  |  | Cổ bên | 1 | 25 | 0 |
| Cổ | Cổ sau |  |  | Cổ sau | 0 | 3 | 0 |
| Cổ | Cổ trước |  |  | Cổ trước | 1 | 16 | 0 |
| Cổ | Gáy |  |  | Cổ gáy | 0 | 5 | 0 |
| Cổ | Gáy |  |  | Gáy | 0 | 25 | 0 |
| Cổ | Vị trí cổ chưa phân nhóm | Bên cổ |  | Bên cổ | 0 | 1 | 0 |
| Cổ | Vị trí cổ chưa phân nhóm | Cổ trước-ngực trên |  | Cổ trước-ngực trên | 0 | 1 | 0 |
| Cổ | Vị trí cổ chưa phân nhóm | Cổ-ngực |  | Cổ-ngực | 1 | 0 | 0 |
| Cổ | Vị trí cổ chưa phân nhóm | Gáy sau |  | Gáy sau | 0 | 1 | 0 |
| Cổ | Vị trí cổ chưa phân nhóm | Nếp cổ |  | Nếp cổ | 0 | 5 | 0 |
| Cổ | Vị trí cổ chưa phân nhóm | Nếp gấp cổ |  | Nếp gấp cổ | 1 | 3 | 0 |
| Cổ | Vị trí cổ chưa phân nhóm | Vùng cổ bên |  | Vùng cổ bên | 0 | 2 | 0 |
| Cổ | Vị trí cổ chưa phân nhóm | Vùng cổ gáy |  | Vùng cổ gáy | 0 | 2 | 0 |
| Cổ | Vị trí cổ chưa phân nhóm | Vùng cổ ngực |  | Vùng cổ ngực | 0 | 3 | 0 |
| Cổ | Vị trí cổ chưa phân nhóm | Vùng cổ trước |  | Vùng cổ trước | 0 | 1 | 0 |
| Cổ | Vị trí cổ chưa phân nhóm | Vùng gáy |  | Vùng gáy | 0 | 1 | 0 |
| Da | Nếp gấp |  |  | Nếp gấp | 12 | 117 | 0 |
| Da | Nếp gấp/kẽ | kẽ ngón |  | kẽ ngón | 1 | 7 | 0 |
| Da | Nếp gấp/kẽ | Kẽ ngón cái |  | Kẽ ngón cái | 0 | 1 | 0 |
| Da | Nếp gấp/kẽ | Kẽ sinh dục |  | Kẽ sinh dục | 0 | 1 | 0 |
| Da | Nếp gấp/kẽ | Mép nếp gấp |  | Mép nếp gấp | 0 | 1 | 0 |
| Da | Nếp gấp/kẽ | Nếp bẹn trái |  | Nếp bẹn trái | 0 | 2 | 0 |
| Da | Nếp gấp/kẽ | nếp bụng |  | nếp bụng | 0 | 1 | 0 |
| Da | Nếp gấp/kẽ | nếp da |  | nếp da | 1 | 4 | 0 |
| Da | Nếp gấp/kẽ | Nếp dưới bụng |  | Nếp dưới bụng | 0 | 1 | 0 |
| Da | Nếp gấp/kẽ | Nếp gấp bẹn |  | Nếp gấp bẹn | 0 | 3 | 0 |
| Da | Nếp gấp/kẽ | Nếp gấp bụng |  | Nếp gấp bụng | 0 | 1 | 0 |
| Da | Nếp gấp/kẽ | Nếp gấp chi |  | Nếp gấp chi | 0 | 3 | 0 |
| Da | Nếp gấp/kẽ | Nếp gấp nách |  | Nếp gấp nách | 0 | 2 | 0 |
| Da | Nếp gấp/kẽ | Nếp kẽ |  | Nếp kẽ | 0 | 18 | 0 |
| Da | Nếp gấp/kẽ | Nếp kẽ hậu môn |  | Nếp kẽ hậu môn | 0 | 1 | 0 |
| Da | Nếp gấp/kẽ | Nếp kẽ quanh sinh dục |  | Nếp kẽ quanh sinh dục | 0 | 1 | 0 |
| Da | Nếp gấp/kẽ | Nếp nách |  | Nếp nách | 0 | 1 | 0 |
| Da | Nếp gấp/kẽ | Quanh nếp gấp |  | Quanh nếp gấp | 1 | 1 | 0 |
| Da | Nếp gấp/kẽ | Theo kẽ ngón |  | Theo kẽ ngón | 0 | 1 | 0 |
| Da | Nếp gấp/kẽ | Theo nếp gấp |  | Theo nếp gấp | 0 | 2 | 0 |
| Da | Nếp gấp/kẽ | Vùng nếp bẹn |  | Vùng nếp bẹn | 0 | 1 | 0 |
| Da | Vùng da có lông |  |  | Da có lông | 13 | 85 | 0 |
| Da | Vị trí chưa xác định |  |  | Da | 2 | 31 | 0 |
| Da | Vị trí da chưa phân nhóm | Da đầu hói |  | Da đầu hói | 0 | 1 | 0 |
| Niêm mạc | Vị trí niêm mạc chưa xác định | Mép niêm mạc |  | Mép niêm mạc | 0 | 1 | 0 |
| Niêm mạc | Vị trí niêm mạc chưa xác định | Niêm mạc |  | Niêm mạc | 1 | 12 | 0 |
| Niêm mạc | Vị trí niêm mạc chưa xác định | Niêm mạc hậu môn |  | Niêm mạc hậu môn | 0 | 1 | 0 |
| Niêm mạc | Vị trí niêm mạc chưa xác định | Niêm mạc sinh dục |  | Niêm mạc sinh dục | 0 | 1 | 0 |
| Niêm mạc | Vị trí niêm mạc chưa xác định | Niêm mạc âm hộ |  | Niêm mạc âm hộ | 0 | 1 | 0 |
| Niêm mạc | Vị trí niêm mạc chưa xác định | Vùng niêm mạc |  | Vùng niêm mạc | 0 | 1 | 0 |
| Thân mình |  |  |  | Thân mình | 46 | 380 | 77 |
| Thân mình | Bẹn |  |  | Bẹn | 6 | 26 | 0 |
| Thân mình | Bẹn | Nếp bẹn |  | Nếp bẹn | 6 | 40 | 0 |
| Thân mình | Bụng |  |  | Bụng | 8 | 62 | 0 |
| Thân mình | Bụng | bụng bên |  | bụng bên | 0 | 2 | 0 |
| Thân mình | Bụng | Bụng dưới |  | Bụng dưới | 2 | 11 | 0 |
| Thân mình | Bụng | Bụng trên |  | Bụng trên | 0 | 1 | 0 |
| Thân mình | Bụng | Quanh rốn |  | Quanh rốn | 7 | 11 | 0 |
| Thân mình | Hông |  |  | Hông | 1 | 4 | 0 |
| Thân mình | Lưng |  |  | Lưng | 5 | 80 | 0 |
| Thân mình | Lưng | Lưng dưới |  | Lưng dưới | 1 | 3 | 0 |
| Thân mình | Lưng | Lưng trên |  | Lưng trên | 4 | 78 | 0 |
| Thân mình | Mạn sườn |  |  | Mạn sườn | 0 | 3 | 0 |
| Thân mình | Ngực |  |  | Ngực | 5 | 56 | 0 |
| Thân mình | Ngực | Hõm ức |  | Hõm ức | 0 | 1 | 0 |
| Thân mình | Ngực | Ngực bên |  | Ngực bên | 0 | 6 | 0 |
| Thân mình | Ngực | Ngực trên |  | Ngực trên | 3 | 30 | 0 |
| Thân mình | Ngực | Ngực trước |  | ngực trước | 0 | 7 | 0 |
| Thân mình | Ngực | Quanh xương ức |  | Quanh xương ức | 0 | 1 | 0 |
| Thân mình | Ngực | Vùng thượng đòn |  | thượng đòn | 0 | 1 | 0 |
| Thân mình | Ngực | Vú |  | vú | 0 | 1 | 0 |
| Thân mình | Ngực | Vú | Bề mặt vú | Bề mặt vú | 0 | 2 | 0 |
| Thân mình | Ngực | Vú | Núm vú | núm vú | 2 | 3 | 0 |
| Thân mình | Ngực | Vú | Nếp dưới vú | Nếp dưới vú | 2 | 17 | 0 |
| Thân mình | Ngực | Vú | Quanh núm vú | Quanh núm vú | 0 | 3 | 0 |
| Thân mình | Ngực | Vú | Quanh quầng vú | Quanh quầng vú | 0 | 1 | 0 |
| Thân mình | Ngực | Vú | Quầng vú | Quầng vú | 2 | 5 | 0 |
| Thân mình | Ngực | Xương ức |  | Xương ức | 0 | 1 | 0 |
| Thân mình | Nách |  |  | Nách | 11 | 82 | 0 |
| Thân mình | Nách | Hố nách |  | Hố nách | 1 | 4 | 0 |
| Thân mình | Thân dưới |  |  | Thân dưới | 0 | 2 | 0 |
| Thân mình | Thân mình bên |  |  | Thân mình bên | 0 | 2 | 0 |
| Thân mình | thân mình sau |  |  | thân mình sau | 0 | 1 | 0 |
| Thân mình | Thân mình trước |  |  | Thân mình trước | 0 | 1 | 0 |
| Thân mình | Thân trên |  |  | Thân trên | 2 | 8 | 0 |
| Thân mình | Vùng chậu | Vùng chậu |  | Vùng chậu | 0 | 1 | 0 |
| Thân mình | Vùng chậu | Vùng cùng cụt |  | Vùng cùng cụt | 0 | 1 | 0 |
| Thân mình | Vùng tã |  |  | Vùng tã | 2 | 4 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Giữa bụng |  | Giữa bụng | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Giữa ngực |  | Giữa ngực | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Hông bên |  | Hông bên | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Lưng bên |  | Lưng bên | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Lưng giữa |  | Lưng giữa | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Lồng ngực |  | Lồng ngực | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Ngang thân mình |  | Ngang thân mình | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Ngực bụng |  | Ngực bụng | 0 | 2 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Ngực giữa |  | Ngực giữa | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Nách bên |  | Nách bên | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Nách trước |  | Nách trước | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Thành ngực |  | Thành ngực | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Thành ngực bên |  | Thành ngực bên | 0 | 2 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Thân bên |  | Thân bên | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Thắt lưng |  | Thắt lưng | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Thắt lưng cùng |  | Thắt lưng cùng | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Toàn lưng |  | Toàn lưng | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Vùng bụng dưới |  | Vùng bụng dưới | 0 | 3 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Vùng dưới áo ngực |  | Vùng dưới áo ngực | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Vùng giữa ngực |  | Vùng giữa ngực | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Vùng hông |  | Vùng hông | 1 | 2 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Vùng lưng |  | Vùng lưng | 0 | 2 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Vùng lưng dưới |  | Vùng lưng dưới | 0 | 2 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Vùng lưng trên |  | Vùng lưng trên | 0 | 1 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Vùng ngực trên |  | Vùng ngực trên | 0 | 3 | 0 |
| Thân mình | Vị trí thân mình chưa phân nhóm | Vùng thắt lưng |  | Vùng thắt lưng | 0 | 3 | 0 |
| Vùng sinh dục-hậu môn | Bìu |  |  | Bìu | 1 | 10 | 0 |
| Vùng sinh dục-hậu môn | Dương vật |  |  | Dương vật | 1 | 9 | 0 |
| Vùng sinh dục-hậu môn | Dương vật | bao quy đầu |  | bao quy đầu | 0 | 3 | 0 |
| Vùng sinh dục-hậu môn | Dương vật | Quy đầu |  | quy đầu | 1 | 18 | 0 |
| Vùng sinh dục-hậu môn | Dương vật | Rãnh quy đầu |  | rãnh quy đầu | 0 | 8 | 0 |
| Vùng sinh dục-hậu môn | Dương vật | Thân dương vật |  | Thân dương vật | 0 | 5 | 0 |
| Vùng sinh dục-hậu môn | Dương vật | Vành quy đầu |  | Vành quy đầu | 0 | 2 | 0 |
| Vùng sinh dục-hậu môn | Hậu môn | cạnh hậu môn |  | cạnh hậu môn | 0 | 1 | 0 |
| Vùng sinh dục-hậu môn | Hậu môn | Quanh hậu môn |  | Quanh hậu môn | 3 | 21 | 0 |
| Vùng sinh dục-hậu môn | Hậu môn | Vùng hậu môn |  | Vùng hậu môn | 0 | 2 | 0 |
| Vùng sinh dục-hậu môn | Niệu đạo | Quanh niệu đạo |  | Quanh niệu đạo | 0 | 1 | 0 |
| Vùng sinh dục-hậu môn | Tầng sinh môn |  |  | Tầng sinh môn | 0 | 3 | 0 |
| Vùng sinh dục-hậu môn | Vị trí chưa phân nhóm | Bẹn sinh dục |  | Bẹn sinh dục | 0 | 1 | 0 |
| Vùng sinh dục-hậu môn | Vị trí chưa phân nhóm | hậu môn |  | hậu môn | 0 | 2 | 0 |
| Vùng sinh dục-hậu môn | Vị trí chưa phân nhóm | Quanh rãnh quy đầu |  | Quanh rãnh quy đầu | 0 | 1 | 0 |
| Vùng sinh dục-hậu môn | Vị trí chưa phân nhóm | Quanh sinh dục |  | Quanh sinh dục | 0 | 2 | 0 |
| Vùng sinh dục-hậu môn | Vị trí chưa phân nhóm | Quanh vành quy đầu |  | Quanh vành quy đầu | 0 | 1 | 0 |
| Vùng sinh dục-hậu môn | Vị trí chưa phân nhóm | Quy đầu dương vật |  | Quy đầu dương vật | 0 | 1 | 0 |
| Vùng sinh dục-hậu môn | Vị trí chưa phân nhóm | Rìa hậu môn |  | Rìa hậu môn | 1 | 0 | 0 |
| Vùng sinh dục-hậu môn | Vị trí chưa phân nhóm | Sinh dục |  | Sinh dục | 1 | 29 | 0 |
| Vùng sinh dục-hậu môn | Vị trí chưa phân nhóm | Sinh dục ngoài |  | Sinh dục ngoài | 1 | 0 | 0 |
| Vùng sinh dục-hậu môn | Vị trí chưa phân nhóm | Tiền đình âm hộ |  | Tiền đình âm hộ | 1 | 0 | 0 |
| Vùng sinh dục-hậu môn | Âm hộ |  |  | Âm hộ | 4 | 14 | 0 |
| Vùng sinh dục-hậu môn | Âm hộ | Môi bé |  | Môi bé | 0 | 1 | 0 |
| Vùng sinh dục-hậu môn | Âm hộ | Môi lớn |  | Môi lớn | 0 | 1 | 0 |
| Vùng sinh dục-hậu môn | Âm đạo | Tiền đình âm đạo |  | Tiền đình âm đạo | 0 | 1 | 0 |
| Vị trí cơ thể chưa phân định | Mỏm cụt |  |  | Mỏm cụt | 0 | 4 | 0 |
| Vị trí cơ thể chưa phân định | Quanh khớp |  |  | Quanh khớp | 0 | 4 | 0 |
| Đầu |  |  |  | Đầu | 1 | 2 | 0 |
| Đầu | Da đầu |  |  | Da đầu | 21 | 156 | 24 |
| Đầu | Da đầu | Da đầu trước |  | Da đầu trước | 0 | 3 | 0 |
| Đầu | Da đầu | Da đầu vùng trán |  | Da đầu trán | 0 | 3 | 0 |
| Đầu | Da đầu | Thái dương |  | Thái dương | 10 | 50 | 0 |
| Đầu | Da đầu | Vùng chẩm |  | Chẩm | 1 | 6 | 0 |
| Đầu | Da đầu | Vùng chẩm |  | Da đầu vùng chẩm | 0 | 3 | 0 |
| Đầu | Da đầu | Vùng da đầu có tóc |  | Vùng có tóc | 0 | 2 | 0 |
| Đầu | Da đầu | Đường chân tóc |  | chân tóc | 1 | 5 | 0 |
| Đầu | Da đầu | Đường chân tóc |  | Chân tóc trán | 0 | 1 | 0 |
| Đầu | Da đầu | Đường chân tóc |  | Gần đường chân tóc | 0 | 2 | 0 |
| Đầu | Da đầu | Đường chân tóc |  | Quanh chân tóc | 0 | 2 | 0 |
| Đầu | Da đầu | Đường chân tóc |  | Rìa chân tóc | 0 | 1 | 0 |
| Đầu | Da đầu | Đường chân tóc |  | Sát chân tóc | 0 | 3 | 0 |
| Đầu | Da đầu | Đường chân tóc |  | Sát đường chân tóc | 0 | 1 | 0 |
| Đầu | Da đầu | Đường chân tóc |  | Vùng chân tóc | 1 | 3 | 0 |
| Đầu | Da đầu | Đường chân tóc |  | đường chân tóc | 4 | 22 | 0 |
| Đầu | Da đầu | Đường rẽ tóc |  | đường rẽ tóc | 0 | 1 | 0 |
| Đầu | Da đầu | Đỉnh đầu |  | Đỉnh đầu | 2 | 23 | 0 |
| Đầu | Mắt | Củng mạc |  | Củng mạc | 0 | 1 | 0 |
| Đầu | Mắt | Kết mạc |  | Kết mạc | 1 | 1 | 0 |
| Đầu | Mắt | Kết mạc mắt |  | Kết mạc mắt | 0 | 1 | 0 |
| Đầu | Mặt |  |  | Mặt | 16 | 161 | 64 |
| Đầu | Mặt | Cằm |  | Cằm | 11 | 112 | 0 |
| Đầu | Mặt | Gò má |  | gò má | 2 | 6 | 0 |
| Đầu | Mặt | Hàm |  | Hàm | 4 | 17 | 0 |
| Đầu | Mặt | Má |  | Má | 35 | 413 | 0 |
| Đầu | Mặt | Má hai bên |  | Hai má | 0 | 25 | 0 |
| Đầu | Mặt | Má trên |  | má trên | 2 | 3 | 0 |
| Đầu | Mặt | Mũi |  | mũi | 16 | 130 | 0 |
| Đầu | Mặt | Mũi | Chóp mũi | Chóp mũi | 0 | 9 | 0 |
| Đầu | Mặt | Mũi | Cánh mũi | cánh mũi | 1 | 18 | 0 |
| Đầu | Mặt | Mũi | Cạnh mũi | Cạnh mũi | 0 | 10 | 0 |
| Đầu | Mặt | Mũi | Quanh mũi | Quanh mũi | 3 | 14 | 0 |
| Đầu | Mặt | Mũi | Sống mũi | Sống mũi | 1 | 23 | 0 |
| Đầu | Mặt | Mũi | Đầu mũi | Đầu mũi | 1 | 6 | 0 |
| Đầu | Mặt | Quanh miệng | Bờ môi | Bờ môi | 0 | 1 | 0 |
| Đầu | Mặt | Quanh miệng | Cạnh môi trên | Cạnh môi trên | 0 | 1 | 0 |
| Đầu | Mặt | Quanh miệng | Cằm dưới môi | Cằm dưới môi | 0 | 1 | 0 |
| Đầu | Mặt | Quanh miệng | Dưới môi dưới | Dưới môi dưới | 0 | 1 | 0 |
| Đầu | Mặt | Quanh miệng | Giữa môi | Giữa môi | 0 | 2 | 0 |
| Đầu | Mặt | Quanh miệng | Khóe miệng | khóe miệng | 7 | 37 | 0 |
| Đầu | Mặt | Quanh miệng | Khóe miệng | mép miệng | 2 | 3 | 0 |
| Đầu | Mặt | Quanh miệng | khóe miệng phải | khóe miệng phải | 0 | 1 | 0 |
| Đầu | Mặt | Quanh miệng | Mép môi | Mép môi | 0 | 3 | 0 |
| Đầu | Mặt | Quanh miệng | Mép môi trái | Mép môi trái | 0 | 1 | 0 |
| Đầu | Mặt | Quanh miệng | Mép môi trên | Mép môi trên | 0 | 1 | 0 |
| Đầu | Mặt | Quanh miệng | Môi âm hộ | Môi âm hộ | 0 | 1 | 0 |
| Đầu | Mặt | Quanh miệng | ngang môi | ngang môi | 0 | 1 | 0 |
| Đầu | Mặt | Quanh miệng | Quanh miệng | Quanh miệng | 10 | 135 | 0 |
| Đầu | Mặt | Quanh miệng | Quanh mép miệng | Quanh mép miệng | 0 | 2 | 0 |
| Đầu | Mặt | Quanh miệng | Quanh môi | Quanh môi | 1 | 10 | 0 |
| Đầu | Mặt | Quanh miệng | Quanh môi trên | Quanh môi trên | 0 | 1 | 0 |
| Đầu | Mặt | Quanh miệng | Ria mép | Ria mép | 0 | 2 | 0 |
| Đầu | Mặt | Quanh miệng | Rãnh liên môi | Rãnh liên môi | 0 | 1 | 0 |
| Đầu | Mặt | Quanh miệng | Rãnh mũi má | Rãnh mũi má | 2 | 12 | 0 |
| Đầu | Mặt | Quanh miệng | Theo viền môi | Theo viền môi | 0 | 1 | 0 |
| Đầu | Mặt | Quanh miệng | Viền môi | Viền môi | 0 | 2 | 0 |
| Đầu | Mặt | Quanh miệng | Đường viền môi | Đường viền môi | 0 | 1 | 0 |
| Đầu | Mặt | Quanh mắt |  | Quanh mắt | 14 | 99 | 0 |
| Đầu | Mặt | Quanh mắt | Bờ mi | Bờ mi mắt | 1 | 0 | 0 |
| Đầu | Mặt | Quanh mắt | Bờ mi | Bờ mi trên | 0 | 2 | 0 |
| Đầu | Mặt | Quanh mắt | Bờ mi | Quanh bờ mi | 0 | 1 | 0 |
| Đầu | Mặt | Quanh mắt | Cung mày | Cung mày | 0 | 2 | 0 |
| Đầu | Mặt | Quanh mắt | Dưới lông mày | Dưới lông mày | 0 | 1 | 0 |
| Đầu | Mặt | Quanh mắt | Gian mày | Gian mày | 1 | 2 | 0 |
| Đầu | Mặt | Quanh mắt | Góc mắt trong | Góc mắt trong | 2 | 11 | 0 |
| Đầu | Mặt | Quanh mắt | Khóe mắt | Khóe mắt | 0 | 6 | 0 |
| Đầu | Mặt | Quanh mắt | Lông mi | Chân lông mi | 0 | 1 | 0 |
| Đầu | Mặt | Quanh mắt | Lông mi | Lông mi | 0 | 1 | 0 |
| Đầu | Mặt | Quanh mắt | Lông mi | Quanh lông mi | 0 | 1 | 0 |
| Đầu | Mặt | Quanh mắt | Mi mắt | Mi mắt | 5 | 35 | 0 |
| Đầu | Mặt | Quanh mắt | Mi mắt dưới | Mi mắt dưới | 5 | 20 | 0 |
| Đầu | Mặt | Quanh mắt | Mi mắt trên | Mi mắt trên | 6 | 38 | 0 |
| Đầu | Mặt | Quanh mắt | Mi trên trong | Mi trên trong | 0 | 2 | 0 |
| Đầu | Mặt | Quanh mắt | Quanh lông mày | Quanh lông mày | 0 | 4 | 0 |
| Đầu | Mặt | Trán |  | Trán | 19 | 109 | 0 |
| Đầu | Mặt | Vùng dưới mắt |  | Má dưới mắt | 0 | 2 | 0 |
| Đầu | Mặt | Vùng quanh mắt | Lông mày | Lông mày | 0 | 8 | 0 |
| Đầu | Mặt | Vùng râu |  | Vùng ria mép | 0 | 4 | 0 |
| Đầu | Mặt | Vùng râu |  | Vùng râu | 4 | 10 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Bờ vành tai | Bờ vành tai | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | cạnh mắt | cạnh mắt | 1 | 0 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Da đầu sau tai | Da đầu sau tai | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Da đầu thái dương | Da đầu thái dương | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Da đầu vùng trán | Da đầu vùng trán | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Dưới cằm | Dưới cằm | 0 | 2 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Dưới dái tai | Dưới dái tai | 0 | 2 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Dưới hàm | Dưới hàm | 0 | 2 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Dưới mi mắt | Dưới mi mắt | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Dưới mũi | Dưới mũi | 1 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Dưới mắt | Dưới mắt | 0 | 9 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Dưới tai | Dưới tai | 1 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Giữa mặt | Giữa mặt | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Góc hàm | Góc hàm | 0 | 6 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Góc ngoài mắt | Góc ngoài mắt | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Góc trong mắt | Góc trong mắt | 0 | 2 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Gần khóe mắt | Gần khóe mắt | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Hai mắt | Hai mắt | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | hai mặt áp sát | hai mặt áp sát | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Hàm dưới | Hàm dưới | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Hàm trên trước | Hàm trên trước | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Hố loa tai | Hố loa tai | 0 | 2 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | hố tai | hố tai | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | khóe mi mắt | khóe mi mắt | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | khóe mắt ngoài | khóe mắt ngoài | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Khóe mắt trong | Khóe mắt trong | 0 | 2 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mi mắt ngoài | Mi mắt ngoài | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | má bên | má bên | 0 | 3 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Má cạnh mũi | Má cạnh mũi | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | má dưới | má dưới | 0 | 3 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | má phải | má phải | 2 | 0 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | má thái dương | má thái dương | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Má trong | Má trong | 0 | 3 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | má trẻ nhỏ | má trẻ nhỏ | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mũi má | Mũi má | 0 | 4 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mặt bên bàn chân | Mặt bên bàn chân | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | mặt bên cổ | mặt bên cổ | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mặt bên cổ chân | Mặt bên cổ chân | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mặt bên ngón | Mặt bên ngón | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mặt bên ngón tay | Mặt bên ngón tay | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mặt dưới vú | Mặt dưới vú | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | mặt gan | mặt gan | 0 | 2 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mặt gan ngón | Mặt gan ngón | 1 | 0 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mặt gan tay | Mặt gan tay | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mặt lòng | Mặt lòng | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mặt lòng bàn chân | Mặt lòng bàn chân | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mặt lòng ngón | Mặt lòng ngón | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mặt lưng | Mặt lưng | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mặt lưng bên | Mặt lưng bên | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mặt lưng ngón | Mặt lưng ngón | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | mặt mu | mặt mu | 1 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mặt trước cẳng chân | Mặt trước cẳng chân | 0 | 2 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mặt trước cổ | Mặt trước cổ | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mặt trước gối | Mặt trước gối | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | mặt trước ngoài | mặt trước ngoài | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Mặt trước đùi | Mặt trước đùi | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Nếp gấp tai | Nếp gấp tai | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Nếp kẽ sau tai | Nếp kẽ sau tai | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Nếp mũi má | Nếp mũi má | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Nếp sau tai | Nếp sau tai | 0 | 2 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | nếp tai | nếp tai | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Quai hàm | Quai hàm | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | quanh khóe mắt trong | quanh khóe mắt trong | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Quanh lỗ mũi | Quanh lỗ mũi | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Quanh mi mắt | Quanh mi mắt | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Quanh mắt dưới | Quanh mắt dưới | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Sau dái tai | Sau dái tai | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Tiền đình mũi | Tiền đình mũi | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | trán bên | trán bên | 1 | 0 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Trán thái dương | Trán thái dương | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Viền hàm | Viền hàm | 1 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Vành mũi | Vành mũi | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Vùng da mặt | Vùng da mặt | 1 | 2 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Vùng dưới mũi | Vùng dưới mũi | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Vùng dưới mắt | Vùng dưới mắt | 0 | 2 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Vùng gò má | Vùng gò má | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Vùng hàm | Vùng hàm | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Vùng lông mày | Vùng lông mày | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Vùng má hàm | Vùng má hàm | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Vùng mũi | Vùng mũi | 0 | 2 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Vùng quanh lông mày | Vùng quanh lông mày | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | vùng quanh tai | vùng quanh tai | 1 | 0 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Vùng râu cằm | Vùng râu cằm | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Vùng tai | Vùng tai | 1 | 0 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Vùng trán bên | Vùng trán bên | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Vùng trán thái dương | Vùng trán thái dương | 0 | 1 | 0 |
| Đầu | Mặt | Vị trí mặt chưa phân nhóm | Đuôi mắt | Đuôi mắt | 0 | 1 | 0 |
| Đầu | Mặt | Đường hàm |  | Đường hàm | 0 | 3 | 0 |
| Đầu | Tai |  |  | Tai | 2 | 14 | 0 |
| Đầu | Tai | Dái tai |  | Dái tai | 3 | 13 | 0 |
| Đầu | Tai | Quanh tai |  | Quanh tai | 2 | 17 | 0 |
| Đầu | Tai | Sau tai |  | Sau tai | 2 | 17 | 0 |
| Đầu | Tai | Trước tai |  | Trước tai | 4 | 16 | 0 |
| Đầu | Tai | Vành tai |  | Loa tai | 0 | 3 | 0 |
| Đầu | Tai | Vành tai |  | Vành tai | 0 | 24 | 0 |

### Cây khoang miệng/niêm mạc

Bài báo tách riêng cây khoang miệng; các nhãn niêm mạc trong DermNet cũng được để riêng.

| Level 1 | Level 2 | Level 3 | Level 4 | Nhãn trong dữ liệu | QA dương Val | QA dương Test | Judgement phủ định Test |
|---|---|---|---|---|---:|---:|---:|
| Hầu họng | Hầu họng |  |  | Hầu họng | 0 | 1 | 0 |
| Khoang miệng | Khẩu cái | Khẩu cái |  | Khẩu cái | 2 | 5 | 0 |
| Khoang miệng | Khẩu cái | Khẩu cái cứng |  | Khẩu cái cứng | 0 | 1 | 0 |
| Khoang miệng | Khẩu cái | Khẩu cái mềm |  | Khẩu cái mềm | 0 | 2 | 0 |
| Khoang miệng | Khẩu cái | Vòm khẩu cái |  | Vòm khẩu cái | 1 | 0 | 0 |
| Khoang miệng | Khẩu cái | Vòm miệng |  | Vòm miệng | 0 | 3 | 0 |
| Khoang miệng | Lưỡi | Bờ bên lưỡi |  | Bờ bên lưỡi | 0 | 3 | 0 |
| Khoang miệng | Lưỡi | Giữa lưỡi |  | Giữa lưỡi | 0 | 3 | 0 |
| Khoang miệng | Lưỡi | Gần gốc lưỡi |  | Gần gốc lưỡi | 0 | 1 | 0 |
| Khoang miệng | Lưỡi | Gốc lưỡi |  | Gốc lưỡi | 0 | 1 | 0 |
| Khoang miệng | Lưỡi | Lưng lưỡi |  | Lưng lưỡi | 0 | 1 | 0 |
| Khoang miệng | Lưỡi | Lưỡi |  | Lưỡi | 0 | 10 | 0 |
| Khoang miệng | Lưỡi | Mặt dưới lưỡi |  | Mặt dưới lưỡi | 0 | 3 | 0 |
| Khoang miệng | Lưỡi | Mặt lưng lưỡi |  | Mặt lưng lưỡi | 0 | 25 | 0 |
| Khoang miệng | Lưỡi | Rìa lưỡi |  | Rìa lưỡi | 0 | 1 | 0 |
| Khoang miệng | Lưỡi | Sau lưỡi |  | Sau lưỡi | 0 | 1 | 0 |
| Khoang miệng | Lưỡi | Đầu lưỡi |  | Đầu lưỡi | 0 | 3 | 0 |
| Khoang miệng | Lợi |  |  | lợi | 0 | 1 | 0 |
| Khoang miệng | Lợi | Lợi hàm dưới |  | Lợi hàm dưới | 0 | 2 | 0 |
| Khoang miệng | Lợi | Lợi hàm trên |  | Lợi hàm trên | 0 | 1 | 0 |
| Khoang miệng | Lợi | Lợi răng trước |  | Lợi răng trước | 0 | 1 | 0 |
| Khoang miệng | Lợi | Lợi trên |  | Lợi trên | 0 | 1 | 0 |
| Khoang miệng | Lợi | Lợi viền |  | Lợi viền | 0 | 1 | 0 |
| Khoang miệng | Môi | Môi |  | Môi | 1 | 25 | 0 |
| Khoang miệng | Môi | Môi dưới |  | Môi dưới | 5 | 89 | 0 |
| Khoang miệng | Môi | Môi trên |  | Môi trên | 6 | 55 | 0 |
| Khoang miệng | Môi | Mặt trong môi |  | Mặt trong môi | 1 | 4 | 0 |
| Khoang miệng | Môi | Niêm mạc môi |  | Niêm mạc môi | 0 | 15 | 0 |
| Khoang miệng | Môi | Niêm mạc môi dưới |  | Niêm mạc môi dưới | 1 | 1 | 0 |
| Khoang miệng | Niêm mạc miệng | Khoang miệng |  | Khoang miệng | 0 | 4 | 0 |
| Khoang miệng | Niêm mạc miệng | Niêm mạc miệng |  | Niêm mạc miệng | 1 | 13 | 0 |
| Khoang miệng | Niêm mạc miệng | Niêm mạc má | Niêm mạc má | Niêm mạc má | 0 | 6 | 0 |
| Khoang miệng | Răng | bờ răng |  | bờ răng | 0 | 1 | 0 |
| Khoang miệng | Răng | quanh cổ răng |  | quanh cổ răng | 0 | 2 | 0 |
| Khoang miệng | Răng | Vùng răng cửa |  | Vùng răng cửa | 0 | 1 | 0 |
| Khoang miệng | Sàn miệng | Sàn miệng |  | Sàn miệng | 0 | 1 | 0 |

## Danh sách giá trị theo nhóm

Số liệu tần suất trong file TSV đếm số dòng QA có nhãn đúng hoặc lựa chọn trong câu hỏi; đáp án `Judgement=Không` được ghi riêng là nhãn được hỏi phủ định, không tính là bằng chứng dương.

| Nhóm từ vựng | Số bản ghi | Ví dụ nhãn có nhiều QA dương nhất |
|---|---:|---|
| Anatomical_field_other / Unmapped_or_mixed_value_from_Anatomical_Distribution | 91 | Bờ bên (6); nếp (6); Đầu xa (6); Vùng đỉnh (4); tương đối (4) |
| Attribute_value_list / Color | 145 | Đỏ hồng (2121); Hồng (710); hồng nhạt (627); Đỏ (529); đỏ nâu (528) |
| Attribute_value_list / Hair_Morphology | 86 | tóc (35); Thưa tóc (14); Tóc thưa (6); Vảy mịn (5); Lỗ chân lông giãn (3) |
| Attribute_value_list / Hair_and_Surface_Characteristics | 168 | Thưa tóc (23); Vảy mịn (19); Tóc thưa (18); Vảy tiết (16); Bề mặt bóng (14) |
| Attribute_value_list / Nail_Morphology | 42 | Dày móng (21); Biến dạng móng (20); tách móng (18); Loạn dưỡng móng (14); Đổi màu móng (14) |
| Attribute_value_list / Secondary_Change | 46 | Trợt (16); vảy tiết (15); loét (11); Loét nông (9); trợt da (9) |
| Attribute_value_list / Shape | 344 | bất quy tắc (1818); Ranh giới mờ (1035); Ranh giới rõ (976); tròn (897); Không đều (746) |
| Attribute_value_list / Surface_or_secondary_change | 1,109 | Vảy tiết (619); bong vảy (502); Vảy mịn (350); Bong vảy mịn (251); Khô da (234) |
| Attribute_value_list / Swelling | 4 | phù ‘Sưng nề (2); sưng (2); Hồng ban (1); Phù nề ngón tay (1) |
| Body_region / Body region | 493 | Má (448); Thân mình (426); Cẳng chân (354); Mu bàn tay (204); Cẳng tay (187) |
| Diagnosis_answer_list / Diagnosis | 354 | Acne vulgaris (93); Actinic keratosis (68); Lichen planus (67); Viêm da cơ địa (63); Discoid eczema (51) |
| Distribution_pattern / Annular_or_circular_arrangement | 5 | Ngoại vi thành vòng (1); Thành vòng (1); Vòng (1); Xếp thành vòng (1); Đồng tâm (1) |
| Distribution_pattern / Body_surface_orientation | 4 | Mặt trước (11); Mặt bên (9); Mặt trong (5); Mặt ngoài (2) |
| Distribution_pattern / Clustered | 14 | Thành cụm (195); Cụm (72); Thành đám (28); Tụ đám (13); Cụm nhỏ (12) |
| Distribution_pattern / Density | 3 | Dày đặc (23); Thưa (4); Thưa thớt (3) |
| Distribution_pattern / Exposure_or_contact | 3 | Vùng da hở (15); Da phơi nắng (2); Vùng phơi nắng (1) |
| Distribution_pattern / Extent | 6 | Khu trú (2084); Rải rác (1290); Lan tỏa (620); Lan rộng (16); Lan tỏa nhẹ (3) |
| Distribution_pattern / Extent_and_configuration | 7 | Hợp lưu (45); Tập trung (26); Liên tục (7); Rải rác hợp lưu (3); Rời rạc (1) |
| Distribution_pattern / Extent_and_laterality | 1 | Khu trú một bên (6) |
| Distribution_pattern / Flexural_or_extensor | 2 | Mặt duỗi (23); Mặt gấp (7) |
| Distribution_pattern / Flexural_or_linear | 1 | Dọc nếp gấp (8) |
| Distribution_pattern / Follicular_pattern | 2 | Quanh nang lông (11); Theo nang lông (8) |
| Distribution_pattern / Irregular_or_asymmetric_pattern | 1 | Không đều (8) |
| Distribution_pattern / Laterality_and_symmetry | 5 | Đối xứng (446); Hai bên (422); Một bên (76); Lệch bên (4); Đối xứng hai bên (1) |
| Distribution_pattern / Linear_or_row | 20 | Dọc (21); Thành dải (18); Dọc chi (8); Thành hàng (8); Lan dọc (5) |
| Distribution_pattern / Number | 15 | Đơn độc (450); Nhiều ổ (59); Nhiều ngón (52); Nhiều tổn thương (28); Nhiều móng (18) |
| Distribution_pattern / Periungual_pattern | 1 | Dọc nếp móng (2) |
| Distribution_pattern / Reticular_or_network_pattern | 1 | Dạng mạng lưới (1) |
| Distribution_pattern / Spatial_or_configuration_candidate | 24 | Trung tâm (48); Liên kết (7); Gần nhau (5); Lân cận (5); Ngoại vi (4) |
| Lesion_list / Primary_Lesion_Type | 104 | Sẩn (271); Sẩn đỏ (90); Dát (79); Sẩn nhỏ (77); Nốt (60) |
| Lesion_list / Primary_and_Secondary_Morphology | 103 | Sẩn (84); Trợt (54); Sẩn đỏ (38); Vảy tiết (33); Mụn nước (26) |
| Oral_region / Oral cavity | 36 | Môi dưới (94); Môi trên (61); Môi (26); Mặt lưng lưỡi (25); Niêm mạc môi (15) |
| Source_disease_label / Source disease | 381 | Acne vulgaris (1004); Actinic keratosis (543); Lichen planus (521); Atopic dermatitis (516); Basal cell carcinoma (495) |

## Phạm vi và giới hạn dữ liệu

- Danh sách `Diagnosis_answer_list` lấy từ đáp án của nhóm Diagnosis; `Source_disease_label` là nhãn bệnh nguồn gắn với ảnh. Hai danh sách được giữ riêng, không coi là đồng nghĩa và không dịch hàng loạt.
- Bài báo liệt kê các chiều thuộc tính như kích thước, màu, hình dạng, số lượng, phân bố và ranh giới. TSV DermNet có nhóm màu/hình dạng và trường đặc điểm rộng, nhưng không có cột riêng nhất quán cho kích thước, số lượng hoặc ranh giới; từ điển giữ theo trường nguồn thay vì tự suy ra các nhãn không được ghi rõ.
- Các TSV hiện không có nhóm `Lesion_Reasoning`, `Spatial_Relation` hoặc `Suggestion & Treatment`; vì vậy không tạo danh sách cho ba năng lực đó.
- Kiểm tra nhãn và cấu trúc được kế thừa từ vòng rà soát trước; không có bác sĩ thẩm định thủ công từng ảnh. Cây vị trí là bản ánh xạ ứng viên; các dòng được đánh dấu cần ánh xạ thủ công không nên dùng như quan hệ ontology đã xác nhận.
- Có 256 nhãn vị trí đang để `requires_manual_body_region_mapping`. Còn 91 giá trị trong `Anatomical_Distribution` được giữ ở nhóm chưa phân loại/hỗn hợp vì có thể là vị trí, kiểu phân bố, cấu hình hoặc dữ liệu khác. Có 0 câu Judgement chưa bóc tách được khái niệm mục tiêu; các câu này nằm trong `DermNet_lexical_unparsed_judgements_20260925.tsv`.

## File đầu ra

- `DermNet_lexical_inventory_20260925.tsv`: toàn bộ danh sách từ vựng dạng bảng, có Level 1–4, thuật ngữ, nhóm nguồn và số QA/image theo Val, Test và Test_1of3.
- `DermNet_lexical_tree_20260925.json`: metadata, cây vị trí cơ thể, cây khoang miệng tách riêng và bản ghi từ vựng đầy đủ cho xử lý bằng code.
- `DermNet_lexical_unparsed_judgements_20260925.tsv`: câu hỏi Có/Không chưa trích được thuật ngữ mục tiêu.

## Kiểm tra phạm vi

| Split | QA | Ảnh |
|---|---:|---:|
| Val_4k | 2,721 | 2,387 |
| Test_1of3 | 7,891 | 1,827 |
| Test | 23,681 | 5,483 |
