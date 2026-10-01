# DermNet VQA — cây vị trí và danh sách thuộc tính

Bản đọc nhanh theo bố cục Table 3 và Table 7 của bài báo. Nhãn lấy từ lexical inventory tạo từ dữ liệu DermNet Val_4k và Test; bảng sắp xếp lại để dễ xem, không tự thêm đặc điểm y khoa.

Nguồn bố cục: [MedLesionVQA, ICLR 2026, Supplementary Tables 3 and 7](https://proceedings.iclr.cc/paper_files/paper/2026/file/d82c24b7a4237aa4283b38e12047dc38-Paper-Conference.pdf).

## Table 3 — Cây vị trí cơ thể (Level 1–4)

Cây chính gồm 237 nhãn có status curated_body_region_path trong inventory. Mỗi hàng là một nhánh Level 1–Level 2; cột Level 3 liệt kê nhánh con, cột Level 4 ghi quan hệ nhãn con. Khi nhãn chuẩn khác chữ ở đường dẫn, ghi dạng đường dẫn (nhãn chuẩn: ...).

Status curated_body_region_path là trạng thái ánh xạ trong pipeline, không đồng nghĩa bác sĩ đã duyệt từng nhãn.

| Level 1 | Level 2 | Level 3 | Level 4 / nhãn con |
|---|---|---|---|
| Đầu | — | — | — |
| Đầu | Da đầu | Da đầu trước · Da đầu vùng trán · Thái dương · Vùng chẩm · Vùng da đầu có tóc · Đường chân tóc · Đường rẽ tóc · Đỉnh đầu | Da đầu vùng trán → Da đầu trán · Vùng chẩm → Chẩm · Vùng chẩm → Da đầu vùng chẩm · Vùng da đầu có tóc → Vùng có tóc · Đường chân tóc → chân tóc<br>Đường chân tóc → Chân tóc trán · Đường chân tóc → Gần đường chân tóc · Đường chân tóc → Quanh chân tóc · Đường chân tóc → Rìa chân tóc · Đường chân tóc → Sát chân tóc<br>Đường chân tóc → Sát đường chân tóc · Đường chân tóc → Vùng chân tóc · Đường chân tóc → đường chân tóc · Đường rẽ tóc → đường rẽ tóc |
| Đầu | Mắt | Củng mạc · Kết mạc · Kết mạc mắt | — |
| Đầu | Mặt | Cằm · Gò má · Hàm · Má · Má hai bên · Má trên · Mũi · Quanh miệng · Quanh mắt · Trán<br>Vùng dưới mắt · Vùng quanh mắt · Vùng râu · Đường hàm | Gò má → gò má · Má hai bên → Hai má · Má trên → má trên · Mũi → Chóp mũi · Mũi → Cánh mũi (nhãn chuẩn: cánh mũi)<br>Mũi → Cạnh mũi · Mũi → mũi · Mũi → Quanh mũi · Mũi → Sống mũi · Mũi → Đầu mũi<br>Quanh miệng → Bờ môi · Quanh miệng → Cạnh môi trên · Quanh miệng → Cằm dưới môi · Quanh miệng → Dưới môi dưới · Quanh miệng → Giữa môi<br>Quanh miệng → Khóe miệng (nhãn chuẩn: khóe miệng) · Quanh miệng → Khóe miệng (nhãn chuẩn: mép miệng) · Quanh miệng → khóe miệng phải · Quanh miệng → Mép môi · Quanh miệng → Mép môi trái<br>Quanh miệng → Mép môi trên · Quanh miệng → Môi âm hộ · Quanh miệng → ngang môi · Quanh miệng → Quanh miệng · Quanh miệng → Quanh mép miệng<br>Quanh miệng → Quanh môi · Quanh miệng → Quanh môi trên · Quanh miệng → Ria mép · Quanh miệng → Rãnh liên môi · Quanh miệng → Rãnh mũi má<br>Quanh miệng → Theo viền môi · Quanh miệng → Viền môi · Quanh miệng → Đường viền môi · Quanh mắt → Bờ mi (nhãn chuẩn: Bờ mi mắt) · Quanh mắt → Bờ mi (nhãn chuẩn: Bờ mi trên)<br>Quanh mắt → Bờ mi (nhãn chuẩn: Quanh bờ mi) · Quanh mắt → Cung mày · Quanh mắt → Dưới lông mày · Quanh mắt → Gian mày · Quanh mắt → Góc mắt trong<br>Quanh mắt → Khóe mắt · Quanh mắt → Lông mi · Quanh mắt → Lông mi (nhãn chuẩn: Chân lông mi) · Quanh mắt → Lông mi (nhãn chuẩn: Quanh lông mi) · Quanh mắt → Mi mắt<br>Quanh mắt → Mi mắt dưới · Quanh mắt → Mi mắt trên · Quanh mắt → Mi trên trong · Quanh mắt → Quanh lông mày · Vùng dưới mắt → Má dưới mắt<br>Vùng quanh mắt → Lông mày · Vùng râu → Vùng ria mép |
| Đầu | Tai | Dái tai · Quanh tai · Sau tai · Trước tai · Vành tai | Vành tai → Loa tai |
| Cổ | — | — | — |
| Cổ | Cổ bên | — | — |
| Cổ | Cổ sau | — | — |
| Cổ | Cổ trước | — | — |
| Cổ | Gáy | — | Gáy → Cổ gáy |
| Thân mình | — | — | — |
| Thân mình | Bẹn | Nếp bẹn | — |
| Thân mình | Bụng | bụng bên · Bụng dưới · Bụng trên · Quanh rốn | — |
| Thân mình | Hông | — | — |
| Thân mình | Lưng | Lưng dưới · Lưng trên | — |
| Thân mình | Mạn sườn | — | — |
| Thân mình | Ngực | Hõm ức · Ngực bên · Ngực trên · Ngực trước · Quanh xương ức · Vùng thượng đòn · Vú · Xương ức | Ngực trước → ngực trước · Vùng thượng đòn → thượng đòn · Vú → Bề mặt vú · Vú → Núm vú (nhãn chuẩn: núm vú) · Vú → Nếp dưới vú<br>Vú → Quanh núm vú · Vú → Quanh quầng vú · Vú → Quầng vú · Vú → vú |
| Thân mình | Nách | Hố nách | — |
| Thân mình | Thân dưới | — | — |
| Thân mình | Thân mình bên | — | — |
| Thân mình | thân mình sau | — | — |
| Thân mình | Thân mình trước | — | — |
| Thân mình | Thân trên | — | — |
| Thân mình | Vùng chậu | Vùng chậu · Vùng cùng cụt | — |
| Thân mình | Vùng tã | — | — |
| Chi trên | — | — | — |
| Chi trên | Bàn tay | Hai bàn tay · Lòng bàn tay · Mu bàn tay · Mu hai bàn tay · Mô cái · Ngón tay | Bàn tay → bàn tay · Mô cái → Vùng mô cái · Ngón tay → Khớp ngón tay · Ngón tay → Kẽ ngón tay · Ngón tay → Mu ngón tay<br>Ngón tay → Móng tay · Ngón tay → Móng tay (nhãn chuẩn: Móng ngón tay) · Ngón tay → ngón tay · Ngón tay → Ngón tay cái · Ngón tay → Đầu ngón tay |
| Chi trên | Cánh tay | Cánh tay trên · Mặt ngoài cánh tay · Mặt trong cánh tay | — |
| Chi trên | Cẳng tay | — | — |
| Chi trên | Cổ tay | — | — |
| Chi trên | Khuỷu tay | Nếp khuỷu | Nếp khuỷu → Gấp khuỷu tay · Nếp khuỷu → Nếp gấp khuỷu tay · Nếp khuỷu → Nếp khuỷu tay |
| Chi trên | Vai | — | — |
| Chi dưới | — | — | — |
| Chi dưới | Bàn chân | Gót chân · Lòng bàn chân · Mu bàn chân · Ngón chân · Vòm bàn chân | Bàn chân → bàn chân · Gót chân → Vùng gót · Gót chân → Vùng gót chân · Ngón chân → Kẽ ngón chân (nhãn chuẩn: kẽ ngón chân) · Ngón chân → Móng chân<br>Ngón chân → Móng chân (nhãn chuẩn: Móng chân cái) · Ngón chân → Móng chân (nhãn chuẩn: Móng ngón chân cái) · Ngón chân → Móng chân (nhãn chuẩn: Móng ngón chân) · Ngón chân → Móng chân (nhãn chuẩn: Quanh móng chân) · Ngón chân → Móng chân (nhãn chuẩn: Rìa móng chân)<br>Ngón chân → Ngón chân cái · Ngón chân → Đầu ngón chân |
| Chi dưới | Cẳng chân | Cẳng chân dưới | — |
| Chi dưới | Cổ chân | Gân gót · Mắt cá · Quanh cổ chân · Vùng cổ chân | Gân gót → Vùng gân gót · Mắt cá → Mắt cá chân · Mắt cá → quanh mắt cá |
| Chi dưới | Gối | Hai gối · Nếp khoeo · Quanh gối · Vùng gối | Nếp khoeo → Khoeo chân |
| Chi dưới | Mông | Nếp liên mông | — |
| Chi dưới | Đùi | Mặt sau đùi · Mặt trong đùi | — |
| Vùng sinh dục-hậu môn | Bìu | — | — |
| Vùng sinh dục-hậu môn | Dương vật | bao quy đầu · Quy đầu · Rãnh quy đầu · Thân dương vật · Vành quy đầu | Quy đầu → quy đầu · Rãnh quy đầu → rãnh quy đầu |
| Vùng sinh dục-hậu môn | Hậu môn | cạnh hậu môn · Quanh hậu môn · Vùng hậu môn | — |
| Vùng sinh dục-hậu môn | Niệu đạo | Quanh niệu đạo | — |
| Vùng sinh dục-hậu môn | Tầng sinh môn | — | — |
| Vùng sinh dục-hậu môn | Âm hộ | Môi bé · Môi lớn | — |
| Vùng sinh dục-hậu môn | Âm đạo | Tiền đình âm đạo | — |
| Da | Nếp gấp | — | — |
| Da | Nếp gấp/kẽ | kẽ ngón · Kẽ ngón cái · Kẽ sinh dục · Mép nếp gấp · Nếp bẹn trái · nếp bụng · nếp da · Nếp dưới bụng · Nếp gấp bẹn · Nếp gấp bụng<br>Nếp gấp chi · Nếp gấp nách · Nếp kẽ · Nếp kẽ hậu môn · Nếp kẽ quanh sinh dục · Nếp nách · Quanh nếp gấp · Theo kẽ ngón · Theo nếp gấp · Vùng nếp bẹn | — |
| Da | Vùng da có lông | — | Vùng da có lông → Da có lông |

## Nhãn vị trí chờ xác nhận ánh xạ

Có 256 nhãn đang mang status requires_manual_body_region_mapping. Tôi để chúng ngoài cây chính để không biến đường dẫn tạm thành vị trí giải phẫu đã xác nhận.

| Nhãn chờ xác nhận |
|---|
| Bàn tay phải · Bên cổ · Bên ngón · Bản móng · Bẹn sinh dục · Bờ bàn chân · Bờ bàn tay · Bờ bên móng<br>Bờ móng bên · Bờ ngoài bàn chân · Bờ ngón tay · Bờ trong bàn chân · Bờ vành tai · Chi · Chi có lông · Các ngón<br>Các ngón tay · Cánh tay ngoài · Cánh tay trong · cạnh mắt · Cổ trước-ngực trên · Cổ vai lưng · Cổ-ngực · Da<br>Da chi lông · Da lòng bàn tay · Da vùng chi · Da đầu hói · Da đầu sau tai · Da đầu thái dương · Da đầu vùng trán · Dưới cằm<br>Dưới dái tai · Dưới hàm · Dưới mi mắt · Dưới móng · Dưới mũi · Dưới mắt · Dưới ngón chân · Dưới tai<br>gan bàn tay · Gan ngón · Gan ngón chân · Giữa bụng · Giữa mặt · Giữa ngực · Gáy sau · Góc hàm<br>Góc ngoài mắt · Góc trong mắt · Gần gốc móng · Gần khóe mắt · Gần khớp ngón cái · Gốc ngón · gốc ngón cái · Gốc ngón tay<br>hai bàn chân · Hai chân · Hai cánh tay · Hai cẳng tay · Hai khoeo · Hai khuỷu tay · Hai lòng bàn tay · Hai móng<br>Hai mắt · hai mặt áp sát · Hai ngón · Hai tay · Hai vai · Hai đùi · Hàm dưới · Hàm trên trước<br>Hông bên · hông đùi · hậu môn · Hố loa tai · hố tai · Khe liên mông · Khe mông · Khoeo<br>khuỷu · khóe mi mắt · khóe mắt ngoài · Khóe mắt trong · khớp ngón · Kẽ liên mông · Kẽ mông · Lưng bên<br>Lưng giữa · Lồng ngực · Mi mắt ngoài · Mu khớp ngón · Mu ngón · Mu đốt ngón · má bên · Má cạnh mũi<br>má dưới · má phải · má thái dương · Má trong · má trẻ nhỏ · Mép niêm mạc · Móng · Móng ngón cái<br>Mông bẹn · Mông trên · Mũi má · Mặt bên bàn chân · mặt bên cổ · Mặt bên cổ chân · Mặt bên ngón · Mặt bên ngón tay<br>Mặt dưới vú · mặt gan · Mặt gan ngón · Mặt gan tay · Mặt lòng · Mặt lòng bàn chân · Mặt lòng ngón · Mặt lưng<br>Mặt lưng bên · Mặt lưng ngón · mặt mu · Mặt trước cẳng chân · Mặt trước cổ · Mặt trước gối · mặt trước ngoài · Mặt trước đùi<br>Mỏm cụt · Một ngón tay · Ngang thân mình · Ngón · Ngón cái · Ngón áp út · Ngực bụng · Ngực giữa<br>Ngực vai · Niêm mạc · Niêm mạc hậu môn · Niêm mạc sinh dục · Niêm mạc âm hộ · Nách bên · Nách trước · Nếp cổ<br>Nếp gấp cổ · Nếp gấp cổ chân · Nếp gấp cổ tay · Nếp gấp gối · Nếp gấp khuỷu · Nếp gấp tai · Nếp gấp tay · Nếp gấp đùi<br>Nếp kẽ mông · Nếp kẽ sau tai · Nếp móng · Nếp móng bên · Nếp móng gần · Nếp mông · Nếp mũi má · Nếp sau tai<br>nếp tai · Phiến móng · Quai hàm · Quanh bờ móng · Quanh gốc móng · quanh khóe mắt trong · Quanh khớp · quanh lỗ chân lông<br>Quanh lỗ mũi · Quanh mi mắt · Quanh móng · Quanh mắt dưới · Quanh ngón · Quanh rãnh quy đầu · Quanh sinh dục · Quanh vành quy đầu<br>Quy đầu dương vật · Rãnh liên mông · Rãnh móng bên · Rìa bàn chân · rìa bên móng · Rìa hậu môn · Rìa móng · Rìa ngón<br>Rìa ngón tay · Rìa xa móng · Sau dái tai · Sinh dục · Sinh dục ngoài · Sát móng · tay · Thành ngực<br>Thành ngực bên · Thân bên · Thắt lưng · Thắt lưng cùng · Tiền đình mũi · Tiền đình âm hộ · Toàn bộ móng · Toàn lưng<br>trán bên · Trán thái dương · Trước cẳng chân · Vai cổ · Vai gáy · Viền hàm · Vành mũi · Vòm chân<br>Vòm gan chân · Vùng bụng dưới · Vùng cánh tay · Vùng cẳng chân · Vùng cổ bên · Vùng cổ gáy · Vùng cổ ngực · Vùng cổ trước<br>Vùng da mặt · Vùng dưới mũi · Vùng dưới mắt · Vùng dưới áo ngực · Vùng giữa ngực · Vùng gáy · Vùng gò má · Vùng hàm<br>Vùng hông · Vùng khoeo · Vùng khuỷu tay · Vùng lông mày · Vùng lưng · Vùng lưng dưới · Vùng lưng trên · Vùng má hàm<br>Vùng mông · Vùng mũi · Vùng ngực trên · Vùng niêm mạc · Vùng quanh lông mày · vùng quanh tai · Vùng râu cằm · Vùng tai<br>Vùng thắt lưng · Vùng trán bên · Vùng trán thái dương · Vùng đùi · Vùng đầu gối · Đuôi mắt · Đùi háng · Đùi trong<br>Đùi trên · Đùi trước · Đầu gối · Đầu móng · Đầu ngón · Đầu xa móng · Đốt ngón · Đốt ngón tay |

## Bổ sung — Vị trí khoang miệng

Inventory có 36 nhãn khoang miệng/niêm mạc. Bài báo tách nội dung này thành bảng riêng; phần dưới giữ riêng để không nhập lẫn vào cây vị trí cơ thể.

| Level 1 | Level 2 | Level 3 | Level 4 / nhãn con |
|---|---|---|---|
| Khoang miệng | Khẩu cái | Khẩu cái · Khẩu cái cứng · Khẩu cái mềm · Vòm khẩu cái · Vòm miệng | — |
| Khoang miệng | Lưỡi | Bờ bên lưỡi · Giữa lưỡi · Gần gốc lưỡi · Gốc lưỡi · Lưng lưỡi · Lưỡi · Mặt dưới lưỡi · Mặt lưng lưỡi · Rìa lưỡi · Sau lưỡi<br>Đầu lưỡi | — |
| Khoang miệng | Lợi | Lợi hàm dưới · Lợi hàm trên · Lợi răng trước · Lợi trên · Lợi viền | Lợi → lợi |
| Khoang miệng | Môi | Môi · Môi dưới · Môi trên · Mặt trong môi · Niêm mạc môi · Niêm mạc môi dưới | — |
| Khoang miệng | Niêm mạc miệng | Khoang miệng · Niêm mạc miệng · Niêm mạc má | Niêm mạc má → Niêm mạc má |
| Khoang miệng | Răng | bờ răng · quanh cổ răng · Vùng răng cửa | — |
| Khoang miệng | Sàn miệng | Sàn miệng | — |
| Hầu họng | Hầu họng | — | — |

## Table 7 — Giá trị thuộc tính (bỏ Size)

Giữ 5 nhóm còn lại trong Table 7: Color, Shape, Quantity, Distribution và Boundary. Danh sách là các nhãn đang có trong inventory.

| Thuộc tính | Danh sách giá trị |
|---|---|
| **Color** (145) | Ban đỏ đầu ngón · Bình thường · Bạc · Cam · Da · Da bình thường · Da cam · da hồng nhạt<br>Da màu · Da màu hồng · Da nhạt màu · Da nâu · Da sẫm · Da sẫm màu · da đầu · Giảm sắc tố<br>Hơi đỏ · Hồng · Hồng ban · Hồng bóng · Hồng cam · Hồng cam nhạt · Hồng da · Hồng ngọc<br>hồng nhạt · Hồng nâu · Hồng trắng · Hồng tím · Hồng tím nhạt · Không đồng nhất · màu da · Màu da bình thường<br>Màu da nhạt · Mất sắc tố · Nhạt màu · Nâu · Nâu cam · Nâu da · Nâu hồng · Nâu hồng nhạt<br>nâu hổ phách · Nâu nhạt · Nâu nền · Nâu sẫm · Nâu tím · Nâu xám · Nâu đen · Nâu đậm<br>Nâu đồng · Nâu đồng đều · Nền hồng nhẹ · Trong · Trong bóng · Trong mờ · Trung tâm nhạt · trung tâm nhạt màu<br>Trung tâm sẫm · Trung tâm tím sẫm · Trung tâm đỏ sẫm · Trắng · Trắng bóng · Trắng bạc · Trắng da · Trắng hồng<br>Trắng kem · Trắng ngà · Trắng ngọc · Trắng ngọc trai · Trắng nhạt · Trắng nhợt · Trắng sữa · Trắng trong<br>Trắng vàng · Trắng vàng nhạt · Trắng vàng trung tâm · Trắng xanh · Trắng xám · Trắng đục · Trắng ẩm · Tím<br>Tím bầm · tím hồng · tím nhạt · Tím nâu · Tím sẫm · Tím xanh · Tím xám · Tím đen<br>Tím đỏ · Tóc trắng xám · Tăng sắc tố · Vàng · Vàng cam · Vàng da · Vàng hồng · vàng kem<br>Vàng mật ong · Vàng mủ · Vàng ngà · Vàng nhạt · Vàng nâu · Vàng trong · Vàng xanh · Vàng xám<br>Vàng đục · Xanh lam · Xanh lục · Xanh lục nhạt · Xanh lục sáng · Xanh lục đậm · Xanh nâu · xanh tím<br>Xanh xám · Xanh đen · Xám · Xám nhạt · Xám nâu · Xám vàng · Xám xanh · Xám đen<br>Xám đục · Đen · Đen nâu · Đen sẫm · Đen trung tâm · Đen tím · Đen xanh · đen xám<br>Đỏ · Đỏ bóng · Đỏ cam · Đỏ hồng · Đỏ hồng nhạt · Đỏ hồng nhẹ · Đỏ nhạt · Đỏ nhẹ<br>đỏ nâu · Đỏ nâu nhạt · Đỏ quanh gốc · Đỏ sẫm · Đỏ tía · Đỏ tím · Đỏ tím nhạt · Đỏ tươi<br>Đỏ đậm |
| **Shape** (344) | Bia bắn · Bia bắn mờ · Bia đích · Bán cầu · Bản đồ · bất quy tắc · Bất đối xứng · Bầu dục<br>Bầu dục dài · Bầu dục kéo dài · Bầu dục nhỏ · Bề mặt gồ · Bề mặt gồ ghề · Bề mặt không đều · Bề mặt lồi · Bờ cong<br>bờ cung · Bờ cuộn · Bờ gồ · Bờ gồ cao · Bờ gồ nhẹ · Bờ khá rõ · Bờ không rõ · Bờ không đều<br>Bờ lượn sóng · Bờ mờ · Bờ ngoằn ngoèo · Bờ nham nhở · Bờ nông · Bờ nổi · bờ rõ · Bờ thẳng<br>Bờ tua gai · Bờ tương đối rõ · Bờ viền rõ · Bờ vòng · Bờ vòng cung · Bờ vụn · Bờ xa · Bờ xa bên<br>Bờ đa cung · Bờ đều · Chia thùy · Chuỗi hạt · Chóp nhọn · Chùm · Chùm nho · Chấm<br>Chấm nhỏ · Chấm tròn · Chữ C · Chữ nhật · Cong · Cong lồi · Cong móc · Cong ngắn<br>Cung · Cung tròn · Cuống nhỏ · Cánh bướm · Có cuống · Có vảy · Căng · cột nhỏ<br>Diện rộng · Dài · Dày da · Dạng bia · Dạng bản đồ · Dạng bầu dục · Dạng chùm · Dạng chấm<br>Dạng chấm tròn · Dạng cung · Dạng cục · Dạng củ · Dạng dài · Dạng dát · Dạng dát nhỏ · Dạng dải<br>Dạng dải dài · Dạng dọc · Dạng gai · Dạng giọt · Dạng hạt · Dạng khe · Dạng khe nứt · dạng khảm<br>Dạng kẽ · Dạng lưới · Dạng lưới mờ · Dạng múi · dạng mạng lưới · Dạng nhánh · Dạng nhú · Dạng nhẫn<br>Dạng nếp gấp · Dạng súp lơ · Dạng sừng · dạng thùy · Dạng thùy múi · Dạng vân · Dạng vòm · Dạng vòng<br>Dạng vòng cung · Dạng vệt · Dạng ô lưới · Dạng đường · Dạng đường cong · Dạng đường nứt · Dạng đường thẳng · Dải<br>Dải co kéo · Dải cong · Dải cung · dải dài · Dải dọc · Dải hình cung · dải ngang · Dải ngoại vi<br>Dải ngoằn ngoèo · Dải ngắn · Dải rộng · Dải song song · Dải theo nếp · Dải thẳng · Dẹt · Dọc<br>Dọc giữa · Dọc móng · Dọc nếp da · Dọc nếp gấp · Dọc rãnh móng · Giãn mạch · Giọt nước · Giới hạn khá rõ<br>Góc móng tù · Gần tròn · Gần vuông · Gồ · Gồ cao · Gồ cục · Gồ ghề · Gồ nhẹ<br>Gồ nổi · Gồ vòm · Hoại tử · Hình bia bắn · Hình bán cầu · Hình bán nguyệt · Hình bản đồ · Hình bầu dục<br>Hình chấm · Hình chấm nhỏ · hình chữ nhật · Hình cung · Hình cánh bướm · Hình cầu · Hình dải · Hình nhẫn<br>Hình nón · Hình que · Hình sao · Hình tam giác · Hình thoi · Hình tròn · Hình tròn nhỏ · Hình trụ<br>hình vòm · Hình vòng · Hình vòng cung · Hình vòng nhỏ · Hình đa cung · Hình đa giác · Hình đồng xu · hơi dạng vòng<br>Hơi dẹt · Hơi gồ · hơi không đều · hơi không đối xứng · Hơi thùy múi · Hơi tròn · Hẹp · Khe<br>Khe dài · Khe nhỏ · Khe nứt · Khe nứt dài · Khe nứt ngắn · khe nứt thẳng · Khép kín · khía múi<br>Không rõ · Không đều · Không đối xứng · Kéo dài · Kích thước khác nhau · Kích thước không đều · Kích thước nhỏ · Kích thước đa dạng<br>kích thước đều · Lan dài · Lan từ bờ xa · Li ti · Loang lổ · Lobul · Loét · Lõm<br>Lõm giữa · Lõm nhỏ · Lõm trung tâm · lưới · Lấm tấm · Lốm đốm · Lồi · Lồi dạng vòm<br>Lồi gồ · Lồi nhô · Lỗ giãn · Mép nham nhở · Múi · Múi hóa · Mũi thoi · Mạch phân nhánh<br>Mạng lưới · Mờ · Mờ ranh giới · Ngoằn ngoèo · Nham nhở · Nhiều thùy · Nhô cao · Nhú<br>nhú sừng · Nhẫn · Nhẵn · Nhọn · Nhỏ · Nhỏ giọt · Nhỏ li ti · Nhỏ đều<br>Nón cong · Nón sừng · Nông · Nếp gấp · Nếp gấp sâu · Nốt tròn · nổi cao · Phân nhánh<br>Phân vùng ngang · Phù nề · Phẳng nhẹ · Ranh giới · Ranh giới không rõ · Ranh giới không đều · Ranh giới mờ · Ranh giới rõ<br>Ranh giới tương đối rõ · Rãnh dọc · Rãnh gờ · Rìa nham nhở · Rất nhỏ · Song song · sùi · Sẩn vòm<br>Sẹo dải · Sợi dài · Sừng nhô cao · Tam giác · Theo hình xăm · Theo nếp gấp · Thoi · Thon dài<br>Thuôn dài · Thuỳ · Thành chùm · Thùy · Thùy múi · Thùy nhỏ · Thẳng · Trung tâm lõm<br>Trung tâm nhạt · tròn · Tròn bầu dục · Tròn không đều · Tròn nhỏ · Tròn đều · Tròn-bầu dục · Trụ cong<br>Trụ cụt · tuyến tính · U cục nhỏ · Uốn cong · uốn lượn · Viền cong · Viền gồ · Viền quanh móng<br>Viền đỏ · Vân da rõ · Vân lưới · Vân lưới mờ · Vòm · Vòm nhỏ · Vòm thấp · vòm tròn<br>vòng · Vòng cung · Vòng hở · Vòng không hoàn toàn · Vòng một phần · Vòng nhỏ · Vòng quanh móng · Vảy tiết<br>Vệt tuyến tính · Đa cung · Đa dạng · Đa dạng kích thước · Đa giác · đa giác lớn · Đa giác nhỏ · Đa hình<br>Đa kích thước · đa thùy · đa vòng · Đám không đều · Đường · Đường dài · Đường dọc · Đường mảnh<br>Đường ngoằn ngoèo · Đường nứt · Đường nứt dài · Đường nứt nẻ · Đường thẳng · Đường vân · Đầu tù · Đều nhau<br>Đốm · Đốm nhỏ · Đồng dạng · Đồng tiền · Đồng tâm · Đồng xu · Đồng đều · Ổ nhỏ |
| **Quantity** (15) | Một móng · Một tổn thương chính · Một ổ · Nhiều · Nhiều móng · Nhiều ngón · Nhiều nốt · Nhiều tổn thương<br>Nhiều vị trí · Nhiều đám · Nhiều ổ · Riêng lẻ · Đa ổ · Đơn độc · Đơn ổ |
| **Distribution** (100) | Chồng lấp · Cụm · Cụm nhỏ · Cụm thưa · Da phơi nắng · Dày đặc · Dạng dải · Dạng mạng lưới<br>Dạng tuyến · Dạng đường · Dải · Dọc · Dọc chi · Dọc khe · Dọc nếp gấp · Dọc nếp móng<br>Dọc rãnh · Dọc trục · Gần nhau · Hai bên · Hợp lưu · Khu trú · Khu trú một bên · Không đều<br>Lan dọc · Lan rộng · Lan tỏa · Lan tỏa nhẹ · Liên kết · Liên kết mảng · Liên tục · Lân cận<br>Lệch bên · Mặt bên · Mặt duỗi · Mặt gấp · Mặt ngoài · Mặt trong · Mặt trước · Một bên<br>Ngoại vi · Ngoại vi thành vòng · Nhóm · Nằm ngang · Quanh lỗ xỏ · Quanh mạch · Quanh mảng · Quanh nang lông<br>Quanh tổn thương · Rải rác · Rải rác hợp lưu · Rời rạc · Theo chiều ngang · Theo dải · Theo hàng · Theo nang lông<br>Theo sẹo · Theo đường · Theo đường rẽ tóc · Thành cụm · Thành dải · Thành hàng · Thành mảng · Thành vòng<br>Thành đám · Thưa · Thưa thớt · Toàn thân · Trung tâm · Trung tâm móng · Trung tâm mảng · Trung tâm nếp<br>Trung tâm quầng · Trung tâm ảnh · Trên nền mảng · Tuyến tính · Tách biệt · Tập trung · Tập trung cụm · Tụ cụm<br>Tụ thành chùm · Tụ thành cụm · Tụ thành đám · Tụ đám · Từng cụm · Vòng · Vòng quanh · Vùng da hở<br>Vùng phơi nắng · Vệ tinh · Xung quanh · Xếp dọc · Xếp hàng · Xếp thành hàng · Xếp thành vòng · Đám<br>Đường giữa · Đối xứng · Đối xứng hai bên · Đồng tâm |
| **Boundary** (68) | Bờ bong tróc · Bờ cong · bờ cung · Bờ cuộn · Bờ gồ · Bờ gồ cao · Bờ gồ nhẹ · Bờ gờ<br>Bờ hoạt động · Bờ hơi mờ · Bờ khá mờ · Bờ khá rõ · Bờ không rõ · Bờ không đều · Bờ loang lổ · Bờ lượn sóng<br>Bờ lồi · Bờ lởm chởm · Bờ múi · bờ mảnh · Bờ mờ · Bờ ngoằn ngoèo · Bờ nham nhở · Bờ nhòe<br>Bờ nhạt màu · Bờ nông · Bờ nổi · bờ nổi nhẹ · Bờ phẳng · Bờ quầng rõ · Bờ rõ · Bờ thẳng<br>Bờ trơn · Bờ trợt · Bờ tua gai · Bờ tím · Bờ tím đen · Bờ tăng sắc tố · Bờ tương đối rõ · Bờ viêm<br>Bờ viêm đỏ · Bờ viền rõ · Bờ viền đậm · Bờ vòng · Bờ vòng cung · Bờ vụn · Bờ xa · Bờ xa bên<br>Bờ đa cung · Bờ đậm · Bờ đều · Bờ đỏ · Bờ đỏ nhẹ · Bờ đỏ viêm · Giới hạn khá rõ · giới hạn mờ<br>Giới hạn rõ · Lan từ bờ xa · Mờ bờ · Mờ ranh giới · Ranh giới · Ranh giới khá rõ · Ranh giới không rõ · Ranh giới không đều<br>Ranh giới mờ · Ranh giới rõ · Ranh giới tương đối rõ · Vỡ bờ móng |

**Ghi chú về Boundary:** inventory chưa có trường Boundary độc lập. Danh sách trên được gom theo từ khóa “bờ”, “ranh giới” hoặc “giới hạn” trong các nhãn thuộc tính. Ba cặp nhãn chỉ khác viết hoa/thường được gộp ở bản đọc này; TSV inventory vẫn giữ nguyên. Một số nhãn có thể trùng với nhóm Shape/Characteristics và Boundary chưa được bác sĩ gán nhãn riêng.

Các nhóm thuộc tính như bề mặt, tóc, móng và sưng/phù có trong inventory đầy đủ nhưng không nhập vào Table 7 này, vì bố cục bảng đang bám theo các nhóm thuộc tính của bài báo.

## Lưu ý sử dụng

Đây là bản trình bày lexical labels theo dữ liệu hiện có, không phải ontology da liễu đã được bác sĩ xác nhận từng nhãn. Xem TSV inventory đầy đủ khi cần nguồn gốc, nhóm gốc hoặc trạng thái ánh xạ.
