from pathlib import Path
from collections import defaultdict, Counter
import json, html, re
from urllib.parse import quote

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent/'images'
def read(n):return json.loads((OUT/n).read_text())
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
S=read('summary.json'); F=read('folders.json'); I=read('inventory.json'); G=read('duplicates_pixels.json'); N=read('near_duplicate_candidates.json'); O=read('folder_overlap.json')
fi={x['folder']:x for x in F}; ii={x['path']:x for x in I}
rules=[]
def rule(names,action,target,reason,source=None):
    for n in names.split('|'):
        if n not in fi: continue
        local=OUT.parent/'contents'/f'Toàn bộ nội dung - {n}.txt'
        rules.append(dict(folder=n,action=action,target=target,reason=reason,source_url=source,local_source=str(local) if local.exists() else None))

rule('Papular mucinosis','merge_synonym','Lichen myxoedematosus','Tên đồng nghĩa; giữ nguồn và phân nhóm của từng ảnh. 2 ảnh ở nhãn nguồn không trùng pixel với 6 ảnh ở nhãn đích.','https://dermnetnz.org/topics/lichen-myxoedematosus')
rule('Pyoderma','quarantine_label_review','Cutaneous abscess (chỉ ảnh xác minh đúng áp xe)','Hai tệp contents giống hệt, nhưng Pyoderma có 76 ảnh, Cutaneous abscess có 6 và không trùng pixel giữa hai thư mục. Không thể suy ra cả 76 ảnh đều là áp xe. Bỏ nhãn Pyoderma khỏi tập đơn nhãn tạm thời; kiểm tra từng ảnh trước khi gộp.','https://dermnetnz.org/topics/cutaneous-abscess')
rule('Vulval intraepithelial neoplasia','quarantine_image_review',None,'14 ảnh pixel duy nhất trùng với 9 nhãn khác; đã thấy ảnh bàn tay, cằm/râu và tổn thương ngoài vùng âm hộ trong mẫu. Ưu tiên rà soát ảnh số thứ tự; giữ riêng ảnh VIN đã xác minh. Không xoá bệnh VIN.','https://dermnetnz.org/topics/vulval-intraepithelial-neoplasia')
rule('Non-albicans candida infections','quarantine_label_review','Candidiasis + vị trí + bằng chứng loài','36/69 ảnh pixel duy nhất trùng với 6 nhãn khác. Toàn bộ 15 ảnh của Vulvovaginal candidiasis nằm trong nhóm này. Hình lâm sàng không chứng minh được loài non-albicans; chỉ giữ nhãn loài khi có nguồn xét nghiệm. Không chuyển toàn bộ onychomycosis thành Candida.','https://dermnetnz.org/topics/non-albicans-candida-infections')
rule('Facial psoriasis|Flexural psoriasis|Genital psoriasis','merge_if_coarse','Psoriasis','Nhãn vị trí/biểu hiện của psoriasis. Nếu phân loại cấp bệnh, gộp và lưu vị trí làm metadata. Nếu phân loại chi tiết, giữ nhãn con và rà soát ảnh trong nhãn cha.','https://dermnetnz.org/topics/psoriasis')
rule('Erythrodermic psoriasis|Generalised pustular psoriasis','hierarchy_keep_subtype','Psoriasis','Cùng nhóm nhưng thể bệnh có ý nghĩa lâm sàng; ưu tiên cấu trúc cha–con, chỉ gộp khi bài toán dùng cấp bệnh rộng.','https://dermnetnz.org/topics/psoriasis')
rule('Discoid lupus erythematosus','merge_if_coarse','Cutaneous lupus erythematosus','DLE là thể lupus da; không đồng nhất với systemic lupus hoặc neonatal lupus.','https://dermnetnz.org/topics/cutaneous-lupus-erythematosus')
rule('Macular amyloidosis|Amyloidosis cutis dyschromica','merge_if_coarse','Cutaneous amyloidosis','Các thể của amyloidosis da; lưu subtype.','https://dermnetnz.org/topics/cutaneous-amyloidosis')
rule('Erythema dyschromicum perstans','merge_if_coarse','Acquired dermal macular hyperpigmentation','Nhãn con trong nhóm tăng sắc tố dát trung bì mắc phải; không tự gộp melasma/postinflammatory hyperpigmentation.','https://dermnetnz.org/topics/erythema-dyschromicum-perstans')
rule('Cold urticaria|Delayed pressure urticaria|Solar urticaria','merge_if_coarse','Urticaria','Các dạng mày đay cảm ứng; lưu yếu tố khởi phát.','https://dermnetnz.org/topics/urticaria-an-overview')
rule('Rhinophyma','merge_if_coarse','Rosacea','Biểu hiện phymatous ở mũi thuộc phổ rosacea; lưu phenotype.','https://dermnetnz.org/topics/rosacea')
rule('Hyperkeratotic palmar dermatitis','merge_if_coarse','Hand dermatitis','Thể chàm bàn tay tăng sừng. Không đồng nghĩa mọi hand dermatitis đều là atopic dermatitis.')
rule('Acute localised exanthematous pustulosis','hierarchy_keep_subtype','AGEP/ALEP spectrum','Nguồn mô tả ALEP là thể khu trú của AGEP. Nếu gộp nên dùng nhãn phổ chung, tránh gọi ảnh khu trú là generalised.')
rule('Acral lentiginous melanoma|Amelanotic melanoma|Nodular melanoma|Melanoma in situ|Metastatic melanoma','hierarchy_keep_subtype','Melanoma','Các trục subtype, sắc tố và giai đoạn chồng lấp nhau. Có thể gộp ở cấp melanoma nhưng phải giữ stage/phenotype; không dùng chúng làm lớp đơn nhãn loại trừ nhau mà chưa rà soát.','https://dermnetnz.org/topics/melanoma')
rule('Ocular melanoma','keep_separate','Melanoma (cha khái niệm)','Giữ riêng modality/vị trí mắt; không tự trộn vào bài toán ảnh da ngoài.')
rule('Atypical melanocytic naevus|Blue nevus|Halo naevus|Meyerson naevus|Naevus of Ota, naevus of Ito and naevus of Hori','hierarchy_keep_subtype','Melanocytic naevus','Có thể tổ chức phân cấp nhóm tổn thương tế bào hắc tố, nhưng đặc điểm và vị trí khác nhau. Ưu tiên giữ chi tiết; Ota/Ito/Hori còn là nhãn chứa nhiều thực thể.','https://dermnetnz.org/topics/melanocytic-naevus')
rule('Anogenital squamous cell carcinoma|Oral squamous cell carcinoma','hierarchy_keep_subtype','Squamous cell carcinoma','Phân theo vị trí; giữ miền ảnh niêm mạc/sinh dục, chỉ gộp nếu scope bao gồm các vị trí này.')
rule('Marjolin ulcer','hierarchy_keep_subtype','Squamous cell carcinoma (khi nguồn xác nhận)','Ung thư phát sinh trên sẹo/loét, thường là SCC. Cần lịch sử/nguồn chẩn đoán; không coi là loét chân thông thường.')
rule('Flea bite','merge_if_coarse','Arthropod bites and stings','Nhóm con theo tác nhân; lưu tác nhân nếu có nguồn xác minh.')
rule('Papular urticaria','keep_separate','Arthropod bite reaction (liên hệ)','Phản ứng dạng sẩn do côn trùng; không gộp vào Urticaria chỉ vì tên có urticaria.','https://dermnetnz.org/topics/papular-urticaria')
rule('Exercise-induced vasculitis|Henoch–Schönlein purpura|Urticarial vasculitis|Erythema elevatum diutinum','hierarchy_keep_subtype','Cutaneous vasculitis (cấp biểu hiện da)','Có liên hệ nhóm viêm mạch nhưng khác căn nguyên/bối cảnh; giữ nhãn cụ thể. HSP là IgA vasculitis có thể có tổn thương hệ thống; không gộp vào Purpura như một bệnh đồng nghĩa.')
rule('Boil','hierarchy_keep_subtype','Bacterial folliculitis / Cutaneous abscess','Viêm nang lông sâu và có thể tạo áp xe; quan hệ chồng lấp, không phải ba tên đồng nghĩa.')
rule('Kerion','review_per_image','Tinea capitis / Tinea faciei / Tinea corporis','Biểu hiện viêm mạnh do dermatophyte; chỉ chuyển vào Tinea capitis nếu xác nhận ở da đầu.','https://dermnetnz.org/topics/kerion')
rule('Tinea barbae|Tinea capitis|Tinea corporis|Tinea cruris|Tinea faciei|Tinea manuum|Tinea pedis|Majocchi granuloma','hierarchy_keep_subtype','Dermatophytosis (cha mới nếu cần)','Giữ site/depth làm metadata. Đích chung chưa có trong images; không lấy Tinea corporis làm cha cho tất cả. Majocchi là nhiễm sâu.')
rule('Atopic dermatitis|Asteatotic eczema|Nummular eczematous dermatitis|Dyshidrotic eczema (pompholyx)|Venous eczema|Disseminated secondary eczema|Napkin dermatitis|Nipple eczema|Airborne contact dermatitis|Eyelid contact dermatitis|Nickel allergy','keep_separate','Dermatitis/eczema (cha khái niệm)','Các nhãn theo căn nguyên, vị trí, cơ chế khác nhau; có thể chồng lấp. Không gộp tất cả vào Atopic dermatitis. Với bài toán rộng tạo nhóm eczema mới và lưu các trục nhãn.')
for name,reason in {
 'Pruritus':'Triệu chứng ngứa, nhiều nguyên nhân; ảnh trùng với Notalgia paraesthetica và Prurigo không làm ba nhãn đồng nghĩa.',
 'Purpura':'Dấu hiệu xuất huyết da, chứa nhiều căn nguyên; 27 ảnh chung với Pigmented purpura và toàn bộ 6 ảnh HSP.',
 'Exanthems':'Thuật ngữ phát ban lan toả, không chỉ một bệnh hay một tác nhân.',
 'Cutaneous horn':'Hình thái sừng da, có nhiều tổn thương nền; không tự gộp vào Actinic keratosis hoặc SCC.',
 'Pityriasis amiantacea':'Kiểu phản ứng da đầu, có nhiều bệnh nền; không tự gộp vào psoriasis hoặc tinea.',
 'Elastosis':'Thuật ngữ thay đổi mô đàn hồi bao gồm nhiều tình trạng, không đồng nghĩa Elastosis perforans serpiginosa.',
 'Leg ulcer':'Loét chân là biểu hiện với nhiều nguyên nhân. Tách theo căn nguyên nếu nguồn cho phép.',
 'Madarosis':'Rụng lông mi/lông mày do nhiều bệnh; không tự gộp vào Alopecia areata.',
 'Poliosis':'Lông/tóc trắng khu trú, nhiều nguyên nhân; không tự gộp vào Vitiligo.',
 'Melanonychia':'Sắc tố móng, có nhiều nguyên nhân lành/ác tính; không tự chuyển thành melanoma.',
 'Macroglossia':'Lưỡi to là dấu hiệu với nhiều căn nguyên.',
 'Dry gangrene':'Biểu hiện hoại tử khô, nhiều nguyên nhân mạch máu/hệ thống.',
 'Amputation stump dermatoses':'Nhóm nhiều bệnh ở mỏm cụt; cần nhãn bệnh cụ thể cho từng ảnh.',
 'Granulomatous dermatitis':'Nhóm kiểu phản ứng/mô bệnh học; không tự gộp vào sarcoidosis hay granuloma annulare.',
 'Panniculitis':'Nhóm viêm mô mỡ dưới da; có thể giữ làm cha, không coi mọi ảnh là erythema nodosum.',
 'Palmoplantar keratoderma':'Nhóm dày sừng lòng bàn tay/bàn chân, có dạng di truyền và mắc phải.',
 'Ichthyosis':'Nhóm bệnh nhiều thể; giữ làm cha hoặc gán subtype từ nguồn.',
 'Oral mucositis - Stomatitis':'Viêm niêm mạc miệng do nhiều nguyên nhân; không đồng nghĩa Aphthous ulcer.',
 'Wound infection':'Nhiễm trùng vết thương nhiều tác nhân; không đồng nhất với Pyoderma hoặc áp xe.',
 'Malignant histiocytoses':'Nhãn nhóm rộng/thuật ngữ lịch sử; cần rà từng thực thể, không gộp toàn bộ vào Leukaemia cutis.',
}.items(): rule(name,'separate_sign_or_umbrella',None,reason)
rule('Atypical solar lentigo','quarantine_label_review',None,'Thuật ngữ dùng cho tổn thương không chắc là solar lentigo lành tính hay melanoma in situ; không gộp mù vào nhóm lành tính.')
rule('Palmoplantar pustulosis|Acrodermatitis continua of Hallopeau','keep_separate','Pustular disease spectrum','Giữ subtype; palmoplantar pustulosis có thể là thực thể riêng dù liên hệ psoriasis. Không tự gộp vào Generalised pustular psoriasis.','https://dermnetnz.org/topics/palmoplantar-pustulosis')
rule('Psoriatic arthritis|Systemic lupus erythematosus|Neonatal lupus|Epidermolysis bullosa acquisita|Pemphigoid gestationis|Pemphigus foliaceus|Pemphigus vulgaris|Sézary syndrome|Pseudoporphyria|Acquired lymphangiectasia|Lymphatic malformation|Erysipelas|Cellulitis','keep_separate',None,'Giữ chẩn đoán riêng; có quan hệ hoặc biểu hiện giống nhãn khác nhưng không đủ cơ sở coi là đồng nghĩa hay xoá lớp.')
renames={
 'Benign familial pemphigus':('Hailey-Hailey disease','Tên đồng nghĩa, không gộp vào pemphigus tự miễn.'),
 'Bazin_s hydroa vacciniforme':('Hydroa vacciniforme','Chuẩn hoá tên theo bài nguồn.'),
 'Blue nevus':('Blue naevus','Thống nhất chính tả naevus/nevus; không bắt buộc.'),
 'Nevus comedonicus':('Naevus comedonicus','Thống nhất chính tả; không phải melanocytic naevus.'),
 'Milker_s nodule':("Milker’s nodule",'Sửa dấu nháy bị thay bằng dấu gạch dưới.'),
 'Swimmer_s itch':("Swimmer’s itch",'Sửa dấu nháy bị thay bằng dấu gạch dưới.'),
 'Chloracne _ MADISH':('Chloracne (MADISH)','Chuẩn hoá dấu phân cách.'),
 'Plasma cell balanitis_vulvitis':('Plasma cell balanitis / vulvitis','Chuẩn hoá nhãn; giữ thông tin vị trí.')}
for n,(target,why) in renames.items():rule(n,'rename_only',target,why)
save('taxonomy_recommendations.json',rules)
by=defaultdict(list)
for r in rules:by[r['folder']].append(r)
allrows=[]
for f in F:
    allrows.append({**f,'recommendations':by.get(f['folder'],[]),'status':'flagged_or_related' if f['folder'] in by else 'no_specific_merge_flag_from_name_screen','note':'Sàng lọc taxonomy không xác nhận chẩn đoán từng ảnh.'})
save('all_folder_assessments.json',allrows)

# Reviewable within-folder de-duplication proposal, never an executable deletion script.
manifest=[]
for g in G:
    buckets=defaultdict(list)
    for x in g['paths']:buckets[x.split('/')[0]].append(x)
    for folder,ps in buckets.items():
        if len(ps)<2:continue
        ps.sort(key=lambda x:(bool(re.fullmatch(r'\d+',Path(x).stem)), -ii[x]['bytes'],x))
        for x in ps[1:]:manifest.append({'action':'proposed_remove_redundant_within_folder','keep':ps[0],'redundant':x,'pixel_sha256':g['hash'],'reason':'Same dimensions and identical decoded RGBA pixels after EXIF orientation. Keep all source/path metadata before any removal.','executed':False})
save('within_folder_dedup_proposal.json',manifest)
cross=[{'id':f'P{i+1:04d}',**g,'action':'review_labels_and_provenance_before_dedup'} for i,g in enumerate(G) if g['scope']=='cross_folder']
save('cross_folder_label_review.json',cross)
samples=read('review_sample_pairs.json')
for x in samples:x['visual_review']='same_photographic_content_in_contact_sheet; diagnosis_not_verified'
save('visual_review_sample.json',samples)

sources=[]
for n in sorted(by):
    p=OUT.parent/'contents'/f'Toàn bộ nội dung - {n}.txt'
    if p.exists():
        t=p.read_text(); pos=t.find('## ')
        sources.append({'folder':n,'local_file':str(p),'heading':t.splitlines()[0],'definition_excerpt':t[pos:pos+1400] if pos>=0 else t[:1400],'caution':'Local translated text may contain translation errors; not independent clinical verification.'})
save('local_source_evidence.json',sources)

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join('---' for _ in headers)+'|\n'+ '\n'.join('| '+' | '.join(str(c).replace('|',' / ').replace('\n',' ') for c in row)+' |' for row in rows)+'\n'
def srclink(r):
    if r['source_url']:return '[DermNet]('+r['source_url']+')'
    if r['local_source']:return '[Bài nguồn lưu cục bộ](<'+r['local_source']+'>)'
    return 'Đề xuất thiết kế nhãn; cần xác minh nguồn từng ảnh'
lines=['# Kiểm tra dermnet-output/images', '', 'Ngày kiểm tra: 28–29/09/2026. Không di chuyển, đổi tên hoặc xoá ảnh/thư mục gốc. Báo cáo phục vụ chuẩn hoá bộ dữ liệu, không xác nhận chẩn đoán từ ảnh.', '', '## 1. Kết quả toàn bộ dữ liệu', '',table(['Chỉ số','Kết quả'],[
('Thư mục nhãn',S['folders']),('Ảnh được đọc/giải mã',S['image_files']),('Ảnh lỗi đọc / thư mục không có ảnh','0 / 0'),('Tệp ngoài ảnh','88 tệp .DS_Store'),('Nhóm trùng byte SHA-256',S['bytes_duplicate_groups']),('Bản dư theo byte',S['bytes_redundant_copies']),('Nhóm trùng pixel',S['pixel_duplicate_groups']),('Bản dư theo pixel trên toàn bộ tập',S['pixel_redundant_copies']),('Ảnh pixel duy nhất',len(I)-S['pixel_redundant_copies']),('Bản dư trong cùng thư mục',len(manifest)),('Nhóm pixel xuất hiện ở nhiều thư mục',len(cross)),('Cặp thư mục có chung ảnh pixel',len(O)),('Cặp nghi gần trùng, cần duyệt',len(N))]),
 '358 bản dư theo byte nằm trong 380 bản dư theo pixel: không cộng hai số này. 380 = 146 bản dư nội bộ + 234 bản dư còn lại giữa nhãn sau khi khử trùng nội bộ. Nếu chỉ khử trùng nội bộ, còn 7.548 tệp; nếu mỗi nội dung pixel chỉ giữ một bản toàn bộ tập, còn 7.314 ảnh. Đây là phép đếm, chưa phải khuyến nghị xoá 380 ảnh ngay.', '',
 '## 2. Cách kiểm tra và giới hạn', '',
 '- Đọc đệ quy toàn bộ images; lập danh mục đường dẫn, kích thước, định dạng, SHA-256. Không dùng tên file để kết luận trùng.',
 '- Trùng byte: toàn bộ nội dung tệp giống nhau. Trùng pixel: cùng kích thước và cùng RGBA sau khi áp dụng EXIF orientation, bất kể metadata/container. Ảnh động được loại khỏi so sánh pixel/pHash; bộ hiện tại không có ảnh động.',
 '- Gần trùng: pHash 63 bit từ DCT ảnh xám 32×32, khoảng cách Hamming ≤6; tìm toàn bộ tập bằng BK-tree, mỗi nhóm pixel chỉ lấy một đại diện. 175 là số cặp ứng viên, không phải 175 ảnh có thể xoá và không cộng vào 380.',
 '- Đã xem trực quan 32 cặp mẫu: 20 cặp gần trùng và 12 cặp trùng pixel đáng ngờ ở VIN; cả mẫu đều cho thấy cùng nội dung ảnh chụp. 155 cặp gần trùng còn lại chưa được duyệt trực quan. Danh sách mẫu nằm trong visual_review_sample.json.',
 '- pHash có thể bỏ sót crop lớn, xoay/lật, watermark lớn hoặc góc chụp khác; không phát hiện đầy đủ cùng bệnh nhân. Không bảo đảm mọi trùng ảnh biến đổi đã được tìm thấy.',
 '- Pixel hash không chuẩn hoá ICC profile; không khẳng định hiển thị giống hệt trong mọi phần mềm quản lý màu. Phát hiện trùng ảnh không xác định nhãn nào đúng.',
 '- Sàng lọc toàn bộ 338 tên thư mục; đối chiếu bài nguồn cục bộ cho các trường hợp được gắn cờ và DermNet gốc cho các quan hệ chính. Không phải thẩm định lâm sàng từng ảnh của cả 338 lớp. Bản dịch nguồn có lỗi, ví dụ một số chỗ dịch Hydroa vacciniforme thành thủy đậu.', '',
 '## 3. Gộp tên đồng nghĩa và vấn đề cần xử lý trước', '',table(['Thư mục','Số ảnh','Đề xuất','Lý do / bằng chứng'],[(r['folder'],fi[r['folder']]['images'],r['action']+' → '+str(r['target'] or 'rà soát'),r['reason']+' '+srclink(r)) for r in rules if r['action'] in ('merge_synonym','quarantine_label_review','quarantine_image_review')]),
 'Ở snapshot ban đầu, cặp Papular mucinosis → Lichen myxoedematosus đủ rõ để đề xuất hợp nhất tên lớp ngay về mặt taxonomy. Pyoderma có bài nguồn trùng nhưng tập ảnh khác nên chưa được coi là thư mục thừa có thể xoá. Ảnh số thứ tự không có thông tin nguồn đủ để tự gán lại nhãn.', '',
 '## 4. Các nhóm cha–con có thể gộp tùy cấp phân loại', '',
 'Khuyến nghị lưu `canonical_disease`, `subtype`, `site`, `stage`, `source_label`, `source_url`, `image_group_id`. Với bộ đơn nhãn, chọn cùng cấp độ phân loại; tránh cho nhãn cha và nhãn con cạnh tranh nhau. Gộp là giữ ảnh không trùng và metadata, không xoá ảnh chỉ vì thuộc subclass.', '',table(['Thư mục','Quan hệ / đích','Đề xuất và nguồn'],[(r['folder'],r['target'],r['reason']+' '+srclink(r)) for r in rules if r['action'] in ('merge_if_coarse','hierarchy_keep_subtype','review_per_image')]),
 '## 5. Thuật ngữ, dấu hiệu và nhóm quá rộng', '',
 'Nếu mục tiêu là nhận diện một bệnh cụ thể từ một ảnh, đề xuất tách các nhãn sau khỏi tập chẩn đoán đơn nhãn sang nhóm dấu hiệu/nhãn tổng quát hoặc rà từng ảnh để gán bệnh nền. Không đề xuất xoá vĩnh viễn thư mục: các nhãn này vẫn hữu ích cho bài toán nhận diện hình thái hoặc đa nhãn.', '',table(['Thư mục','Lý do / hành động'],[(r['folder'],r['reason']+' '+srclink(r)) for r in rules if r['action']=='separate_sign_or_umbrella']),
 '## 6. Các tên gần nhau nhưng không nên gộp tự động', '',table(['Thư mục','Lưu ý'],[(r['folder'],r['reason']+' '+srclink(r)) for r in rules if r['action']=='keep_separate']),
 'Đặc biệt: Epidermolysis bullosa acquisita không phải chỉ là tên khác của nhóm EB di truyền; Hailey-Hailey không phải pemphigus tự miễn; Sézary syndrome và Mycosis fungoides cần giữ riêng; Pseudoporphyria không đồng nghĩa Porphyria cutanea tarda. Giữ VIN/penile intraepithelial neoplasia riêng với SCC xâm lấn. Trùng ảnh có thể do đồng bệnh, chẩn đoán phân biệt hoặc lỗi thu thập.', '',
 '## 7. Chuẩn hoá tên, không xoá lớp', '',table(['Tên hiện tại','Tên đề xuất','Lý do'],[(r['folder'],r['target'],r['reason']) for r in rules if r['action']=='rename_only']),
 '## 8. Các cặp thư mục trùng ảnh nhiều nhất', '',
 'Số chung tính theo ảnh pixel duy nhất, tỷ lệ có mẫu số là số ảnh pixel duy nhất trong từng thư mục. Trùng toàn bộ tập ảnh không đồng nghĩa hai bệnh đồng nghĩa.', '',table(['Thư mục A','Thư mục B','Ảnh chung','% A','% B'],[(r['a'],r['b'],r['shared_unique_images'],f"{100*r['coverage_a']:.1f}",f"{100*r['coverage_b']:.1f}") for r in O[:25]]),
 '## 9. Các thư mục có nhiều bản dư nội bộ', '',table(['Thư mục','Tệp ảnh','Ảnh pixel duy nhất','Bản dư'],[(f['folder'],f['images'],f['unique_pixels'],f['readable']-f['unique_pixels']) for f in sorted(F,key=lambda x:x['readable']-x['unique_pixels'],reverse=True)[:20]]),
 '## 10. Đề xuất xử lý theo thứ tự', '',
 '1. Cách ly khỏi tập huấn luyện các ảnh/nhãn chưa rõ ở VIN, Pyoderma và Non-albicans candida infections; xác minh nguồn. Không gộp các bệnh không liên quan vì ảnh trùng.',
 '2. Chuẩn hoá Papular mucinosis thành Lichen myxoedematosus; quyết định dùng nhãn cấp bệnh hay nhãn phân cấp trước khi gộp các subclass.',
 '3. Rà manifest 146 bản dư nội bộ; giữ một ảnh đại diện và tất cả metadata/nguồn. Manifest chỉ là đề xuất, không chứa lệnh xoá.',
 '4. Với 228 nhóm trùng xuyên thư mục, giữ một image_id và tập nhãn đã xác minh; nếu cần đơn nhãn thì giải quyết xung đột trước. Không chọn nhãn theo thứ tự tên thư mục.',
 '5. Duyệt 175 cặp gần trùng trong gallery; ưu tiên bản có độ phân giải tốt sau khi kiểm tra nội dung. Không khử trùng chỉ dựa trên pHash.',
 '6. Chia train/validation/test theo nhóm ảnh/case/patient nếu có, để ảnh trùng và biến thể của nó không rơi vào các split khác nhau.', '',
 '## 11. Tệp kết quả', '',
 '- `gallery.html`: xem từng nhóm trùng và cặp nghi gần trùng, lọc theo tên thư mục/đường dẫn, mở ảnh gốc.',
 '- `within_folder_dedup_proposal.json`: 146 bản dư nội bộ và ảnh đại diện đề xuất.',
 '- `cross_folder_label_review.json`: 228 nhóm trùng xuyên nhãn.',
 '- `duplicates_bytes.json`, `duplicates_pixels.json`: toàn bộ nhóm trùng chính xác.',
 '- `near_duplicate_candidates.json`: 175 cặp nghi gần trùng; đại diện pixel-group, không mở rộng thành mọi tổ hợp file.',
 '- `folder_overlap.json`: toàn bộ 103 cặp thư mục có chung ảnh.',
 '- `taxonomy_recommendations.json`: khuyến nghị có điều kiện cho từng thư mục được gắn cờ.',
 '- `all_folder_assessments.json`: đủ 338 thư mục, bao gồm những thư mục chưa thấy vấn đề taxonomy rõ từ sàng lọc tên.',
 '- `local_source_evidence.json`: đường dẫn và đoạn định nghĩa nguồn cục bộ; `inventory.json`, `folders.json`, `summary.json`: danh mục và thống kê.',
 '- `audit.py`, `build_report.py`: mã tái lập. Chạy bằng `dermnet-output/.venv/bin/python`; chỉ ghi trong audit-images.', '']
report='\n'.join(lines).replace('338',str(S['folders'])).replace('7.548',f"{len(I)-len(manifest):,}".replace(',', '.')).replace('7.314',f"{len(I)-S['pixel_redundant_copies']:,}".replace(',', '.'))
report += '\n## Thay đổi trong lúc kiểm tra\n\nLần quét đầu có 338 thư mục và 7.694 ảnh. Khi kiểm tra lại, hai ảnh Papular mucinosis/01.jpg và 02.jpg không còn, thư mục Papular mucinosis cũng không còn. Các thao tác kiểm tra không xoá hoặc di chuyển ảnh. Đã quét lại trạng thái hiện tại; snapshot ban đầu được giữ ở initial_snapshot/. Quan hệ đồng nghĩa Papular mucinosis → Lichen myxoedematosus vẫn đúng, nhưng không còn thư mục nguồn để đề xuất gộp ở trạng thái cuối.\n'
(OUT/'REPORT_VI.md').write_text(report,encoding='utf-8')

cards=[]
for i,g in enumerate(G):cards.append({'id':f'P{i+1:04d}','kind':'Trùng pixel','scope':g['scope'],'paths':g['paths'],'detail':g['hash'][:16]})
for i,g in enumerate(N):cards.append({'id':f'N{i+1:04d}','kind':'Nghi gần trùng','scope':g['scope'],'paths':[g['a'],g['b']],'detail':f"pHash distance={g['phash_distance']}; chỉ là ứng viên"})
doc='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Kiểm tra ảnh DermNet</title>
<style>body{font:16px system-ui;margin:24px;background:#f4f6f8;color:#17232e}header{position:sticky;top:0;background:#f4f6f8;padding:12px 0;z-index:1}input,select{padding:10px;margin:4px;max-width:95%}input{width:420px}article{background:white;border:1px solid #cbd5df;border-radius:10px;padding:18px;margin:16px 0}.imgs{display:flex;flex-wrap:wrap;gap:18px}figure{margin:0;width:280px}img{width:280px;height:220px;object-fit:contain;background:#edf0f3}figcaption{overflow-wrap:anywhere;font-size:14px}small{color:#52606d}button{padding:10px;margin:4px}a{color:#155a9a}</style>
<header><h1>Kiểm tra ảnh DermNet</h1><p>7.694 ảnh · 338 thư mục · 366 nhóm trùng pixel · 175 cặp nghi gần trùng. Chưa xoá hoặc đổi nhãn ảnh.</p><input id="q" placeholder="Tên bệnh hoặc đường dẫn"><select id="kind"><option value="">Tất cả</option><option>Trùng pixel</option><option>Nghi gần trùng</option></select><select id="scope"><option value="">Mọi phạm vi</option><option value="within_folder">Trong cùng thư mục</option><option value="cross_folder">Giữa các thư mục</option></select><div id="count"></div></header><p>Ảnh lâm sàng dùng để kiểm tra dữ liệu. Trùng ảnh không chứng minh hai bệnh đồng nghĩa. Các cặp pHash cần duyệt; không có nút xoá.</p><main id="items"></main><button id="prev">Trang trước</button><button id="next">Trang sau</button>
<script>const data=__DATA__;let page=0;const size=30;const el=id=>document.getElementById(id);function render(){const q=el('q').value.toLowerCase(),kind=el('kind').value,scope=el('scope').value;const rows=data.filter(r=>(!kind||r.kind===kind)&&(!scope||r.scope===scope)&&(!q||r.paths.some(p=>p.toLowerCase().includes(q))));page=Math.min(page,Math.max(0,Math.ceil(rows.length/size)-1));el('items').replaceChildren();el('count').textContent=`${rows.length} nhóm/cặp · Trang ${page+1}/${Math.max(1,Math.ceil(rows.length/size))}`;for(const r of rows.slice(page*size,(page+1)*size)){const a=document.createElement('article'),h=document.createElement('h3'),s=document.createElement('p'),wrap=document.createElement('div');h.textContent=`${r.id} · ${r.kind} · ${r.scope==='cross_folder'?'Giữa các thư mục':'Cùng thư mục'}`;s.textContent=r.detail;wrap.className='imgs';a.append(h,s,wrap);for(const p of r.paths){const f=document.createElement('figure'),link=document.createElement('a'),im=document.createElement('img'),cap=document.createElement('figcaption');link.href='../images/'+p.split('/').map(encodeURIComponent).join('/');link.target='_blank';im.src=link.href;im.loading='lazy';im.alt=p;cap.textContent=p;link.append(im);f.append(link,cap);wrap.append(f)}el('items').append(a)}el('prev').disabled=page===0;el('next').disabled=(page+1)*size>=rows.length}for(const id of ['q','kind','scope'])el(id).addEventListener('input',()=>{page=0;render()});el('prev').onclick=()=>{page--;render();window.scrollTo(0,0)};el('next').onclick=()=>{page++;render();window.scrollTo(0,0)};render();</script></html>'''
doc=doc.replace('7.694',f'{len(I):,}'.replace(',', '.')).replace('338',str(S['folders']))
(OUT/'gallery.html').write_text(doc.replace('__DATA__',json.dumps(cards,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
assert len(manifest)==146
assert len(allrows)==S['folders']
print(json.dumps({'recommendation_rows':len(rules),'flagged_folders':len(by),'within_folder_proposals':len(manifest),'gallery_cards':len(cards)},ensure_ascii=False))
