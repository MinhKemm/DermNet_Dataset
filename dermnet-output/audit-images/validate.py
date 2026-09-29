from pathlib import Path
from PIL import Image, ImageOps
import hashlib,json
P=Path(__file__).resolve().parent
root=P.parent/'images'
def load(n):return json.loads((P/n).read_text())
inv=load('inventory.json'); groups=load('duplicates_pixels.json');near=load('near_duplicate_candidates.json')
for r in inv:
    f=root/r['path']
    assert f.exists(),r['path']
    assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'],r['path']
for g in groups:
    arrays=[]
    for rel in g['paths']:
        with Image.open(root/rel) as im:
            im=ImageOps.exif_transpose(im).convert('RGBA');arrays.append((im.size,im.tobytes()))
    assert all(x==arrays[0] for x in arrays)
u={}
for r in inv:u.setdefault(r['pixel_sha256'],r)
rs=list(u.values());found=set()
for i,r in enumerate(rs):
    for s in rs[:i]:
        if (r['phash63']^s['phash63']).bit_count()<=6:found.add(tuple(sorted((r['path'],s['path']))))
assert found=={tuple(sorted((r['a'],r['b']))) for r in near}
m=load('within_folder_dedup_proposal.json');by={r['path']:r for r in inv}
for r in m:
    assert r['keep'].split('/')[0]==r['redundant'].split('/')[0]
    assert by[r['keep']]['pixel_sha256']==by[r['redundant']]['pixel_sha256']
assert len({r['redundant'] for r in m})==len(m)
assert not {r['keep'] for r in m}&{r['redundant'] for r in m}
for f in P.glob('*.json'):json.loads(f.read_text())
result={'snapshot_images_sha256_verified':len(inv),'pixel_groups_verified_by_direct_comparison':len(groups),'phash_BK_tree_matches_exhaustive_search':True,'near_pairs_verified_algorithmically':len(found),'dedup_proposal_valid':len(m),'scope':'Validates latest snapshot; initial snapshot differences are recorded separately.'}
(P/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
