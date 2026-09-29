"""Read-only image audit. Outputs only into this script's directory."""
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, itertools, re, time
import numpy as np
from PIL import Image, ImageOps

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent / 'images'
N=32
DCT=np.cos(np.pi*(2*np.arange(N)[None,:]+1)*np.arange(8)[:,None]/(2*N))
def dump(name, data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def inspect(p):
    r={'path':p.relative_to(ROOT).as_posix(),'folder':p.relative_to(ROOT).parts[0],'bytes':p.stat().st_size}
    try:
        r['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        with Image.open(p) as im:
            r.update(format=im.format,frames=getattr(im,'n_frames',1))
            im=ImageOps.exif_transpose(im).convert('RGBA')
            r.update(width=im.width,height=im.height)
            r['pixel_sha256']=hashlib.sha256(f'{im.size}:RGBA:'.encode()+im.tobytes()).hexdigest()
            gray=im.convert('L').resize((32,32),Image.Resampling.LANCZOS)
            low=(DCT@np.asarray(gray,dtype=float)@DCT.T).ravel()[1:]
            r['phash63']=int(''.join('1' if v>np.median(low) else '0' for v in low),2)
            r['gray_std']=float(np.asarray(gray).std())
    except Exception as e: r['error']=str(e)
    return r
def groups(records,key):
    d=defaultdict(list)
    for r in records:
        if key in r:d[r[key]].append(r['path'])
    return [{'hash':k,'paths':v,'scope':'cross_folder' if len({p.split('/')[0] for p in v})>1 else 'within_folder'} for k,v in d.items() if len(v)>1]
def main():
    paths=sorted(p for p in ROOT.rglob('*') if p.is_file())
    image_ext={'.jpg','.jpeg','.png','.webp','.gif','.bmp','.tif','.tiff','.avif'}
    candidates=[p for p in paths if p.suffix.lower() in image_ext]
    with ThreadPoolExecutor(max_workers=8) as ex: records=list(ex.map(inspect,candidates))
    dump('inventory.json',records)
    exact=groups(records,'sha256'); pixels=groups([r for r in records if r.get('frames')==1],'pixel_sha256')
    dump('duplicates_bytes.json',exact); dump('duplicates_pixels.json',pixels)
    # Compare one representative per pixel-identical group. BK tree searches all hashes within radius 6.
    unique={}
    for r in records:
        if 'pixel_sha256' in r and r.get('frames')==1:unique.setdefault(r['pixel_sha256'],r)
    tree=None; near=[]
    for r in unique.values():
        h=r['phash63']
        if tree is None:tree=[h,[r],{}];continue
        stack=[tree]
        while stack:
            node=stack.pop(); d=(h^node[0]).bit_count()
            if d<=6:
                for s in node[1]:
                    aspect=max(r['width']/r['height'],s['width']/s['height'])/min(r['width']/r['height'],s['width']/s['height'])
                    near.append({'a':s['path'],'b':r['path'],'phash_distance':d,'aspect_ratio_factor':round(aspect,4),'scope':'within_folder' if s['folder']==r['folder'] else 'cross_folder','review':'candidate_only'})
            stack.extend(n for k,n in node[2].items() if d-6<=k<=d+6)
        node=tree
        while True:
            d=(h^node[0]).bit_count()
            if d==0:node[1].append(r);break
            if d not in node[2]:node[2][d]=[h,[r],{}];break
            node=node[2][d]
    near.sort(key=lambda x:(x['phash_distance'],x['a'],x['b']))
    dump('near_duplicate_candidates.json',near)
    folders=[]
    byfolder=defaultdict(list)
    for r in records:byfolder[r['folder']].append(r)
    for p in sorted(ROOT.iterdir()):
        if not p.is_dir():continue
        rs=byfolder[p.name]; good=[r for r in rs if 'error' not in r]
        folders.append({'folder':p.name,'images':len(rs),'readable':len(good),'unique_bytes':len({r['sha256'] for r in good}),'unique_pixels':len({r['pixel_sha256'] for r in good}),'bytes':sum(r['bytes'] for r in rs)})
    dump('folders.json',folders)
    pairs=defaultdict(lambda:{'shared_unique_images':0,'examples':[]})
    for g in pixels:
        fs=sorted({p.split('/')[0] for p in g['paths']})
        for a,b in itertools.combinations(fs,2):
            q=pairs[a,b];q['shared_unique_images']+=1
            if len(q['examples'])<3:q['examples'].append(g['paths'])
    counts={f['folder']:f['unique_pixels'] for f in folders}
    overlap=[]
    for (a,b),q in pairs.items():
        overlap.append(dict(a=a,b=b,**q,coverage_a=round(q['shared_unique_images']/counts[a],4),coverage_b=round(q['shared_unique_images']/counts[b],4)))
    overlap.sort(key=lambda x:-x['shared_unique_images']);dump('folder_overlap.json',overlap)
    texts=defaultdict(list); headings=defaultdict(list)
    for f in folders:
        p=OUT.parent/'contents'/('Toàn bộ nội dung - '+f['folder']+'.txt')
        if p.exists():
            t=p.read_text(encoding='utf-8-sig');texts[hashlib.sha256(t.encode()).hexdigest()].append(f['folder'])
            headings[t.splitlines()[0]].append(f['folder'])
    dump('source_duplicates.json',{'same_text':[v for v in texts.values() if len(v)>1],'same_heading':[{'heading':k,'folders':v} for k,v in headings.items() if len(v)>1]})
    summary={'root':str(ROOT),'folders':len(folders),'all_files':len(paths),'image_files':len(records),'non_image_files':[str(p.relative_to(ROOT)) for p in paths if p not in set(candidates)],'unreadable':[r for r in records if 'error' in r],'empty_image_folders':[f['folder'] for f in folders if not f['images']], 'animated_files':[r['path'] for r in records if r.get('frames',1)>1], 'bytes_duplicate_groups':len(exact),'bytes_redundant_copies':sum(len(g['paths'])-1 for g in exact),'pixel_duplicate_groups':len(pixels),'pixel_redundant_copies':sum(len(g['paths'])-1 for g in pixels),'pixel_cross_folder_groups':sum(g['scope']=='cross_folder' for g in pixels),'pixel_within_only_groups':sum(g['scope']=='within_folder' for g in pixels),'pixel_within_folder_redundant_copies':sum(f['readable']-f['unique_pixels'] for f in folders),'near_candidate_pairs':len(near),'near_cross_folder_pairs':sum(g['scope']=='cross_folder' for g in near),'folder_overlap_pairs':len(overlap)}
    dump('summary.json',summary); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
