"""Posthoc evidence panels from frozen masks and hash-verified original RGB."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw

R=Path(__file__).resolve().parent/'tracking';F=R/'figures';F.mkdir(exist_ok=True)


def rgb(clip,fid):
    audit=json.loads((R/'full'/clip/'INPUTS.json').read_text())
    f=next(f for f in audit['frames'] if f['frame_id']==fid)
    p=Path(f['color_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==f['color_sha256']
    im=Image.open(p).convert('RGB')
    if clip.startswith('3rscan'):im=im.transpose(Image.Transpose.ROTATE_90)
    assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==f['rgb_decoded_sha256']
    return im


def paint(im,path,label):
    arr=np.asarray(im).copy();d=np.load(path);shape=d['mask_shape']
    masks=np.unpackbits(d['masks_packed'],axis=-1)[...,:shape[-1]].astype(bool)
    for i,mask in enumerate(masks):
        mask=np.asarray(Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST))
        color=np.array([(235,85,60),(30,175,245),(225,200,40),(180,80,225)][i%4])
        arr[mask]=(.6*arr[mask]+.4*color).astype(np.uint8)
    out=Image.fromarray(arr);draw=ImageDraw.Draw(out);draw.rectangle((0,0,out.width,24),fill='black')
    draw.text((5,5),f'{label}: {len(masks)} masks',fill='white')
    return out


def panel(images,path,width=400,height=320):
    out=Image.new('RGB',(width*len(images),height),'#f5f5f5')
    for i,im in enumerate(images):
        im=im.copy();im.thumbnail((width,height));out.paste(im,(i*width+(width-im.width)//2,(height-im.height)//2))
    out.save(path)


def main():
    selection=[('scannet_chair',40),('3rscan_bed',0),('3rscan_bed',6),('orbbec_curtain',1305),('orbbec_reel',2420)]
    for clip,fid in selection:
        im=rgb(clip,fid);base=R/'full'/clip
        panel([im,paint(im,base/'image'/f'{fid:06}.npz','independent image'),
               paint(im,base/'video'/f'{fid:06}.npz','video tracking')],F/f'{clip}_{fid:06}_comparison.png')
    clip='orbbec_reel';fid=2400;im=rgb(clip,fid);base=R/'full'/clip
    panel([im,paint(im,base/'image'/f'{fid:06}.npz','text: cable reel'),
           paint(im,base/'visual_image_positive'/f'{fid:06}.npz','assisted image box'),
           paint(im,base/'video_visual'/f'{fid:06}.npz','assisted video box')],F/'orbbec_reel_assistance.png')
    canvas=Image.new('RGB',(320*4,260*4),'white')
    for i,fid in enumerate(range(2400,2480,5)):
        im=rgb(clip,fid);im.thumbnail((320,240));canvas.paste(im,(i%4*320,i//4*260+20))
        ImageDraw.Draw(canvas).text((i%4*320+4,i//4*260+4),str(fid),fill='black')
    canvas.save(F/'orbbec_reel_rgb_contact.png')
    blue=R/'blue_equipment';im=Image.open(blue/'rgb.png').convert('RGB')
    panel([im]+[paint(im,blue/arm/'004811.npz',arm) for arm in ['text_machine','text_blue_machine','visual_positive','visual_positive_negative']],F/'blue_equipment_comparison.png',320,260)
    records=[]
    for path in sorted((R/'full').glob('*/*_RESULT.json')):
        data=json.loads(path.read_text());records.append({'clip':path.parent.name,'arm':path.stem.replace('_RESULT','').lower(),
            'seconds':data['seconds'],'frames':len(data['frames']),
            'nonempty_frames':sum(f['mask_count']>0 for f in data['frames']),
            'projected_map_points_union':data['metrics']['segmented_map_points_union'],
            'mean_foreground_iou':data['metrics']['mean_foreground_iou'],
            'mean_matched_mask_iou':data['per_mask_metrics']['mean_matched_iou'],
            'unmatched_mask_observations':data['per_mask_metrics']['unmatched_mask_observations'],
            'merge_like_transitions':data['per_mask_metrics']['merge_like_targets'],
            'split_like_transitions':data['per_mask_metrics']['split_like_sources']})
    (R/'SUMMARY.json').write_text(json.dumps({'rows':records,'accuracy_measured':False,'no_default_promotion':True},indent=2)+'\n')
    print(json.dumps({'figures':[str(p.relative_to(R)) for p in sorted(F.glob('*.png'))],'rows':len(records)}))


if __name__=='__main__':main()
