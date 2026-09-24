"""Build a credential-free review of fixed-crop API benchmark receipts."""
import argparse
import base64
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    plan = read(root/'plan/PLAN.json')
    local = read(root/'local24/RECORDS.json')
    local_by_id = {row['id']:row for row in local}
    durations = sorted(r['request_seconds'] for r in local)
    baseline = {'profile':'qwen3vl_2b_nf4', 'requests':len(local),
        'requests_seconds':sum(durations),'p50_request_seconds':statistics.median(durations),
        'p95_request_seconds':durations[int(.95*len(durations))],
        'unknown':sum(r['label']=='unknown' for r in local),
        **read(root/'local24/PROFILE_SUMMARY.json')}
    profiles=[]; all_records=[]; sources=[]
    for name in ('deepseek-serial','concurrent4','recovery'):
        directory=root/name
        if not (directory/'SUMMARY.json').exists():
            continue
        records=read(directory/'RECORDS.json')
        profiles_plan={p['id']:p for p in read(directory/'PLAN.json')['profiles']}
        for summary in read(directory/'SUMMARY.json'):
            rows=[r for r in records if r['profile']==summary['profile']]
            p=profiles_plan[summary['profile']]
            report={**summary,'endpoint':p['endpoint'],'model':p['model'],
                'effort':p['effort'],'concurrency':p['concurrency'],
                'crop_ids':[r['crop_id'] for r in rows],
                'format_invalid':sum(r['ok'] and not r.get('valid') for r in rows),
                'truncated':sum(r.get('finish_reason')=='length' or r.get('status')=='incomplete' for r in rows),
                'failures':dict(Counter(r.get('error','HTTP failure') for r in rows if not r['ok'])),
                'returned_efforts':dict(Counter((r.get('returned_reasoning') or {}).get('effort','not returned') for r in rows if r['ok'])),
                'effort_mismatches':sum(r['ok'] and (r.get('returned_reasoning') or {}).get('effort') not in (None,p['effort']) for r in rows),
                'exact_label_matches_to_local':sum(r.get('label')==local_by_id[r['crop_id']]['label'] for r in rows if r['ok'] and r.get('completed') and r.get('valid')),
                'exact_label_match_is_accuracy':False}
            output_detail='output_tokens_details' if p['wire']=='responses' else 'completion_tokens_details'
            input_detail='input_tokens_details' if p['wire']=='responses' else 'prompt_tokens_details'
            for label,keys in [('reasoning_tokens',(output_detail,'reasoning_tokens')),
                               ('cached_tokens',(input_detail,'cached_tokens'))]:
                values=[(r.get('usage') or {}).get(keys[0],{}).get(keys[1]) for r in rows]
                known=[v for v in values if isinstance(v,(int,float))]
                report[label]=sum(known) if known else None
            profiles.append(report)
        all_records.extend(records)
        sources.extend([directory/'PLAN.json',directory/'RECORDS.json',directory/'SUMMARY.json'])
        if (directory/'COMPLETE.json').exists():
            sources.append(directory/'COMPLETE.json')
    sources.extend(root/p for p in ['plan/PLAN.json','MODELS.json','smoke/RECORDS.json',
        'local24/RECORDS.json','local24/PROFILE_SUMMARY.json','local24/MODEL.json','VISUAL_NOTES.json'])
    receipt={'scope':'Fixed-crop naming only. No end-to-end speed or GT accuracy claim.',
        'requested_luna_unavailable':read(root/'smoke/RECORDS.json'),
        'baseline':baseline,'profiles':profiles,
        'crop_selection':plan['selection'],
        'crops':[{k:c[k] for k in ('id','scene','sha256','frame_id','mask_id')} for c in plan['crops']],
        'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    receipt['instrumentation_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__),Path(__file__).with_name('benchmark_cloud_vlm.py'))}
    receipt['api_memory_snapshot']=read(root/'concurrent4/MEMORY_SNAPSHOT.json')
    receipt['baseline_model']=read(root/'local24/MODEL.json')
    receipt['repeat_comparisons']=[]
    by_profile={p['profile']:{r['crop_id']:r for r in all_records if r['profile']==p['profile']} for p in profiles}
    for effort in ('none','high'):
        name='deepseek_flash_'+effort
        if name not in by_profile or name+'_c4' not in by_profile:
            continue
        left,right=by_profile[name],by_profile[name+'_c4']
        receipt['repeat_comparisons'].append({'effort':effort,'scope':'two runs differ in concurrency and time; no causal attribution to concurrency',
            'label_changes':[{'crop_id':i,'serial':left[i].get('label'),'c4':right[i].get('label')}
                for i in left.keys()&right.keys() if left[i].get('label')!=right[i].get('label')]})
    args.receipt.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    notes={r['id']:r for r in read(root/'VISUAL_NOTES.json')['rows']}
    profile_ids=[p['profile'] for p in profiles]
    indexed={(r['profile'],r['crop_id']):r for r in all_records}
    e=lambda s:html.escape(str(s))
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>VLM 同输入对照</title>',
        '<style>body{font:14px system-ui;margin:28px;color:#182029}table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:9px;vertical-align:top;min-width:120px}th{position:sticky;top:0;background:#eef3f7}img{max-width:160px;max-height:150px}small{color:#657280}.bad{background:#ffe6e3}.unknown{background:#fff3d6}.wrap{overflow:auto}h1{font-size:24px}td{max-width:220px;overflow-wrap:anywhere}</style>',
        '<h1>24 张固定裁图：VLM 名称与接口状态</h1>',
        '<p>来自 3 个 ScanNet 场景窗口及 2 段 Orbbec 录制。颜色仅表示协议失败或 unknown，不表示正确率。视觉备注是助手抽查，不是人工 GT。GPT 经用户提供的中转，DeepSeek 使用官方接口。</p>',
        '<div class="wrap"><table><tr><th>裁图与视觉备注</th><th>本地 Qwen</th>']
    parts.extend('<th>'+e(p)+'</th>' for p in profile_ids)
    parts.append('</tr>')
    for crop in plan['crops']:
        i=crop['id'];image=root/'crops'/f'{i:02}.png'
        assert hashlib.sha256(image.read_bytes()).hexdigest()==crop['sha256']
        uri='data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()
        parts.append(f'<tr><td><b>ID {i:02}</b> {e(crop["scene"])}<br><img src="{uri}"><br><small>{e(notes[i]["category"])} ({e(notes[i]["confidence"])})</small></td>')
        old=local_by_id[i]
        parts.append(f'<td>{e(old["label"])}<br><small>{old["request_seconds"]:.3f} s</small></td>')
        for profile in profile_ids:
            row=indexed.get((profile,i))
            if not row:
                parts.append('<td>未测</td>');continue
            valid=row['ok'] and row.get('completed') and row.get('valid')
            css='bad' if not valid else 'unknown' if row['label']=='unknown' else ''
            label=row.get('label') if valid else '请求失败' if not row['ok'] else '截断/格式不合格'
            detail=row.get('error') or row.get('finish_reason') or row.get('status') or ''
            parts.append(f'<td class="{css}">{e(label)}<br><small>{row["total_seconds"]:.3f} s · {e(detail)}</small></td>')
        parts.append('</tr>')
    parts.append('</table></div></html>')
    (root/'review.html').write_text(''.join(parts))
    print(json.dumps({'profiles':len(profiles),'crops':len(local),'receipt':str(args.receipt),'review':str(root/'review.html')}))


if __name__=='__main__':
    main()
