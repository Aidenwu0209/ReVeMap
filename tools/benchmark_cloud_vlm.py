"""Bounded image-naming API experiments; credentials are read from stdin only."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import mimetypes
from pathlib import Path
import re
import statistics
import sys
import threading
import time
from urllib.parse import urlparse

import requests


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sanitized(value, credentials):
    text = str(value)
    for secret in credentials.values():
        if secret:
            text = text.replace(secret, '[redacted]')
    return re.sub(r'sk-[A-Za-z0-9_-]+', '[redacted]', text)[:600]


class Client:
    def __init__(self, credentials):
        self.credentials = credentials
        self.local = threading.local()

    def request(self, method, url, key_id, payload=None):
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query:
            raise ValueError('HTTPS endpoint without embedded credentials required')
        token = self.credentials.get(key_id)
        if not token:
            return {'ok': False, 'error': 'missing credential', 'seconds': 0.0}
        if not hasattr(self.local, 'session'):
            self.local.session = requests.Session()
        tick = time.perf_counter()
        try:
            response = self.local.session.request(method, url,
                headers={'Authorization': 'Bearer ' + token}, json=payload,
                timeout=(30, 90), allow_redirects=False)
            elapsed = time.perf_counter() - tick
            try:
                body = response.json()
            except ValueError:
                return {'ok': False, 'http_status': response.status_code,
                        'error': 'non-JSON response', 'seconds': elapsed}
            if response.status_code != 200:
                error = body.get('error', {}) if isinstance(body, dict) else {}
                return {'ok': False, 'http_status': response.status_code,
                        'error': sanitized(error, self.credentials), 'seconds': elapsed}
            return {'ok': True, 'http_status': response.status_code,
                    'body': body, 'seconds': elapsed}
        except requests.RequestException as exc:
            return {'ok': False, 'error': type(exc).__name__,
                    'error_detail': sanitized(str(exc), self.credentials),
                    'seconds': time.perf_counter() - tick}

    def infer(self, profile, crop, prompt):
        tick = time.perf_counter()
        image = Path(crop['file'])
        data = image.read_bytes()
        if hashlib.sha256(data).hexdigest() != crop['sha256']:
            raise ValueError('crop digest changed')
        uri = 'data:' + (mimetypes.guess_type(image.name)[0] or 'image/png')
        uri += ';base64,' + base64.b64encode(data).decode()
        if profile['wire'] == 'responses':
            payload = {'model': profile['model'], 'store': False, 'stream': False,
                'reasoning': {'effort': profile['effort']},
                'max_output_tokens': profile.get('max_output_tokens', 4096),
                'input': [{'role': 'user', 'content': [
                    {'type': 'input_image', 'image_url': uri, 'detail': profile.get('detail', 'original')},
                    {'type': 'input_text', 'text': prompt}]}]}
        else:
            payload = {'model': profile['model'], 'stream': False,
                'max_tokens': profile.get('max_output_tokens', 4096),
                'thinking': {'type': 'disabled' if profile['effort'] == 'none' else 'enabled'},
                'messages': [{'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': uri, 'detail': profile.get('detail', 'original')}},
                    {'type': 'text', 'text': prompt}]}]}
            if profile['effort'] != 'none':
                payload['reasoning_effort'] = profile['effort']
            else:
                payload['temperature'] = 0
        response = self.request('POST', profile['endpoint'], profile['key_id'], payload)
        row = {'crop_id': crop['id'], 'scene': crop['scene'], 'sha256': crop['sha256'],
               'profile': profile['id'], **{k:v for k,v in response.items() if k != 'body'}}
        if response['ok']:
            body = response['body']
            if profile['wire'] == 'responses':
                answer = ''.join(part.get('text', '') for item in body.get('output', [])
                    if item.get('type') == 'message' for part in item.get('content', [])
                    if part.get('type') == 'output_text')
                row['status'] = body.get('status')
                row['incomplete_details'] = body.get('incomplete_details')
                row['returned_reasoning'] = body.get('reasoning')
                row['completed'] = body.get('status') == 'completed'
            else:
                choice = (body.get('choices') or [{}])[0]
                answer = choice.get('message', {}).get('content') or ''
                row['finish_reason'] = choice.get('finish_reason')
                row['completed'] = choice.get('finish_reason') == 'stop'
            answer = sanitized(answer, self.credentials)
            label = answer.strip().strip('`\"\'').strip().lower().rstrip('.').strip()
            valid = bool(re.fullmatch(r'[a-z]+(?:[ -][a-z]+){0,5}', label))
            row.update(raw_response=answer, label=label if valid else 'unknown', valid=valid,
                       response_model=body.get('model'), usage=body.get('usage'),
                       system_fingerprint=body.get('system_fingerprint'))
        row['total_seconds'] = time.perf_counter() - tick
        return row


def summarize(rows, seconds):
    durations = sorted(r['total_seconds'] for r in rows)
    successful = [r for r in rows if r['ok'] and r.get('completed') and r.get('valid')]
    usage = {}
    for row in rows:
        for k,v in (row.get('usage') or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
    return {'requests': len(rows), 'http_successes': sum(r['ok'] for r in rows),
        'completed_valid': len(successful), 'unknown': sum(r.get('label') == 'unknown' for r in successful),
        'wall_seconds': seconds, 'mean_request_seconds': statistics.mean(durations),
        'p50_request_seconds': statistics.median(durations),
        'p95_request_seconds': durations[min(len(durations)-1, int(.95*len(durations)))],
        'response_models': sorted({r['response_model'] for r in rows if r.get('response_model')}),
        'usage_numeric_totals': usage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--discover', help='HTTPS base URL for GET /models')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Never accept tokens as arguments, write them to disk or dump request bodies.
    print('READY_FOR_CREDENTIALS', flush=True)
    credentials = json.loads(sys.stdin.readline())
    client = Client(credentials)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.discover:
        result = client.request('GET', args.discover.rstrip('/')+'/models', 'gateway')
        body = result.pop('body', {})
        if result['ok']:
            result['model_ids'] = [r['id'] for r in body.get('data', []) if isinstance(r,dict) and 'id' in r]
        write(args.output/'MODELS.json', result)
        print(json.dumps(result), flush=True)
        return
    plan = read(args.plan)
    if not plan['crops'] or not plan['profiles']:
        raise ValueError('nonempty crop/profile plan required')
    write(args.output/'PLAN.json', plan)
    rows = []; summaries = []
    for profile in plan['profiles']:
        selected = plan['crops'][:profile.get('limit', len(plan['crops']))]
        tick = time.perf_counter()
        current = []
        with ThreadPoolExecutor(max_workers=profile.get('concurrency',1)) as pool:
            futures = [pool.submit(client.infer,profile,c,plan['prompt']) for c in selected]
            for future in as_completed(futures):
                current.append(future.result())
                write(args.output/'RECORDS.json', rows + sorted(current,key=lambda r:r['crop_id']))
                if len(current) % 4 == 0:
                    print(profile['id'],len(current),'/',len(selected),'finished',flush=True)
        current.sort(key=lambda r:r['crop_id'])
        summary = {'profile': profile['id'], **summarize(current, time.perf_counter()-tick)}
        rows.extend(current); summaries.append(summary)
        write(args.output/'RECORDS.json', rows)
        write(args.output/'SUMMARY.json', summaries)
        print(json.dumps(summary), flush=True)
    unchanged = all(sha(c['file']) == c['sha256'] for c in plan['crops'])
    write(args.output/'COMPLETE.json', {'finished':True,'inputs_unchanged':unchanged,
        'scope':'fixed crop API naming only; no SLAM, SAM3, fusion or GT accuracy'})


if __name__ == '__main__':
    main()
