import concurrent.futures, hashlib, json, os, pathlib, shutil, threading, time, urllib.request

ROOT = pathlib.Path('/raid/downloads/nimlgen/Kimi-K3.1-Pro')
STATE = pathlib.Path('/home/nimlgen/k31-update')
manifest = json.loads((STATE / 'source-manifest.json').read_text())
revision = manifest['sha']
lock = threading.Lock()
counts = {}

def report(message):
  with lock:
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message, flush=True)

def download(entry):
  name, size = entry['rfilename'], entry['size']
  final = ROOT / name
  partial = final.with_name(final.name + '.partial')
  final.parent.mkdir(parents=True, exist_ok=True)
  expected = entry.get('lfs', {}).get('sha256')
  def new_hash():
    h = hashlib.sha256() if expected else hashlib.sha1()
    if not expected: h.update(f'blob {size}\0'.encode())
    return h
  for attempt in range(20):
    try:
      h = new_hash()
      existing = final if final.exists() else partial
      offset = 0
      if existing.exists():
        with existing.open('rb') as f:
          while b := f.read(8 << 20):
            h.update(b)
            offset += len(b)
        if offset > size: raise ValueError(f'{name}: oversized existing file')
      counts[name] = offset
      while offset < size:
        if final.exists(): raise ValueError(f'{name}: truncated final file')
        ranged = size > 64 << 20 or offset > 0
        end = min(offset + (256 << 20), size) - 1 if ranged else size - 1
        begin, chunk_start = offset, time.monotonic()
        req = urllib.request.Request(
          f'https://huggingface.co/moonshotai/Kimi-K3.1-Pro/resolve/{revision}/{name}',
          headers={'Range': f'bytes={offset}-{end}'} if ranged else {})
        with urllib.request.urlopen(req, timeout=45) as response:
          if ranged and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
            raise ValueError(f'{name}: server did not honor resume offset {offset}')
          with partial.open('ab') as f:
            while offset <= end:
              b = response.read(min(8 << 20, end + 1 - offset))
              if not b: raise OSError(f'{name}: response ended at {offset}, expected {end + 1}')
              if shutil.disk_usage(ROOT).free < 20 << 30: raise OSError('Less than 20 GiB free; stopping before disk fills')
              f.write(b)
              h.update(b)
              offset += len(b)
              counts[name] = offset
              elapsed = time.monotonic() - chunk_start
              if elapsed > 30 and (offset - begin) / elapsed < 2 << 20:
                raise TimeoutError(f'{name}: renewing slow connection at {offset}')
            f.flush()
            os.fsync(f.fileno())
      digest = expected or entry['blobId']
      if offset != size: raise ValueError(f'{name}: size {offset}, expected {size}')
      if h.hexdigest() != digest: raise ValueError(f'{name}: checksum mismatch (file preserved for investigation)')
      if not final.exists(): partial.rename(final)
      counts[name] = size
      with lock:
        with (STATE / 'verified-files.jsonl').open('a') as f:
          f.write(json.dumps({'name': name, 'bytes': size, 'hash': digest, 'revision': revision}) + '\n')
      report(f'verified {name} ({size / 1e9:.3f} GB)')
      return
    except Exception as e:
      report(f'attempt {attempt + 1}: {name}: {e}')
      if isinstance(e, (ValueError, OSError)) and ('checksum' in str(e) or 'free' in str(e) or 'existing' in str(e)): raise
      if attempt == 19: raise
      time.sleep(min(30, 2 ** attempt))

if __name__ == '__main__':
  ROOT.mkdir(parents=True, exist_ok=True)
  total = sum(e['size'] for e in manifest['siblings'])
  existing = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
  assert shutil.disk_usage(ROOT).free >= total - existing + (20 << 30), 'Not enough free disk space'
  (ROOT / 'download-manifest.json').write_text(json.dumps(manifest, indent=2))
  report(f'starting {revision}, {total / 1e9:.3f} GB')
  entries = sorted(manifest['siblings'], key=lambda e: (e['rfilename'].endswith('.safetensors'), e['rfilename']))
  started = time.monotonic()
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    jobs = [pool.submit(download, e) for e in entries]
    while any(not j.done() for j in jobs):
      concurrent.futures.wait(jobs, timeout=30)
      done = sum(counts.values())
      report(f'progress {done / 1e9:.3f}/{total / 1e9:.3f} GB, {done / max(1, time.monotonic()-started) / 1e6:.1f} MB/s')
    for job in jobs: job.result()
  (STATE / 'download-complete.json').write_text(json.dumps({'revision': revision, 'files': len(entries), 'bytes': total}))
  report('DOWNLOAD COMPLETE: all repository files verified')
