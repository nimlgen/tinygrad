"""Install prepared K3.1 layers in an unused NVMe region; fully read back writes.

Run on tinys3 under the existing external gpu.lock. Never run alongside the engine.
The final manifest is published only after all layers and the trunk are verified.
"""
import argparse, hashlib, json, os, pathlib, time, zlib
from tinygrad.helpers import round_up
from tinygrad.runtime.support.nvme import PRPList, OP_READ, OP_WRITE
from extra.k3.store import open_store, read_manifest, write_manifest, host_io, EXPERTS, STRIPE

ROOT = pathlib.Path('/home/nimlgen/k3/k31-update')
STAGE = ROOT / 'stage'
REGION, SNAPSHOT, BASE = 5 << 39, 13 << 38, 512 << 30
BATCH = 8

def atomic_json(path, data):
  path.with_suffix('.partial').write_text(json.dumps(data))
  path.with_suffix('.partial').replace(path)

def digest(path):
  with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('action', choices=['check', 'layer', 'finalize'])
  parser.add_argument('--layer', type=int)
  args = parser.parse_args()
  assert os.environ.get('K31_LOCKED') == '1', 'Requires outer flock ~/k3/gpu.lock'
  header = json.loads((ROOT / 'store-header.json').read_text())
  assert header['source']['repo'] == 'moonshotai/Kimi-K3.1-Pro'
  assert header['source']['revision'] == '6d0a0202956796543364d666ab19c6ad1a494ee1'
  assert header['expert_type'] == 'IQ3_XXS' and header['block'] == [256, 98]
  assert header['units'] == [768] * 4 and header['n_experts'] == 896 and header['layers'] == list(range(1, 93))
  parts, total = header['part_bytes'], len(header['layers']) * header['n_experts']
  trunk_offset = round_up(REGION + EXPERTS + total * parts[0], 1 << 30)
  assert trunk_offset + (64 << 30) < SNAPSHOT
  disks, old = open_store(at=BASE)
  try:
    # All previously installed variants and the old snapshot must lie below the new region.
    manifests = [old, read_manifest(disks[0], 0), read_manifest(disks[0], 3 << 39)]
    snapshot = read_manifest(disks[0], 1 << 40)
    for g, disk in enumerate(disks):
      cap = len(disk.drives) * min(d.size for d in disk.drives)
      assert SNAPSHOT + (32 << 30) < cap - (1 << 30), (g, cap)
      for m in manifests:
        assert m['experts_off'] + len(m['layers']) * m['n_experts'] * m['part_bytes'][g] < REGION
        if m.get('trunk_bytes'): assert m['trunk_off'] + m['trunk_bytes'] < REGION
      for t in snapshot['tensors'].values():
        for gg, off, n in t['parts']:
          if gg == g: assert off + n < REGION
    print('Layout checked; drive order:', [[d.pci_dev.pcibus for d in disk.drives] for disk in disks], flush=True)
    (ROOT / 'installed').mkdir(exist_ok=True)
    if args.action == 'check': raise SystemExit(0)

    if args.action == 'layer':
      layer = args.layer
      assert layer in header['layers']
      info = json.loads((STAGE / f'L{layer:03}.json').read_text())
      assert info['header'] == header and info['layer'] == layer
      for g, (disk, size) in enumerate(zip(disks, parts)):
        src = STAGE / f'L{layer:03}.g{g}.bin'
        assert src.stat().st_size == header['n_experts'] * size and digest(src) == info['sha256'][g]
        offset = REGION + EXPERTS + header['layers'].index(layer) * header['n_experts'] * size
        mem, addrs = disk.drives[0].pci_dev.alloc_sysmem(BATCH * size)
        prps = PRPList(disk.drives[0], addrs)
        with src.open('rb', buffering=0) as f:
          for e in range(0, header['n_experts'], BATCH):
            count = min(BATCH, header['n_experts'] - e)
            n = count * size
            got = 0
            while got < n:
              amount = f.readinto(mem.mv[got:n])
              assert amount
              got += amount
            host_io(disk, OP_WRITE, offset + e * size, prps, n)
        h = hashlib.sha256()
        for e in range(0, header['n_experts'], BATCH):
          count = min(BATCH, header['n_experts'] - e)
          n = count * size
          host_io(disk, OP_READ, offset + e * size, prps, n)
          h.update(mem.mv[:n])
          for i in range(count):
            assert zlib.crc32(mem.mv[i * size:(i + 1) * size]) == info['crc'][g][e + i], (layer, g, e + i)
        assert h.hexdigest() == info['sha256'][g], (layer, g, 'disk SHA256 mismatch')
        print(f'layer {layer} GPU {g}: {header["n_experts"]} experts written and fully verified', flush=True)
      atomic_json(ROOT / 'installed' / f'L{layer:03}.json', info)

    if args.action == 'finalize':
      download = json.loads((ROOT / 'download-complete.json').read_text())
      assert download['revision'] == header['source']['revision'] and download['files'] == 112
      crcs = [[] for _ in parts]
      for layer in header['layers']:
        info = json.loads((ROOT / 'installed' / f'L{layer:03}.json').read_text())
        assert info['header'] == header and info['layer'] == layer
        for g in range(len(parts)):
          assert len(info['crc'][g]) == header['n_experts']
          crcs[g].extend(info['crc'][g])
      trunk = json.loads((ROOT / 'trunk-complete.json').read_text())
      src = ROOT / 'Kimi-K3.1-Pro-trunk-q4_k-pr.gguf'
      assert trunk['indexer_tensors'] == 120
      assert src.stat().st_size == trunk['bytes'] and digest(src) == trunk['sha256']
      assert trunk_offset + trunk['bytes'] < SNAPSHOT
      mem, addrs = disks[0].drives[0].pci_dev.alloc_sysmem(64 << 20)
      prps = PRPList(disks[0].drives[0], addrs)
      with src.open('rb', buffering=0) as f:
        off = 0
        while off < trunk['bytes']:
          n = min(len(mem.mv), trunk['bytes'] - off)
          got = 0
          while got < n:
            amount = f.readinto(mem.mv[got:n])
            assert amount
            got += amount
          aligned = round_up(n, 4096)
          mem.mv[n:aligned] = bytes(aligned - n)
          host_io(disks[0], OP_WRITE, trunk_offset + off, prps, aligned)
          off += n
      h, crc, off = hashlib.sha256(), 0, 0
      while off < trunk['bytes']:
        n = min(len(mem.mv), trunk['bytes'] - off)
        host_io(disks[0], OP_READ, trunk_offset + off, prps, round_up(n, 4096))
        h.update(mem.mv[:n])
        crc = zlib.crc32(mem.mv[:n], crc)
        off += n
      assert h.hexdigest() == trunk['sha256']
      result = {**header, 'experts_off': REGION + EXPERTS, 'stripe': STRIPE,
                'drives': [[d.pci_dev.pcibus for d in disk.drives] for disk in disks], 'crc': crcs,
                'trunk_off': trunk_offset, 'trunk_bytes': trunk['bytes'], 'trunk_name': src.name,
                'trunk_crc': crc, 'trunk_sha256': trunk['sha256'], 'snapshot_off': SNAPSHOT}
      write_manifest(disks[0], result, REGION)
      assert read_manifest(disks[0], REGION) == result
      atomic_json(ROOT / 'manifest.json', result)
      atomic_json(ROOT / 'installed-complete.json', {'revision': header['source']['revision'], 'time': time.time(),
                  'experts': total, 'store': REGION, 'snapshot': SNAPSHOT, 'trunk_sha256': trunk['sha256']})
      print('INSTALL COMPLETE: every expert part and the trunk read back and verified; manifest published', flush=True)
  finally:
    for disk in disks:
      for drive in disk.drives: drive.fini()
