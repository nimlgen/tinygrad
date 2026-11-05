import json, glob, os, time, sys
seen, job = set(), os.path.expanduser("~/k3/tb/jobs/" + (sys.argv[1] if len(sys.argv) > 1 else "q3_full"))
log = open(os.path.expanduser("~/k3/m3_tb_serve.log"))
log.seek(0, 2)
while True:
  for f in sorted(glob.glob(f"{job}/*/result.json")):
    if f in seen: continue
    r = json.load(open(f))
    if not r.get("finished_at"): continue
    seen.add(f)
    rew = (r.get("verifier_result") or {}).get("rewards", {}).get("reward")
    exc = (r.get("exception_info") or {}).get("exception_type")
    print(f"{len(seen)} {r.get('task_name')}: reward {rew}" + (f" EXC {exc}" if exc else ""), flush=True)
  for line in log:
    if any(k in line for k in ("MemoryError", "RuntimeError", "AssertionError", "Segmentation")): print("SERVER:", line.strip()[:200], flush=True)
  if os.path.exists(f"{job}/result.json") and json.load(open(f"{job}/result.json")).get("finished_at"):
    print("JOB FINISHED", flush=True)
    sys.exit(0)
  time.sleep(60)
