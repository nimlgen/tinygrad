# the final terminal-bench score: each task's result from the latest job that finished it (jobs in order, later wins)
import glob, json, os, sys
jobs = sys.argv[1:]
res: dict[str, tuple[str, float, str | None]] = {}
for job in jobs:
  for f in glob.glob(os.path.expanduser(f"~/k3/tb/jobs/{job}/*/result.json")):
    r = json.load(open(f))
    if not r.get("finished_at") or not r.get("task_name"): continue
    rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    exc = (r.get("exception_info") or {}).get("exception_type")
    res[r["task_name"]] = (job, float(rew or 0), exc)
tasks = sorted(os.path.basename(os.path.dirname(t)) for t in glob.glob(os.path.expanduser("~/k3/tb/terminal-bench/*/task.toml")))
missing = [t for t in tasks if t not in res]
solved = sum(r[1] for r in res.values())
print(f"{len(tasks)} tasks, {len(res)} with a result, missing {missing}")
print(f"solved {solved:.0f}/{len(tasks)} = {solved / len(tasks):.1%}")
by = {}
for t, (job, rew, exc) in res.items(): by.setdefault(job, []).append(rew)
for job in jobs: print(f"  {job}: {len(by.get(job, []))} tasks, {sum(by.get(job, [])):.0f} solved")
print("timeouts:", sum(1 for r in res.values() if r[2] == "AgentTimeoutError"), "of which solved:", sum(1 for r in res.values() if r[2] == "AgentTimeoutError" and r[1] > 0))
print("other exceptions:", sorted((t, r[2]) for t, r in res.items() if r[2] and r[2] != "AgentTimeoutError"))
print("solved:", " ".join(sorted(t for t, r in res.items() if r[1] > 0)))
