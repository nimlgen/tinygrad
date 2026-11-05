#!/usr/bin/env python3
# he.py make data.jsonl.gz out.jsonl | he.py score data.jsonl.gz replies.jsonl
import gzip, json, os, re, subprocess, sys, tempfile

def make(data, out):
  with open(out, "w") as f:
    for p in map(json.loads, gzip.open(data, "rt")):
      msg = ("Complete the following Python function. Reply with the complete function, including the imports it needs, in one "
             "```python code block.\n\n```python\n" + p["prompt"] + "```")
      f.write(json.dumps({"id": p["task_id"], "messages": [{"role": "user", "content": msg}]}) + "\n")

def program(p, reply):
  reply = reply.split("<|close|>")[0]  # the response channel only
  blocks = re.findall(r"```(?:python|py)?\n(.*?)```", reply, re.DOTALL)
  code = max(blocks, key=len) if blocks else reply
  src = p["prompt"] + "\n" + code if re.search(rf"def\s+{re.escape(p['entry_point'])}\s*\(", code) else p["prompt"] + code
  return src + "\n\n" + p["test"] + f"\ncheck({p['entry_point']})\n"

def score(data, replies):
  probs = {p["task_id"]: p for p in map(json.loads, gzip.open(data, "rt"))}
  got = {r["id"]: r for r in map(json.loads, open(replies))}
  ok = 0
  for tid, r in got.items():
    with tempfile.TemporaryDirectory() as d:
      path = os.path.join(d, "t.py")
      open(path, "w").write(program(probs[tid], r["reply"]))
      try: passed = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, timeout=20).returncode == 0
      except subprocess.TimeoutExpired: passed = False
    ok += passed
    if not passed: print("FAIL", tid)
  toks = [r["tokens"] for r in got.values()]
  print(f"pass@1 {ok}/{len(got)} = {ok / max(len(got), 1):.1%} (of {len(probs)}), mean {sum(toks) / max(len(toks), 1):.0f} tokens, "
        f"median {sorted(r['tok_s'] for r in got.values())[len(got) // 2]:.1f} tok/s")

if __name__ == "__main__": {"make": make, "score": score}[sys.argv[1]](*sys.argv[2:])
