# K3 on tinys3: how the numbers were made

The scripts here are the ones that ran, with their `~/k3/...` paths. Results: HumanEval 161/164 = 98.2% pass@1 (`results/he_q3.jsonl`),
Terminal-Bench 2.0 52/89 = 58.4% (`results/tb_q3.json`, per task). Both on the IQ3 experts, the fast setting (this branch's defaults) with
`TAUC=0`, `THINK=off`, greedy.

## Box

tinys3: 4x Radeon AI PRO R9700 32 GB, 16x SN8100 1 TB (4 per gpu, under the gpu's root complex), 1 socket Genoa with 2x16 GB DDR5,
Ubuntu 24.04, kernel 6.14.

One time, as root: `extra/amdpci/setup_python_cap.sh` (setcap on python3.12), `vm.nr_hugepages=512`,
`options vfio-pci disable_idle_d3=1` in /etc/modprobe.d. After every boot: `sudo rmmod amdgpu` (tb_serve.sh refuses to start without it).
`~/libamd_comgr.so` comes from a torch rocm wheel.

## Data (not in git)

| what | from | where the scripts expect it |
|---|---|---|
| Kimi-K3 checkpoint (native MXFP4, 1.5 TB) | `moonshotai/Kimi-K3` | the quant box |
| imatrix, sha256 `10de290f6f8cababc8d549fb5f8386d2ebc52d095f238d275da420659f27f692` | AtomicChat's K3 imatrix | 153:`~/k3q/imatrix.dat` |
| llama.cpp fork (quantizer, hf -> gguf converter) | `AtomicBot-ai/atomic-llama-cpp-turboquant` at `bb89354ec`, built | 153:`~/k3q/atomic-llama` |
| EAGLE-3.1 draft | `lightseekorg/kimi-k3-eagle3.1-mla` | `~/k3/eagle` |
| prefetch guess maps (181 .npy) | `extra/k3/train_guess.py` on teacher routing traces | `~/k3/guess-v1` |
| this branch | | `~/k3/tg-clean`, venv at `~/k3/venv` (numpy, gguf) |

## Weights

On the quant box, `quant/`:

```
# experts: IQ3_XXS with the imatrix, as shard 2 of a 2-shard split gguf
python3 quant_experts_stream.py <hf dir> experts-IQ3_XXS.gguf IQ3_XXS imatrix.dat <n trunk tensors>
# trunk: everything but the routed experts, bf16 -> Q4_K_M -> the names tinygrad/llm/model.py loads
python3 convert_trunk.py <hf dir> trunk-bf16.gguf bf16
atomic-llama/build/bin/llama-quantize trunk-bf16.gguf trunk-q4_k.gguf Q4_K_M
python3 extra/k3/trunk_names.py trunk-q4_k.gguf trunk-q4_k-pr.gguf    # 35211326176 bytes
```

Onto the drives (raw, layout in `extra/k3/store.py`, the IQ3 store's manifest at disk 0 offset 512 GiB):
`PYTHONPATH=. python3 extra/k3/recv.py <port>` on tinys3, then on the quant box
`python3 extra/k3/pack.py experts-IQ3_XXS.gguf trunk-q4_k-pr.gguf tinys3:<port> --units 768,768,768,768`. Without the trunk in place,
`extra/k3/extract_trunk.py` gets it back off disk 0.

## Runs

Every gpu run goes through `extra/k3/m3run.sh` (`flock ~/k3/gpu.lock`, `VFIO=1`, log in `~/k3/m3_<name>.log`).

- decode: `TG=~/k3/tg-clean bash m3run.sh code prompt_code.txt N=256 DRAFT=~/k3/eagle TAUC=0 SLOTS=6000`
- HumanEval: `python3 he.py make HumanEval.jsonl.gz he_in.jsonl`, `bash he_run.sh` (BENCH mode, resumes),
  `python3 he.py score HumanEval.jsonl.gz he_q3.jsonl`
- Terminal-Bench 2.0: harbor 0.25.0 with its `terminal-bench` dataset (89 tasks, images `alexgshaw/*:20251031`) in `~/k3/tb/terminal-bench`.
  `bash tb_serve.sh` (SERVE=8100: 32k context, 256-token chunks, SLOTS 4800), then `bash tb_harbor.sh q3_full` (terminus-2, n=1,
  temperature 0), `python3 watch.py q3_full` to follow it. Tasks that a server failure killed were rerun by name
  (`tb_harbor.sh q3_rerun3 -t <task> ...`); `python3 merge.py q3_full q3_rerun q3_rerun3` keeps each task's latest result.
  33 tasks hit the agent timeout (7 still solved): deep-context prefill is slow.

## Kimi K3.1 Pro (in progress)

`k31/`: the same pipeline for `moonshotai/Kimi-K3.1-Pro` at revision `6d0a0202956796543364d666ab19c6ad1a494ee1`. Its only architecture change
is a DSA indexer on the 24 MLA layers (top 2048 keys, 32 heads x 128, no rope, LayerNorm keys; 120 tensors, kept f16 in the trunk).

- `download.py`: every repo file, resumable, verified against the hub's hashes (`source-manifest.json`)
- `quantize.py`: the experts as per-gpu IQ3_XXS parts of each layer, with the same imatrix (`store-header.json`)
- `check_parts.py <layer>`: the parts against the K3 quantizer, byte for byte
- `trunk_pipeline.sh` (`convert_trunk.py`, `trunk_names.py`): the q4_k trunk with the indexer tensors
- `install.py check|layer --layer n|finalize`: the layers and trunk onto the drives at 2.5 TiB, past the K3 stores, every write read back

Download, quantization and the trunk are done (`*-complete.json`). install.py has not run, and the engine doesn't run the indexer yet.
