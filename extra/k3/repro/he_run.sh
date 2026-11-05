#!/bin/bash
# humaneval on the q3 experts, the fast setting with TAUC=0 (resumes from he_q3.jsonl)
cd ~/k3 && TG=$HOME/k3/tg-clean TMO=0 bash m3run.sh he_q3 - SLOTS=6000 DRAFT=$HOME/k3/eagle TAUC=0 THINK=off MAXTOK=1024 BENCH=$HOME/k3/he_in.jsonl:$HOME/k3/he_q3.jsonl
