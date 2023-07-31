#!/usr/bin/env python3

import sys
import subprocess
import os
import re
from collections import defaultdict
import colorama

old_sum = defaultdict(float)
old_cnt = defaultdict(int)
old_fwd = []
new_sum = defaultdict(float)
new_cnt = defaultdict(int)
new_fwd = []
def switch_to_prev_commit():
    subprocess.run(["git", "stash"])
    subprocess.run(["git", "checkout", "HEAD^"])

def switch_to_head_commit():
    subprocess.run(["git", "switch", "-"])
    subprocess.run(["git", "stash", "pop"])

def launch_other_script_with_same_args(dsum, dcnt, dfwd, old, runs=1):
    # Get the arguments of the current script
    current_args = sys.argv[1:]

    # Prepare the command to launch the other script with the same arguments
    args = ["python3"] + current_args

    # Launch the other script and capture its output
    os.environ["DEBUG"] = "2"
    if old: switch_to_prev_commit()
    for _ in range(runs):
        process = subprocess.Popen(args, stdout=subprocess.PIPE, text=True)

        # Wait for the process to complete and get the output
        stdout, _ = process.communicate()

        # Print the output of the other script
        for line in stdout.splitlines():
            if not line.startswith("***"):
                continue
            parts = line.split()
            name = parts[2]
            spd = None
            match = re.search(r'tm\s+(\d+\.\d+)us/', line)
            if match:
                spd = float(match.group(1))
            dsum[name] += spd
            dcnt[name] += 1
            dfwd.append(name)
    if old: switch_to_head_commit()

def print_pretty_float_with_percentage(num1, num2):
    colorama.init()  # Initialize colorama

    # Print the number with colors
    percentage_diff = ((num2 - num1) / num1) * 100
    color = colorama.Fore.GREEN if percentage_diff <= 0 else colorama.Fore.RED
    pretty_diff = f"{percentage_diff:.2f}%"
    ss = f"{num1:.2f} -> {num2:.2f} {color}({pretty_diff}){colorama.Fore.WHITE}"

    # Reset colorama settings
    colorama.deinit()
    return ss

if __name__ == "__main__":
    launch_other_script_with_same_args(new_sum, new_cnt, new_fwd, 0)
    launch_other_script_with_same_args(old_sum, old_cnt, old_fwd, 1)
    assert len(old_fwd) == len(new_fwd)
    was = set()
    ntotal = 0
    ototal = 0
    ntotal2 = 0
    ototal2 = 0
    for i in range(len(old_fwd)):
        if old_fwd[i] in was:
            continue
        oldt = old_sum[old_fwd[i]] / old_cnt[old_fwd[i]]
        newt = new_sum[new_fwd[i]] / new_cnt[new_fwd[i]]
        ntotal += new_sum[new_fwd[i]] / new_cnt[new_fwd[i]]
        ototal += old_sum[old_fwd[i]] / old_cnt[old_fwd[i]]
        # ototal2 += old_cnt[new_fwd[i]]
        # ntotal2 += new_cnt[old_fwd[i]]
        print(old_fwd[i], "->", new_fwd[i], "|", print_pretty_float_with_percentage(oldt, newt))
        # was.add(old_fwd[i])
    # assert ntotal2 == ototal2
    print(print_pretty_float_with_percentage(ototal, ntotal))