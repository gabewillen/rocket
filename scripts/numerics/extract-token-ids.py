#!/usr/bin/env python3
"""Extract the engine's emitted token ids (plain value list) from --tokens output.

The engine prints a "--- token ids ---" block as `idx:name` pairs. The
--draft-file loader in decode_main.cu reads one int per output position
(prompt-included not: draft_ids[i] = the value for output position i), so
this writes just the values, one per line, prompt excluded.

Input:  0:12089 | Paris|  1:13 |.| ...
Output: 12089
        13
        ...
"""
import re, sys

txt = sys.stdin.read()
i = txt.find("--- token ids")
if i < 0:
    sys.exit("no token ids section")
tail = txt[i:]
pairs = re.findall(r"(\d+):(\d+)", tail)
if not pairs:
    sys.exit("no id pairs found")
print(" ".join(v for _, v in pairs))
