"""Where are the long pauses in a chunked take? Needs a log from FISH_DRAMA3_CHUNK_LOG=1 (per-chunk sample counts).
Prints, per chunk, its leading/trailing silence, its speaking rate, and the gap at each chunk join.

  python -I chunk_pauses.py take.wav take.log
"""
import re
import sys

import numpy as np
import soundfile as sf

audio, sr = sf.read(sys.argv[1], dtype="float32", always_2d=True)
audio = audio.mean(1)
chunks = [(int(m[0]), m[1]) for m in re.findall(r"drama3_chunk \d+ samples=(\d+) text=(.*)", open(sys.argv[2]).read())]
hop = int(0.02 * sr)


def edge_silence(x):
    rms = np.sqrt((x[: len(x) // hop * hop].reshape(-1, hop) ** 2).mean(1))
    loud = np.nonzero(rms >= 0.003)[0]
    if not len(loud):
        return len(rms) * 0.02, 0.0
    return loud[0] * 0.02, (len(rms) - 1 - loud[-1]) * 0.02


def inner_pauses(x):
    rms = np.sqrt((x[: len(x) // hop * hop].reshape(-1, hop) ** 2).mean(1)) < 0.003
    loud = np.nonzero(~rms)[0]
    if not len(loud):
        return []
    q = rms[loud[0]:loud[-1] + 1]
    runs, n = [], 0
    for v in q:
        if v:
            n += 1
        elif n:
            runs.append(n * 0.02)
            n = 0
    return runs


pos, prev_tail, joins, inner = 0, None, [], []
for i, (n, text) in enumerate(chunks):
    seg = audio[pos:pos + n]
    pos += n
    lead, tail = edge_silence(seg)
    words = len(text.split())
    speech = n / sr - lead - tail
    if prev_tail is not None:
        joins.append(prev_tail + lead)
    inner += inner_pauses(seg)
    print(f"chunk {i:2d} {n / sr:5.1f}s lead {lead:4.2f} tail {tail:4.2f} wpm {words / speech * 60:5.0f}  {text[:60]!r}")
    prev_tail = tail
joins, inner = np.array(joins), np.array(inner)
print(f"joins: n={len(joins)} median {np.median(joins):.2f}s max {joins.max():.2f}s >0.75s: {(joins > 0.75).sum()}")
print(f"pauses inside chunks: n={len(inner)} >0.75s: {(inner > 0.75).sum()} max {inner.max():.2f}s")
