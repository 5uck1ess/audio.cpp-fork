"""Pacing of long-form Drama 3 output: pauses (silence = 20 ms RMS < 0.003, the threshold Fish's HF Space uses),
speaking rate across the take, and Whisper WER against the input text.

  uv run --with numpy --with soundfile --with openai-whisper==20250625 --with jiwer python -I pacing_score.py \
      passage.txt a.wav b.wav ...
"""
import json
import re
import sys

import jiwer
import numpy as np
import soundfile as sf
import whisper

WINDOW_S, SILENCE_RMS = 0.02, 0.003


def pauses(audio, sr):
    hop = int(WINDOW_S * sr)
    frames = audio[: len(audio) // hop * hop].reshape(-1, hop)
    quiet = np.sqrt((frames ** 2).mean(1)) < SILENCE_RMS
    runs, start = [], None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            runs.append((start, i))
            start = None
    lead = runs[0][1] * WINDOW_S if runs and runs[0][0] == 0 else 0.0
    tail = (runs[-1][1] - runs[-1][0]) * WINDOW_S if runs and runs[-1][1] == len(quiet) else 0.0
    inner = [(b - a) * WINDOW_S for a, b in runs if a != 0 and b != len(quiet)]
    return lead, tail, np.array(inner), float(quiet.mean())


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text.lower().replace("’", "'"))).strip()


text = open(sys.argv[1], encoding="utf-8").read()
model = whisper.load_model("base", device=__import__("os").environ.get("WHISPER_DEVICE"))
for path in sys.argv[2:]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    lead, tail, inner, silent = pauses(audio, sr)
    result = model.transcribe(path, language="en", word_timestamps=True)
    words = [w for s in result["segments"] for w in s["words"]]
    # Speaking rate in 10 s windows of speech, to show uneven pace across the take.
    rates = []
    if words:
        edges = np.arange(0, words[-1]["end"] + 10, 10)
        for a, b in zip(edges[:-1], edges[1:]):
            n = sum(1 for w in words if a <= w["start"] < b)
            if n:
                rates.append(n / 10 * 60)
    print(json.dumps({
        "file": path.split("/")[-1], "seconds": round(len(audio) / sr, 1),
        "wer": round(jiwer.wer(norm(text), norm(result["text"])), 3),
        "lead_s": round(lead, 2), "tail_s": round(tail, 2), "silent_frac": round(silent, 3),
        "pauses_over_0.75s": int((inner > 0.75).sum()), "pauses_over_1.5s": int((inner > 1.5).sum()),
        "longest_pause_s": round(float(inner.max()) if len(inner) else 0.0, 2),
        "wpm_median": round(float(np.median(rates)), 1) if rates else 0,
        "wpm_cv": round(float(np.std(rates) / np.mean(rates)), 3) if rates else 0,
    }))
