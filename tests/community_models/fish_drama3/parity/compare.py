"""Compare audio.cpp fish_drama3 logits against a recorded port take (port_dump.py) under teacher forcing.

  python compare.py --take take/ --cpp cpp_dump/ [--cpp-b cpp_dump2/]

take/   forced.json + port_logits.npz from port_dump.py
cpp/    own_prompt.json + main.f32 + tower.f32 from FISH_DRAMA3_FORCE=take/forced.json FISH_DRAMA3_DUMP=cpp/
--cpp-b a second audio.cpp dump (another quant or a repeat run) compared against --cpp as a control.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def kl(p_logits, q_logits):
    lp, lq = log_softmax(p_logits.astype(np.float64)), log_softmax(q_logits.astype(np.float64))
    return (np.exp(lp) * (lp - lq)).sum(-1)


def load_cpp(directory, steps, books, vocab):
    main = np.fromfile(directory / "main.f32", dtype=np.float32)
    tower = np.fromfile(directory / "tower.f32", dtype=np.float32)
    candidates = main.size // (steps + 1)
    if main.size != (steps + 1) * candidates:
        raise SystemExit(f"{directory}/main.f32 holds {main.size} floats, not a multiple of {steps + 1} steps")
    if tower.size != steps * books * vocab:
        raise SystemExit(f"{directory}/tower.f32 holds {tower.size} floats, expected {steps}x{books}x{vocab}")
    return main.reshape(steps + 1, candidates), tower.reshape(steps, books, vocab)


def prompt_report(own, forced):
    own, forced = np.array(own), np.array(forced)
    if own.shape == forced.shape and (own == forced).all():
        return {"identical": True, "rows": len(own)}
    report = {"identical": False, "own_rows": len(own), "port_rows": len(forced)}
    n = min(len(own), len(forced))
    diff = np.nonzero((own[:n] != forced[:n]).any(-1))[0]
    if len(diff):
        first = int(diff[0])
        report.update(first_diff_row=first, differing_rows=int(len(diff)),
                      text_rows_differing=int((forced[diff, 1:] == 0).all(-1).sum()),
                      own_row=own[first].tolist(), port_row=forced[first].tolist())
    return report


def compare(name, a_main, a_tower, b_main, b_tower, forced_main):
    steps = len(forced_main)
    k = kl(a_main, b_main)
    frames_k = k[:steps]
    bins = np.array_split(frames_k, min(8, steps))
    agree = (a_main[:steps].argmax(-1) == b_main[:steps].argmax(-1)).mean()
    lp_a, lp_b = log_softmax(a_main.astype(np.float64)), log_softmax(b_main.astype(np.float64))
    idx = np.arange(steps)
    tk = kl(a_tower, b_tower)
    return {
        "pair": name,
        "main_kl_mean": float(k.mean()), "main_kl_max": float(k.max()), "main_kl_argmax_step": int(k.argmax()),
        "main_kl_first_half": float(k[: len(k) // 2].mean()), "main_kl_second_half": float(k[len(k) // 2:].mean()),
        "main_kl_frames_only": float(frames_k.mean()), "main_kl_eos_step": float(k[-1]),
        "main_kl_by_time_bin": [round(float(b.mean()), 5) for b in bins],
        "main_top1_agreement": float(agree),
        "forced_logprob_mean_a": float(lp_a[idx, forced_main].mean()),
        "forced_logprob_mean_b": float(lp_b[idx, forced_main].mean()),
        "p_eos_last_a": float(np.exp(lp_a[-1, 0])), "p_eos_last_b": float(np.exp(lp_b[-1, 0])),
        "tower_kl_mean_per_book": [round(float(x), 5) for x in tk.mean(0)],
        "tower_kl_max": float(tk.max()),
    }


p = argparse.ArgumentParser()
p.add_argument("--take", type=Path, required=True)
p.add_argument("--cpp", type=Path, required=True)
p.add_argument("--cpp-b", type=Path, default=None)
args = p.parse_args()

forced = json.loads((args.take / "forced.json").read_text(encoding="utf-8"))
port = np.load(args.take / "port_logits.npz")
port_main, port_tower = port["main"], port["tower"].astype(np.float32)
frames = np.array(forced["frames"])
steps = len(frames)
tower_steps = port["tower_steps"] if "tower_steps" in port.files else np.arange(steps)
_, books, vocab = port_tower.shape
if len(port_main) != steps + 1 or len(tower_steps) != len(port_tower):
    raise SystemExit("port dump is inconsistent with forced.json")
forced_main = frames[:, 1] + 1  # candidate index: 0 = end of speech, 1 + c = semantic code c

out = {"steps": steps, "prompt": prompt_report(json.loads((args.cpp / "own_prompt.json").read_text()),
                                                forced["prompt_rows"])}
cpp_main, cpp_tower = load_cpp(args.cpp, steps, books, vocab)
cpp_tower = cpp_tower[tower_steps]
out["port_vs_cpp"] = compare("port||cpp", port_main, port_tower, cpp_main, cpp_tower, forced_main)
out["port_vs_port"] = compare("port||port (control, expect 0)", port_main, port_tower, port_main, port_tower, forced_main)
if args.cpp_b:
    b_main, b_tower = load_cpp(args.cpp_b, steps, books, vocab)
    b_tower = b_tower[tower_steps]
    out["port_vs_cpp_b"] = compare("port||cpp_b", port_main, port_tower, b_main, b_tower, forced_main)
    out["cpp_vs_cpp_b"] = compare("cpp||cpp_b", cpp_main, cpp_tower, b_main, b_tower, forced_main)
print(json.dumps(out, indent=2))
