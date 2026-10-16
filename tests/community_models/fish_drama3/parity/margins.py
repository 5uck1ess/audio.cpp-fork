"""Drift or bug? For each audio.cpp code that differs from strict-FP32 port codes, measure on the port's
own (FP32) residual path how far audio.cpp's pick is from the best one: its rank and its cosine margin.
Only frames whose earlier books all agree are scored, so the residual inputs are the same.

  venvs/drama3/bin/python -I margins.py <ComfyUI root> <drama3_codec.safetensors> <ref.wav> <parity dir> <cpp dump>...
"""
import contextlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

root, weights, wav, outdir = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
sys.path.insert(0, str(root))
with contextlib.redirect_stdout(sys.stderr):
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    port_dir = root / "custom_nodes" / "ComfyUI-Drama3"
    spec = importlib.util.spec_from_file_location("drama3_port", port_dir / "__init__.py",
                                                  submodule_search_locations=[str(port_dir)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["drama3_port"] = pkg
    spec.loader.exec_module(pkg)
    from drama3_port.drama3.codec import Drama3Codec
    import comfy.ops
    import comfy.utils

audio, sr = sf.read(wav, dtype="float32", always_2d=True)
mono = torch.from_numpy(audio.mean(1))[None]
with contextlib.redirect_stdout(sys.stderr):
    model = Drama3Codec(dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init)
    model.load_state_dict(comfy.utils.load_torch_file(weights), strict=True)
    model.eval()

q = model.quantizer
stages = [q.semantic_quantizer.quantizers[0]] + list(q.quantizer.quantizers)
sims, port_codes = [], []
with torch.inference_mode():
    x = mono[:, None]
    x = x if x.shape[-1] % 2048 == 0 else F.pad(x, (0, -x.shape[-1] % 2048))
    residual = q.pre_module(q.downsample(model.encoder(x)))
    for stage in stages:
        latents = stage.in_proj(residual)
        flat = F.normalize(latents.transpose(1, 2).reshape(-1, latents.shape[1]).float())
        s = flat @ F.normalize(stage.codebook.weight.float()).T  # [frames, entries]
        code = s.argmax(-1)
        sims.append(s.numpy())
        port_codes.append(code.numpy())
        residual = residual - stage.decode(code[None], residual.dtype)
port_codes = np.stack(port_codes)  # [10, frames]

for name in sys.argv[5:]:
    rows = np.array(json.loads((outdir / name / "own_prompt.json").read_text()))
    cpp = rows[77:77 + port_codes.shape[1], 1:].T
    print(f"== {name}: all-book match {float((cpp == port_codes).all(0).mean()):.3f}")
    prefix_ok = np.ones(port_codes.shape[1], bool)
    for book in range(10):
        s, frames = sims[book], np.nonzero(prefix_ok)[0]
        mism = frames[cpp[book, frames] != port_codes[book, frames]]
        top = s[mism].max(-1)
        mine = s[mism, cpp[book, mism]]
        ranks = (s[mism] > mine[:, None]).sum(-1) + 1
        gaps = np.sort(s[frames], -1)
        typical = float(np.median(gaps[:, -1] - gaps[:, -2])) if len(frames) else 0.0
        print(f"book {book}: scored {len(frames):3d} mismatched {len(mism):3d}"
              + (f" | cpp rank median {int(np.median(ranks))} max {int(ranks.max())}"
                 f" | cosine margin median {float(np.median(top - mine)):.2e} max {float((top - mine).max()):.2e}"
                 if len(mism) else "")
              + f" | typical top1-top2 gap {typical:.2e}")
        prefix_ok &= cpp[book] == port_codes[book]
