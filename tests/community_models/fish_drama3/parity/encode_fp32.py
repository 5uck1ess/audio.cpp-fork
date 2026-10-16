"""Is audio.cpp's codec mismatch drift or a bug? Encode the reference with the port's codec in strict FP32
(CPU, and CUDA with TF32 off) and cross-compare every encoder's codes.

  venvs/drama3/bin/python -I encode_fp32.py <ComfyUI root> <drama3_codec.safetensors> <ref.wav> <parity dir>
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
    port_dir = root / "custom_nodes" / "ComfyUI-Drama3"
    spec = importlib.util.spec_from_file_location("drama3_port", port_dir / "__init__.py",
                                                  submodule_search_locations=[str(port_dir)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["drama3_port"] = pkg
    spec.loader.exec_module(pkg)
    from drama3_port.drama3.codec import Drama3Codec
    import comfy.ops
    import comfy.utils

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
audio, sr = sf.read(wav, dtype="float32", always_2d=True)
mono = torch.from_numpy(audio.mean(1))[None]
state = comfy.utils.load_torch_file(weights)

codes = {}
for device in ("cpu", "cuda"):
    with contextlib.redirect_stdout(sys.stderr):
        model = Drama3Codec(dtype=torch.float32, device=device, operations=comfy.ops.disable_weight_init)
        model.load_state_dict(state, strict=True)
        model.eval()
    with torch.inference_mode():
        codes[device] = model.encode(mono.to(device))[0].cpu().numpy()
    np.save(outdir / f"port_fp32_{device}.npy", codes[device])
    model = None
    torch.cuda.empty_cache()

forced = json.loads((outdir / "take-p1" / "forced.json").read_text(encoding="utf-8"))
windows = np.array(forced["reference_codes"])


def own_codes(directory):
    rows = np.array(json.loads((outdir / directory / "own_prompt.json").read_text()))
    return rows[77:77 + windows.shape[1], 1:].T


sets = {"port_windows_cuda": windows, "port_fp32_cpu": codes["cpu"], "port_fp32_cuda_noTF32": codes["cuda"],
        "cpp_cpu_fixed": own_codes("cpp-q2k-fixcpu"), "cpp_cuda": own_codes("cpp-q2k-gpucodec")}
names = list(sets)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        x, y = sets[a], sets[b]
        print(f"{a:22s} vs {b:22s} all-book {float((x == y).all(0).mean()):.3f} per-book",
              [round(float((x[k] == y[k]).mean()), 3) for k in range(10)])
