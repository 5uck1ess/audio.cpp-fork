"""Which reference codes are right? Decode the port's and audio.cpp's codes for the same wav and score each
against the wav. Uses only the port's codec (no 38 GB model).

  venvs/drama3/bin/python -I codec_check.py --port-root <ComfyUI> --weights <models/drama3> \
      --ref-wav chris_hemsworth_15s.wav --take take-p1 --cpp cpp-q2k
"""
import argparse
import contextlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--port-root", type=Path, required=True)
p.add_argument("--weights", type=Path, required=True)
p.add_argument("--ref-wav", type=Path, required=True)
p.add_argument("--take", type=Path, required=True)
p.add_argument("--cpp", type=Path, required=True)
args = p.parse_args()

sys.path.insert(0, str(args.port_root))
with contextlib.redirect_stdout(sys.stderr):
    import soundfile as sf
    import torch
    port_dir = args.port_root / "custom_nodes" / "ComfyUI-Drama3"
    spec = importlib.util.spec_from_file_location("drama3_port", port_dir / "__init__.py",
                                                  submodule_search_locations=[str(port_dir)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["drama3_port"] = pkg
    spec.loader.exec_module(pkg)
    from drama3_port.drama3.loader import load_codec
    import comfy.model_management

forced = json.loads((args.take / "forced.json").read_text(encoding="utf-8"))
port_codes = np.array(forced["reference_codes"])  # [10, frames]
frames = port_codes.shape[1]
port_rows = np.array(forced["prompt_rows"])
own_rows = np.array(json.loads((args.cpp / "own_prompt.json").read_text()))
start = next(i for i in range(len(port_rows)) if (port_rows[i:i + frames, 1:].T == port_codes).all())
cpp_codes = own_rows[start:start + frames, 1:].T

wav, sr = sf.read(str(args.ref_wav), dtype="float32", always_2d=True)
mono = wav.mean(1)


def mel_db(x):
    spec = torch.stft(torch.from_numpy(x), 2048, 512, window=torch.hann_window(2048), return_complex=True).abs() ** 2
    freqs = torch.linspace(0, sr / 2, spec.shape[0])
    mel = 2595 * torch.log10(1 + freqs / 700)
    edges = torch.linspace(0, mel.max(), 82)
    bands = torch.stack([spec[(mel >= edges[i]) & (mel < edges[i + 2])].sum(0) for i in range(80)])
    return 10 * torch.log10(bands + 1e-10)


ref_mel = mel_db(mono)
with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
    codec = load_codec(str(args.weights / "drama3_codec.safetensors"))
    comfy.model_management.load_models_gpu([codec.patcher])
    device = comfy.model_management.get_torch_device()
    local = codec.model.encode(torch.from_numpy(mono)[None].to(device))[0].cpu().numpy()
    out = {"frames": frames, "prompt_start_row": start,
           "local_port_encode_vs_windows_port_codes_match": float((local == port_codes).mean()),
           "cpp_vs_port_codes_match": float((cpp_codes == port_codes).mean())}
    for name, codes in (("port_codes", port_codes), ("cpp_codes", cpp_codes)):
        audio = codec.model.decode(torch.from_numpy(codes)[None].to(device), torch.float32)[0].float().cpu().numpy()
        audio = audio[:len(mono)]
        m = mel_db(audio)
        n = min(m.shape[1], ref_mel.shape[1])
        err = mono[:len(audio)] - audio
        out[name] = {"mel_l1_db": float((m[:, :n] - ref_mel[:, :n]).abs().mean()),
                     "snr_db": float(10 * np.log10((mono[:len(audio)] ** 2).sum() / (err ** 2).sum()))}
        sf.write(str(args.cpp.parent / f"decoded_{name}.wav"), audio, sr)
print(json.dumps(out, indent=2))
