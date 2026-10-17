"""Record one Drama 3 take from the ComfyUI-Drama3 port: prompt rows, frames, and every logit it sampled from.

The port matched Fish's fish_omni reference under teacher forcing (mean KL 0.0041 on the main token), so its
take is the trajectory that audio.cpp's fish_drama3 is forced along (see forced_dump in generator.cpp).

Run with the tts-bench drama3 venv:
  venvs/drama3/bin/python port_dump.py --port-root <ComfyUI dir> --weights <models/drama3> \
      --ref-wav chris_hemsworth_15s.wav --text "..." --out take/
Writes take/forced.json (prompt rows, frames) and take/port_logits.npz (main [steps + 1, 4097] f32, tower
[len(tower_steps), books, vocab] for the steps listed in tower_steps). Step i's logits predict frame i; the last
main row predicts end of speech. Long takes: --text-file, --top-k 4097 (no cap, as Fish's HF Space does; top_k=30
can stall in silence), and --tower-every 16 --tower-f16 to keep the file under GitHub's 100 MB limit.
"""
import argparse
import contextlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--port-root", type=Path, required=True, help="ComfyUI checkout holding custom_nodes/ComfyUI-Drama3")
p.add_argument("--weights", type=Path, required=True)
p.add_argument("--ref-wav", type=Path, required=True)
text = p.add_mutually_exclusive_group(required=True)
text.add_argument("--text")
text.add_argument("--text-file", type=Path)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max-frames", type=int, default=600)
p.add_argument("--expert-cache-gb", type=float, default=8.0)
p.add_argument("--top-k", type=int, default=30, help="main/tower sampling top-k; 4097 = no cap")
p.add_argument("--tower-every", type=int, default=1, help="keep tower logits for every Nth step only")
p.add_argument("--tower-f16", action="store_true", help="store kept tower logits as float16")
args = p.parse_args()
if args.text_file:
    args.text = args.text_file.read_text(encoding="utf-8").strip()

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
    from drama3_port import nodes
    from drama3_port.drama3.config import RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, RAS_WINDOW
    from drama3_port.drama3.generate import embed_rows, sample_top_k_top_p
    from drama3_port.drama3.loader import load_codec, load_drama3
    from drama3_port.drama3.prompt import build_prompt
    import comfy.model_management

TEMPERATURE, TOP_P, TOP_K = 1.0, 0.9, args.top_k


def record_take(drama3, prompt):
    """drama3/generate.py's generate(), unchanged except that every logit handed to a sampler is kept."""
    config = drama3.config
    backbone = drama3.model.model
    dtype = drama3.model.dtype
    layers, devices, norm, audio_head, tower = drama3.parts()
    first, last = devices[0], devices[-1]
    rows = prompt.to(last)
    prompt_rows = len(rows)
    capacity = prompt_rows + args.max_frames + 1
    caches = backbone.init_caches(capacity, devices, dtype)
    tower_caches = tower.init_caches(capacity, last, dtype)
    generator = torch.Generator(device=last).manual_seed(args.seed)
    candidates = torch.cat((torch.tensor([config.eos_token_id], device=last),
                            torch.arange(config.semantic_start_token_id, config.semantic_end_token_id + 1, device=last)))
    main_logits, tower_logits, tower_steps, step_tower = [], [], [], []

    def run(frames):
        text = embed_rows(backbone.embed_tokens, frames[:, 0].to(first), dtype)
        x = backbone.run_layers(tower.embed_backbone(frames, text)[None], caches, layers, devices)
        with comfy.model_management.cuda_device_context(last):
            return norm(x)

    def sample(logits):
        return sample_top_k_top_p(logits, TEMPERATURE, TOP_K, TOP_P, generator)

    def tower_sample(logits):
        step_tower.append(logits.float().reshape(-1).cpu().numpy())
        return sample(logits)

    frames = []
    session = drama3.experts.session(first, drama3.patchers()) if drama3.experts is not None else contextlib.nullcontext()
    with session:
        hidden = run(rows)
        history = torch.full((RAS_WINDOW,), -1, dtype=torch.long, device=last)
        window = rows[-RAS_WINDOW:, 0]
        history[RAS_WINDOW - len(window):] = window
        frame = rows[-1:]
        with comfy.model_management.cuda_device_context(last):
            if prompt_rows > 1:
                count = prompt_rows - 1
                tower.run(hidden[:, :-1], rows[1:], rows[:-1],
                          tower.context(torch.arange(count, device=last), count, last, dtype), tower_caches)
            hidden = hidden[:, -1:]
            for step in range(args.max_frames):
                position = prompt_rows - 1 + step
                logits = audio_head(hidden[0])
                main_logits.append(logits.float().reshape(-1).cpu().numpy())
                draw = candidates[sample(logits)]
                retry = candidates[sample_top_k_top_p(logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, generator)]
                repeated = (history == draw[:, None]).any(-1) & tower.is_semantic(draw)
                main = torch.where(repeated, retry, draw)
                step_tower.clear()
                frame = tower.step(main, hidden, frame,
                                   tower.context(torch.tensor([position], device=last), 0, last, dtype),
                                   tower_caches, tower_sample)
                if int(frame[0, 0]) == config.eos_token_id:
                    break
                if step % args.tower_every == 0:
                    kept = np.stack(step_tower)
                    tower_logits.append(kept.astype(np.float16) if args.tower_f16 else kept)
                    tower_steps.append(step)
                frames.append(frame)
                history = torch.cat((history[1:], frame[:, 0]))
                if step + 1 < args.max_frames:
                    hidden = run(frame)
    return torch.cat(frames).cpu(), np.stack(main_logits), np.stack(tower_logits), np.array(tower_steps)


with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
    model = load_drama3(str(args.weights / "drama3_fp8experts_bf16.safetensors"), experts_in_ram=True,
                        expert_cache_gb=args.expert_cache_gb)
    codec = load_codec(str(args.weights / "drama3_codec.safetensors"))
    wav, sr = sf.read(str(args.ref_wav), dtype="float32", always_2d=True)
    transcript = args.ref_wav.with_suffix(".txt").read_text(encoding="utf-8").strip()
    voice = nodes.Drama3EncodeVoice.execute(
        codec, {"waveform": torch.from_numpy(wav.T.copy())[None], "sample_rate": sr}, transcript).args[0]
    comfy.model_management.load_models_gpu(model.patchers())
    prompt = build_prompt(model.tokenizer, model.config, args.text, [(voice["transcript"], voice["codes"])], [])
    frames, main_logits, tower_logits, tower_steps = record_take(model, prompt)

args.out.mkdir(parents=True, exist_ok=True)
(args.out / "forced.json").write_text(json.dumps({
    "text": args.text, "reference_text": transcript, "seed": args.seed,
    "prompt_rows": prompt.tolist(), "frames": frames.tolist(),
    "reference_codes": voice["codes"].tolist(),
    "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "max_frames": args.max_frames,
                 "tower_every": args.tower_every, "tower_f16": args.tower_f16},
}), encoding="utf-8")
np.savez_compressed(args.out / "port_logits.npz", main=main_logits, tower=tower_logits, tower_steps=tower_steps)
print(json.dumps({"prompt_rows": len(prompt), "frames": len(frames), "main": list(main_logits.shape),
                  "tower": list(tower_logits.shape), "hit_max_frames": len(frames) == args.max_frames}))
