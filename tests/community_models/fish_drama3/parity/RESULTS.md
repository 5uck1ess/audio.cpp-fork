# Drama 3 (fish_drama3) teacher-forced parity, 2026-10-16

Port take: `take-p1/` recorded on Windows/RTX 5090 with `port_dump.py` from the ComfyUI-Drama3 port @ 66c4fb9
(FP8 experts + bf16, previously matched Fish's fish_omni reference at mean main-token KL 0.0041). Text
"Open the browser and read my email.", reference chris_hemsworth_15s (wav + transcript), seed 0, 37 frames.
audio.cpp: upstream main 0fdc5723 + the env-gated hook in generator.cpp, RTX 3090 (Linux, CUDA), Jmica GGUFs
@ bcb065b8, context 4096, codec on CPU, cpu_expert_layers 12 (Q2_K) / 24 (Q4_K_S). Prompt rows forced from the port.

| Pair | main KL mean | main KL max | top-1 agree | tower KL mean |
|---|---:|---:|---:|---:|
| port vs audio.cpp Q2_K | 0.148 | 0.468 | 0.78 | ~0.095 |
| port vs audio.cpp Q4_K_S | 0.0123 | 0.037 | 0.86 | 0.0054 |
| port vs audio.cpp Q4_K_S (repeat) | 0.0136 | 0.034 | 0.86 | 0.0058 |
| audio.cpp Q4_K_S vs its own repeat | 0.0043 | 0.016 | 1.00 | 0.0016 |

P(end of speech) on the last step: port 0.9994, Q2_K 0.9974, Q4_K_S 0.9983. KL is flat across the take.
Reading: with identical prompt rows the backbone + tower track the port; Q2->Q4 cuts KL 12x, and Q4_K_S sits
~3x above audio.cpp's own run-to-run noise (0.0043, the same size as the port-vs-reference 0.0041). Consistent
with quantization; no sign of an implementation bug in generation. Run-to-run output is not bitwise stable.

## Reference encoding mismatch
audio.cpp's own prompt matches the port's on every text row (97 rows) but on none of the 324 reference-audio rows.
`codec_check.py` (port codec only): re-encoding the wav with the port's codec on this machine reproduces the
Windows codes 99.1%. Decoding with the port's codec: port codes -> mel L1 3.2 dB, SNR 6.8 dB, Whisper reads the
reference transcript; audio.cpp's codes -> mel L1 11.6 dB, SNR -1.3 dB, Whisper reads nothing. No codebook
permutation, time shift or reversal explains it. audio.cpp's codes are stable across stereo / mean-mono /
left-channel input. So voice cloning in fish_drama3 conditions the model on reference codes that Fish's codec
does not decode to the reference. Not yet isolated: encoder graph vs the packaged codec weights (the GGUF folds
weight normalization), or whether audio.cpp's own decoder reads its codes back.

Scope: a clean comparison shows audio.cpp == Fish's code (via the port), not that the model or Fish's code is
bug-free. The port's codec was not itself checked against Fish's reference here; it is the codec the port's
reference-matched generations were produced with.

## Cloning A/B (Q4_K_S, same text, seeds 1-3)
"I never said she stole the money. Honestly, I just wanted to know where it went, and why nobody told me."
Speaker similarity to the reference (seed-tts-eval WavLM-large SV, tts-bench `scoring/sim.py`):

| Reference given as | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|
| `--voice-ref` wav (audio.cpp encodes it) | -0.149 | -0.137 | -0.087 |
| `references=[{text, vq_codes}]` with the port's codes | 0.655 | 0.730 | 0.763 |

All six transcribe correctly (Whisper base). All runs above used `fish_drama3.codec_backend=cpu`.

## Narrowed: only the CPU codec encodes wrong
Re-encoding the same wav with the codec on CUDA (no `codec_backend=cpu`, Q2_K, 36 CPU expert layers to fit):
80.2% of the 324 reference rows match the port on all 10 codebooks (normal numeric drift); the CPU codec's codes
match 0% and decode to nothing intelligible. So the bug is in the CPU path of reference encoding: it hits
`fish_drama3.codec_backend=cpu` (the documented memory saver) and, presumably, CPU-backend runs. CUDA-codec
cloning: see below. Decoding on the CPU codec looked fine in the smoke tests (LM takes decoded to intelligible
speech), so encode, not decode, is the likely culprit. Not tested: a CPU-only build, Vulkan, Metal.

## End-to-end cloning, CPU vs CUDA codec (Q2_K, 36 CPU expert layers, seeds 1-3, `--voice-ref` wav only)
| Codec | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|
| CUDA (default) | 0.805 | 0.763 | 0.718 |
| `codec_backend=cpu` | -0.091 | -0.101 | -0.039 |

Clips and argv logs in `ab/q2_*`. Frame-step-only main KL (excluding the final EOS row): Q2_K 0.152, Q2_K at 36
layers 0.154, Q4_K_S 0.013, Q4 repeat 0.014, Q4 vs repeat 0.0045. The `ab/codes_s*` runs passed only
`references=` (no `--voice-ref`).

## After the fix (de4f4c97, #936)
CPU codec, Q2_K, 36 CPU expert layers: reference rows match the port 82.7% on all 10 books (per book 0.988-0.895, same as CUDA). Cloning SIM seeds 1-3: 0.686 / 0.771 / 0.756 (was -0.09 / -0.10 / -0.04). Fixed.

## Codec parity: drift or bug? (`encode_fp32.py`, `margins.py`)
Port codec in strict FP32 (TF32 off) gives identical codes on CPU and CUDA. Against it, all-10-book frame match:
the port at its default ComfyUI precision 0.710, audio.cpp CPU (fixed) 0.676, audio.cpp CUDA 0.676. Scoring only
frames whose earlier books agree (same residual input), each book differs on 2-6% of frames, and audio.cpp's pick
is the strict port's 2nd-ranked entry (median; max rank 6) with a median cosine margin of 7e-4 to 6e-3, against a
typical top1-top2 gap of 1.5e-2 to 3.6e-2. The port's own default-precision run shows the same profile, including
the same few larger-margin outliers (up to 0.1-0.13 on book 1, 0.054 on books 5-6). The codebooks are 8-dim cosine
argmax, so near-ties are common, and the residual chain spreads a single flip to the later books. Reading: what
remains looks like numerical drift, not a framework bug. Not checked: pre-quantization latents from audio.cpp.

## PR #937 (BF16 CUDA reference encoder, 5ea78d0d) regression check
Q2_K, 36 CPU expert layers, same cloning text, seeds 1-3, WavLM SIM:

| Encoder | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|
| CUDA, `codec_encoder_bf16_compute` on (new default) | 0.692 | 0.678 | 0.745 |
| CUDA, bf16 off | 0.479 | 0.622 | 0.715 |
| CPU codec | 0.805 | 0.655 | 0.748 |

bf16-off codes equal pre-#937 CUDA codes exactly; the CPU codec still matches the port (82.7%), so #936 survives.
bf16-on codes share only 6% of frames with the FP32 port references, which is expected (the PR targets Fish's BF16
Python codec). All three keep the speaker. Seed spread (bf16-off s1 0.48 here vs 0.81 pre-#937 with identical codes)
shows that generation differs from run to run; three seeds cannot rank the encoders.

## Long take, single pass (take-long1: passage 1, 2830 frames, top_k 4097, Windows port)
audio.cpp Q4_K_S forced along it: main-token KL 0.0101 over frame steps, flat by time bin
(0.0119, 0.0102, 0.0102, 0.0100, 0.0099, 0.0100, 0.0091, 0.0096), top-1 agreement 0.90, tower KL ~0.006.
No long-context drift. Chunking was off (`text_chunk_size=100000`), so this says nothing about chunked pacing.

## Chunked long-form pacing (#920): two causes
Dialogue passage from #920's drama_stress.py, Q4_K_S, default 200-char chunks, scored by `pacing_score.py`.
1. Tag-aware chunker split `."` between `.` and `"`: chunks opened with a stray closing quote (8 of 14 chunks
   in this passage). Fixed in branch `chunk-closing-quotes` (framework). See chunker_probe/ for before/after chunks.
2. `top_k=30` stalls: 7 of 11 takes had 9-21 s of silence inside the take, WER up to 0.31. With top_k=4097
   (Fish's Space setting): 0 of 11, longest silence 4 s, WER 0.02-0.07. Branch `drama3-topk-uncapped`.
Chunk joins themselves add ~0.1 s (`chunk_pauses.py`, needs FISH_DRAMA3_CHUNK_LOG debug print, not upstream).
