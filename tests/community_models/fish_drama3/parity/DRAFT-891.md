**Drama 3: `fish_drama3.codec_backend=cpu` mis-encodes voice references, so cloning loses the speaker. Generation tracks Fish's code.**

Tested `main` @ `0fdc5723` on an RTX 3090 (CUDA build) with Jmica's GGUFs. The comparison is a take recorded from the ComfyUI-Drama3 port. Under teacher forcing that port matched Fish's `fish_omni` reference at a mean KL of 0.0041. Repro, scripts and raw numbers: https://github.com/5uck1ess/audio.cpp-fork/tree/drama3-parity-main/tests/community_models/fish_drama3/parity

**Bug: with `codec_backend=cpu`, the reference audio is encoded into the wrong codes.**
- On the same 15 s wav, the CUDA codec's codes match the port's on 80% of frames across all 10 codebooks (88–99% per book). The CPU codec's codes match on 0%.
- The CPU codes are identical whether the input is stereo, mean-mono or the left channel, and whatever the backbone quant or expert offload.
- Decoded with the port's codec, the CPU codes are unintelligible. The port's codes decode back to the reference transcript.
- End-to-end cloning, Q2_K, everything else identical, three seeds, WavLM speaker similarity to the reference:
  - CUDA codec: 0.81 / 0.76 / 0.72
  - CPU codec: −0.09 / −0.10 / −0.04
  - Q4_K_S with the CPU codec: −0.15 / −0.14 / −0.09
  - Q4_K_S with the port's codes passed as `references=[{text, vq_codes}]`, no `--voice-ref`: 0.66 / 0.73 / 0.76
- All clips say the right words. Only the CPU-codec clips have the wrong speaker.
- Not tested: CPU-only builds, Vulkan, Metal, whether audio.cpp's own decoder reads the CPU codes back, and whether `fish_audio` (S2 Pro) is affected through the shared codec.

**Generation.** On this one 37-frame take, with the port's prompt rows forced in:
- Tokenizer and prompt template match the port on all 97 text rows.
- Main-token KL against the port over the 37 frame steps is 0.152 at Q2_K and 0.013 at Q4_K_S. Two identical Q4_K_S audio.cpp runs differ by 0.0045 (runs are not bitwise stable with 24 CPU expert layers).
- Tower KL is about 0.005 at Q4_K_S. P(end) on the last step is 0.998 against the port's 0.999.
- The Q2→Q4 drop isn't from the different expert offload: Q2_K with 12 or with 36 CPU layers gives the same KL (0.152 vs 0.154).
- That points to quantization rather than an implementation bug, for these logits. Sampling and longer takes weren't checked.

Caveat: this compares audio.cpp with Fish's code (via the port). It can't show whether Fish's code or the model has problems of its own. The port's codec wasn't checked against Fish's encoder, though its codes decode back to the reference words.
