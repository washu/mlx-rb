# Roadmap

Post-0.1 work, sized into two releases. The shape mirrors the briefs
the earlier phases used: background, deliverables, definition of done,
scope guards. Items not on this list are deferred to 0.4+ or to Forge.

---

## 0.2 — HF Hub downloader

### Background

Today `MLX::IO.load_huggingface(path)` requires the checkpoint to
already be on disk. Users invoke `huggingface-cli download` first, which
adds a Python dependency to a gem whose whole premise is "pure Ruby on
top of mlx-c." A native downloader closes the only gap that forces
users out of the gem during normal model loading.

### Deliverables

1. **`lib/mlx/io/hub.rb`** — native HTTP client using stdlib `Net::HTTP`
   (no new runtime gem):
   - `MLX::IO::Hub.download(repo_id, revision: "main", files: nil, allow_patterns: nil)`
     resolves the repo at `https://huggingface.co/api/models/<repo>/revision/<rev>`,
     enumerates the file list, and downloads matching files into the
     standard `$HF_HOME/hub/models--<org>--<name>/snapshots/<sha>/` layout
     with `blobs/` symlinks (same as `huggingface_hub`).
   - LFS-pointer redirects (large `.safetensors` files) followed
     transparently.
   - Resume from partial downloads via HTTP `Range` requests; an
     interrupted run picks up where it left off.
   - Concurrent file downloads via `Thread.new` (4–8 workers default,
     configurable via env var).
   - Auth via `HF_TOKEN` env var or `~/.cache/huggingface/token` file
     (the same locations `huggingface_hub` uses).
   - Progress hook: `download(..., progress: ->(file, bytes, total) { ... })`.

2. **`MLX::IO.load_huggingface(repo_or_path)` extension** — if the arg
   looks like an `org/name` slug and isn't in the cache, call
   `Hub.download` automatically. Pass `download: false` to opt out.

3. **`exe/mlx-rb` CLI** — single binstub:
   ```
   mlx-rb download <repo_id> [--revision REV] [--include PATTERN]
   ```
   Lets users use mlx-rb as their HF CLI replacement.

4. **Specs** in `spec/mlx/io/hub_spec.rb`:
   - WebMock the HF API; round-trip a fake two-file repo through
     `download`, assert the cache layout matches `huggingface_hub`'s.
   - Resume test: write a partial file, start `download`, assert it
     sends `Range: bytes=N-` and grows the file rather than restarting.
   - `:online`-tagged real-HF test that hits a known small public repo
     (auto-skipped without `HF_ONLINE=1`).

### Definition of done

- `MLX::IO.load_huggingface("TinyLlama/TinyLlama-1.1B-Chat-v1.0")`
  works from a fresh `$HF_HOME` with no Python in PATH.
- Cache layout is byte-compatible with `huggingface_hub` — running the
  Python CLI before or after mlx-rb's downloader on the same repo
  doesn't produce duplicate files.
- Resume works after `Ctrl-C` mid-download.

### Scope guards

- **No upload.** Push is out of scope.
- **No Datasets, Spaces.** Models only.
- **No `safetensors` lazy-mmap streaming.** That's a separate Phase 3
  follow-up.
- **No new runtime deps.** `Net::HTTP` only.

---

## 0.3 — Adapter API for QLoRA-style training

### Background

The user-facing ask is "quantization-aware training." The shape that
actually matters is QLoRA: rank-r adapters trained on top of a frozen
4-bit `QuantizedLinear`. Pure QAT (training the quantizer) is rare and
not worth the infra. mlx-rb is a substrate gem — it doesn't ship the
training loop, but it has to expose the primitives that make the loop
possible upstream.

What's already there: `QuantizedLinear` is frozen, its bias is
differentiable, and `value_and_grad` walks the param tree.

What's missing: a clean way to attach a trainable rank-r adapter beside
a frozen `QuantizedLinear` so the training loop in Forge can plug both
into one optimizer.

### Deliverables

1. **`lib/mlx/nn/lora.rb`** — `LoRALinear(in_features, out_features, rank:, alpha:)`:
   - Holds two small dense matrices `A` (in×rank) and `B` (rank×out),
     trainable.
   - `#forward(x)` returns `(x @ A) @ B * (alpha / rank)`.
   - `named_parameters` exposes only `A` and `B`.

2. **`MLX::NN::LoRAQuantizedLinear`** — composite that wraps an existing
   frozen `QuantizedLinear` plus a `LoRALinear`. `#forward(x)` returns
   `base.forward(x) + lora.forward(x)`. `named_parameters` exposes just
   the LoRA pair so optimizers update only those.

3. **`MLX.attach_lora(module, rank:, alpha:, predicate: nil)`** —
   parallel to `quantize_model`. Walks the tree, wraps every matching
   `QuantizedLinear` (or dense `Linear`) in a `LoRAQuantizedLinear`.
   Common idiom: attach to `q_proj` and `v_proj` only.

4. **Adapter checkpoint I/O** — `MLX::IO.save_adapter(module, path)` and
   `load_adapter(module, path)`. Saves *only* the LoRA pairs, not the
   base weights. Roughly 10–50 MB per LoRA on a 7B base, vs 13 GB for
   the base.

5. **Specs** in `spec/mlx/nn/lora_spec.rb`:
   - LoRALinear standalone: gradient flows through A and B.
   - LoRAQuantizedLinear: base weight unchanged after optimizer step,
     A and B updated.
   - attach_lora walker swaps only matching paths.
   - Adapter round-trip through `save_adapter` / `load_adapter`.

### Definition of done

- A toy script in `examples/lora_finetune.rb` runs a 50-step LoRA fit
  on a 1B-shape Llama, loss decreasing monotonically, and persists
  the adapter as a < 50 MB safetensors file.
- Forge can `require "mlx"`, call `attach_lora`, and drive the
  training loop using only public mlx-rb API.

### Scope guards

- **No training loop in mlx-rb.** That's Forge's job.
- **No QLoRA paper-specific tricks** (paged optimizer, double quant).
  Standard LoRA over MLX's existing 4-bit format.
- **No multi-adapter swap-at-runtime API.** One adapter per model.
- **No pure-QAT** (training the scales/biases themselves).

---

## Out of scope for mlx-rb proper

- **GPTQ / AWQ converters.** External one-shot tools that produce
  mlx's existing on-disk quantized format. Belong in a dedicated
  `mlx-convert` CLI or in Forge — not in this gem.
- **Training loops, schedulers-with-warm-restart, distributed.**
  Forge territory.
- **Multi-node.** No.
- **Tokenizers.** External (user already has one).
