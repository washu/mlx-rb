# Changelog

All notable changes to mlx-rb are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-06-13

### Added

- **Native HF Hub downloader** (`MLX::IO::Hub`). Pure stdlib `Net::HTTP`,
  no new runtime deps. Resumable via HTTP `Range`, concurrent-file
  thread pool, byte-compatible cache layout with `huggingface_hub`
  (`blobs/`, `snapshots/<commit>/`, `refs/<rev>` symlinks).
- `MLX::IO.load_huggingface("org/name")` now auto-downloads when the
  repo isn't cached locally. Pass `download: false` to opt out.
- `exe/mlx-rb` CLI binstub:
  `mlx-rb download REPO [--revision REV] [--include PAT] [--workers N]`.
- `HF_TOKEN` env / `~/.cache/huggingface/token` auth, in the same
  locations `huggingface_hub` looks.
- Roadmap doc at `docs/roadmap.md` covering 0.3 (LoRA adapter API) and
  the items deliberately left to Forge.

## [0.1.0] — 2026-06-11

First public release. Pre-stable: the surface is usable and audited but
explicit version bumps reserve the right to break the API before 1.0.

### Phase 1 — tensors & FFI

- `MLX::Array` with constructor (nested Ruby array), `zeros`, `ones`,
  `arange`, `full`, `random_normal`, `random_uniform`.
- Elementwise + reduction + shape ops bound through `mlx-c`.
- Eager-by-default with `MLX.lazy { ... }` for batched evaluation.
- Apple Silicon platform check at `require` time.

### Phase 2 — autograd & nn

- `MLX::Transforms.value_and_grad`, `MLX.grad`.
- `MLX::NN::Module` base class with depth-first parameter walking.
- `Linear`, `LayerNorm`, `RMSNorm`, `Embedding`, `Dropout`,
  `MultiHeadAttention`, plus the functional API (`relu`, `gelu`,
  `silu`, `softmax`, `cross_entropy`, `mse_loss`).

### Phase 3 — optimizers, IO, model loading

- `MLX::Optimizers::SGD`, `AdamW`, `CosineSchedule`, `LinearWarmup`.
- Pure-Ruby safetensors reader and writer.
- `MLX::IO.load_huggingface(path)` for local HF checkpoint directories,
  including sharded ones.
- Reference Llama (Llama-3 family) implementation with KV cache and
  greedy generation in `MLX::Models::Llama`.

### Phase 4 — quantization

- `MLX.quantize`, `MLX.dequantize`, `MLX.quantized_matmul` — thin
  wrappers over mlx-c's affine quantization kernels.
- `MLX::NN::QuantizedLinear` with `.from_linear` to convert dense layers
  in place.
- `MLX.quantize_model(module, bits:, group_size:, predicate:)` walks
  the module tree and swaps `Linear` for `QuantizedLinear`.
- HF loader recognizes the `quantization` block in `config.json` and
  loads pre-quantized safetensors into the right slots.

### Tooling

- GitHub Actions matrix on macOS arm64 across Ruby 3.1/3.2/3.3.
- `bin/setup` verifies the host platform, builds `mlx-c` from the
  vendored source if `libmlxc.dylib` isn't present, and runs a smoke
  test before exiting.
- `bench/` scaffolds for matmul, attention forward, and Llama-1B
  generation throughput against Python `mlx`.

### Architecture notes

Three load-bearing decisions are written down in `docs/adr/`:

- 0001 — eager by default, lazy via block.
- 0002 — PyTorch-style modules; ivar-walking parameter discovery.
- 0003 — Ruby FFI over mlx-c; no C++ or Rust extension.

### Known gaps (deferred to 0.2)

- No tokenizer integration — examples emit token ids.
- HF loader doesn't download from the Hub.
- bf16 / fp16 save path is limited (load works fine).
- No quantization-aware training.
