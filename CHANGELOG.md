# Changelog

All notable changes to mlx-rb are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] — 2026-06-15

### Added

- **Precompiled `arm64-darwin` platform gem.** Users `gem install mlx-rb`
  with no Rust toolchain, no CMake, no Xcode CLT. The 2.9 MB gem ships
  a single self-contained `libmlx_bridge.dylib` (MLX C++, mlx-c, and
  the Rust bridge statically linked together).
- **Source gem fallback.** `ext/mlx_bridge/extconf.rb` runs
  `cargo build --release` at install time for users on the source gem.
  Cargo + Xcode CLT required only in this path.
- `rake compile` / `rake native_gem` / `rake release:gems` tasks for
  local builds.
- `bin/setup` updated to verify the Rust toolchain and call
  `rake compile` instead of the old CMake path.
- **`.github/workflows/release.yml`** — on `v*` tag push, builds both
  the source gem and the precompiled `arm64-darwin` platform gem,
  smoke-tests the platform gem in a clean shell, then publishes both
  to RubyGems via OIDC trusted publishing and attaches them to a
  GitHub Release. Tag↔gemspec version mismatches abort the release.
- **`.github/workflows/main.yml`** — rewritten for the Rust bridge.
  Sets up Rust, selects Xcode, caches `~/.cargo/registry` and the
  bridge `target/` dir, runs `rake compile` then `rspec` across
  Ruby 3.1–3.3 on `macos-14`.

### Verified end-to-end against the Rust bridge on M1 Ultra

- Spec suite: 110 examples, 0 failures, 2 pending.
- Llama-2-7B real-weights load: 12 853 MB → 3975 MB at 4-bit (3.23x)
  with 4/4 token-id parity vs. dense (`"Hello, my name is" →
  " Katie and I"`).
- Platform gem installs in a fresh shell with no toolchain on PATH
  and runs Metal matmul correctly.

## [0.4.0.pre.1] — 2026-06-15

### Changed — substrate swap to Rust bridge

- **Replaced the mlx-c CMake build path with a Rust bridge crate.**
  `ext/mlx_bridge/` (Cargo, depends on `mlx-rs` + `mlx-sys`) compiles
  to a single `libmlx_bridge.dylib` that statically links MLX C++,
  mlx-c, and Metal. The dylib re-exports the mlx-c C symbols (via a
  generated `exports.txt` linker whitelist + a force-keep slice in
  `build.rs`) so `lib/mlx/ffi.rb`'s existing `attach_function :mlx_*`
  calls keep working unchanged.
- `ext/mlx_c/` submodule removed.
- New developer-override env var: `MLX_BRIDGE_LIB`. `MLX_C_LIB` is
  honored as a legacy alias.

### Upstream API changes picked up with this swap

- `mlx_quantize` / `mlx_dequantize` / `mlx_quantized_matmul` lost their
  `mlx_optional_int` parameters and `mode`/`global_scale` slots. The
  Ruby wrappers in `lib/mlx/quantized.rb` were simplified accordingly.
- `mlx_fast_scaled_dot_product_attention` consolidated its two mask
  array slots into a single `mlx_vector_array`. Llama and
  `MultiHeadAttention` call sites updated.
- `mlx_device_count` is gone upstream; `MLX.gpu_available?` now uses
  `mlx_metal_is_available`.

### Not yet in this prerelease

- Precompiled `arm64-darwin` platform gem. Developer checkouts still
  need to `cargo build --release` once.
- rb_sys / rake-compiler integration.
- CI publishing the precompiled gem on tag push.

## [0.3.1] — 2026-06-13

### Added

- Safetensors save now handles `:uint16`, `:uint32`, `:float16`,
  `:bfloat16` — the dtypes mlx-c produces for quantized weights.
  Quantized state from `MLX::NN::QuantizedLinear` can now be persisted
  and reloaded, which the new `mlx-convert` CLI depends on.
- `MLX::FFI` binds `mlx_array_data_uint16`/`uint32`/`float16`/`bfloat16`.
- `MLX::DType` registers `:uint16` and adds the corresponding `BYTES`
  entries.

## [0.3.0] — 2026-06-13

### Added — LoRA adapter API

- **`MLX::NN::LoRALinear(in, out, rank:, alpha:)`** — low-rank adapter
  with the standard Kaiming-uniform A / zero B init so initial delta
  is identically zero.
- **`MLX::NN::LoRAQuantizedLinear`** — composite that wraps a frozen
  `Linear` or `QuantizedLinear` with a trainable `LoRALinear`.
  `named_parameters` exposes only the LoRA pair, so an optimizer
  walking the model touches only the adapter.
- **`MLX.attach_lora(module, rank:, alpha:, predicate:)`** — tree
  walker, sibling to `quantize_model`. Common idiom:
  `{ |path, _| path.end_with?("q_proj", "v_proj") }` for paper-style
  attention adapters.
- **`MLX::IO.save_adapter(model, path)` / `load_adapter(model, path)`**
  — safetensors round-trip of *only* the LoRA pairs. ~5 KB on a tiny
  MLP, scales linearly with rank × layer count.
- `examples/lora_finetune.rb` — 50-step toy regression that drops loss
  by ~900×, verifies the base weights stay unchanged, and confirms
  the adapter round-trips to machine precision.

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
- Roadmap doc at `docs/roadmap.md` covering 0.3 (LoRA adapter API).

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
