# Roadmap

Post-0.3 work. mlx-rb is a standalone gem; everything that used to be
"deferred to Forge" is either in scope here, lives in a sibling CLI
(`mlx-convert`), or is explicitly dropped.

## v0.4 — Rust bridge + precompiled gem

### Background

Today `mlx-rb` builds `mlx-c` (a C wrapper over MLX C++) via CMake at
install time. Users need Xcode CLT and CMake before `gem install` will
produce a working library. v0.4 replaces that path with a Rust bridge
crate over `mlx-rs` that compiles to a single `cdylib`, shipped
prebuilt in an `arm64-darwin`-platform gem. Users get a working install
with no toolchain on disk.

### Deliverables

1. **`ext/mlx_bridge/`** — Rust crate. Cargo dep on `mlx-rs`. Exposes a
   C ABI mirroring the existing `MLX::FFI` surface (~80 functions:
   tensor lifecycle, ops, autograd closures, quantization, fast.h,
   random, optional structs).
2. **`lib/mlx/ffi.rb`** — repointed at the Rust-built dylib; symbol
   names unchanged where mlx-c and the bridge agree, renamed cleanly
   where they don't.
3. **rb_sys + rake-compiler** wiring. `bundle exec rake compile` builds
   the bridge for the host. `bundle exec rake native:arm64-darwin gem`
   produces a precompiled platform gem.
4. **`bin/setup`** — verifies the prebuilt binary loads; falls back to
   `cargo build --release` if the user is on a dev checkout without one.
5. **CI**: macos-14 (Apple Silicon) runner builds + ships the
   precompiled gem on tag push.
6. **`ext/mlx_c/`** — submodule removed. The vendored mlx-c stays in
   git history at the v0.3.x tags for anyone who needs the old path.

### Definition of done

- `gem install mlx-rb` from a fresh shell on M-series macOS produces a
  working gem with no Rust toolchain installed.
- `require "mlx"` loads cleanly, default device is `:gpu`, the full
  spec suite passes against the new bridge.
- The Llama-2-7B end-to-end run (load + 4-bit quantize + generate)
  reproduces the same token ids as on v0.3.1.

### Scope guards

- **No new public Ruby API in v0.4.** Substrate-only swap.
- **Don't drag MLX C++ in directly.** All C++ access goes through
  `mlx-rs`; the bridge crate is for ABI conversion only.
- **No Linux / Intel-Mac builds.** Same platform constraint as the
  existing gem.

## v0.5+ — open questions

- More architectures in `MLX::Models::REGISTRY` (Mistral, Qwen).
- bf16 / fp16 save-path symmetry beyond the 0.3.1 broadening.
- Lazy-mmap safetensors load for very large checkpoints.
- A small training-loop helper (the original "Forge" surface). Lives
  here now, not in a separate gem.

## Explicitly out of scope

- **Linux / Intel-Mac.** MLX is Metal-only.
- **Multi-node distributed.** No.
- **Tokenizers.** External (`tokenizers` gem, sentencepiece, etc.).
- **GPTQ / AWQ pre-dequantization.** Use autogptq/autoawq externally,
  then feed the resulting fp16 dir to `mlx-convert`.
