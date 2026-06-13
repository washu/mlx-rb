# Phase 4 — Summary

Phase 4 makes 7B–34B models actually tractable on consumer Apple Silicon
by adding mlx-c's affine 4/8-bit quantization to the Ruby surface. The
acceptance demo — quantize a synthetic Llama and confirm generated token
ids are stable — produces 8/8 token-id parity with the dense path at
4-bit on the in-tree synthetic.

## What shipped

### Quantization primitives — `lib/mlx/quantized.rb`

- `MLX.quantize(w, bits:, group_size:)` → `[qw, scales, biases]`. Thin
  wrapper around `mlx_quantize`; returns the canonical mlx triple.
- `MLX.dequantize(qw, scales, biases, bits:, group_size:)` → dense
  `MLX::Array`.
- `MLX.quantized_matmul(x, qw, scales, biases, bits:, group_size:, transpose:)`
  — fused dequant+matmul, defaults to `transpose: true` to match the
  `x @ W^T` orientation `Linear` uses.

The wrappers also live under `MLX::Quantized.*` for callers that prefer
the namespaced form.

### FFI additions — `lib/mlx/ffi.rb`

- New `MlxOptionalInt` / `MlxOptionalDtype` structs mirroring
  `mlx_optional_int` / `mlx_optional_dtype` from `upstream/mlx/c/optional.h`.
- `MLX::FFI.opt_int(value)` / `opt_dtype(value)` constructors.
- `mlx_quantize`, `mlx_dequantize`, `mlx_quantized_matmul` bound.

### `MLX::NN::QuantizedLinear` — `lib/mlx/nn/quantized_linear.rb`

- Holds `@weight` (packed uint32), `@scales`, `@biases` as buffers and
  optional dense `@bias` as the only trainable parameter.
- `#forward(x)` routes through `MLX::Quantized.quantized_matmul`.
- `.from_linear(linear, bits:, group_size:)` quantizes an existing
  `Linear` in place and copies the bias verbatim.
- `freeze` is the steady state; `named_parameters` returns just the
  bias, `named_buffers` returns the quantized triple. `#update` accepts
  either parameter or buffer names, so the HF loader can populate it
  directly.

### Walker — `lib/mlx/quantize_model.rb`

- `MLX.quantize_model(module, bits:, group_size:, predicate:)` walks the
  module tree, swapping every matching `Linear` for a `QuantizedLinear`.
  Accepts the predicate either as a kwarg or as a block; the common idiom
  is `{ |path, _| path != "lm_head" }` to leave the LM head dense.
- Recurses into nested modules and arrays of modules (handles
  `LlamaBlock`'s shape out of the box).

### HF loader extension — `lib/mlx/io/huggingface.rb`

- Recognizes the `quantization` block in `config.json`:
  `{ "bits": 4, "group_size": 64, "skip_modules": ["lm_head"] }`.
- Quantizes the matching `Linear` layers *before* loading weights so
  the safetensors names `weight`/`scales`/`biases` land in the right
  slots.
- New `collect_slots` and `assign_slots` helpers handle the
  parameter-vs-buffer split QuantizedLinear introduces.

### Examples

- `examples/quantized_llama.rb` — synthetic-tiny Llama, runs in-process.
  Reproduces 8/8 token-id parity at 4-bit on the in-tree config.
- `examples/quantized_llama_7b.rb` — real Llama 7B path; not run by CI.
  Reads the dense checkpoint, quantizes, generates. Memory target
  documented inline.

### Specs

- `spec/mlx/quantized_spec.rb` — shape invariants, round-trip RMS
  bounds (<0.1 at 4-bit, <0.01 at 8-bit on a standard normal),
  fused-vs-manual matmul agreement, plus `:oracle`-tagged element-wise
  agreement with Python `mlx.core.quantize` / `quantized_matmul`.
- `spec/mlx/nn/quantized_linear_spec.rb` — `from_linear` shape
  preservation, bias copy, `frozen?` + `named_parameters` shape,
  validation errors.
- `spec/mlx/quantize_model_spec.rb` — full-tree swap, predicate-based
  skip, forward equivalence under 8-bit.
- Total suite: 93 examples, 0 failures, 1 pending (Python safetensors
  oracle).

## Release polish (Phase 4 doubled with the 0.1.0 audit)

- Version bumped to `0.1.0` in `lib/mlx/version.rb`; gemspec replaced
  the bundler TODOs with real metadata, MIT license, ffi runtime dep,
  Apple Silicon platform pin.
- `gem build mlx-rb.gemspec` succeeds and `gem install --user-install`
  works from a clean shell.
- `CHANGELOG.md` rewritten as the proper Keep-a-Changelog entry for the
  five phases plus deferred work.
- README rewritten: removed the "phase planned" tables, added three
  worked examples (inference, fine-tuning, quantized inference) that
  match the actual public API.
- `docs/architecture.md` — half-page tour of the layer stack, FFI
  patterns, modules, and quantization.
- `docs/announce.md` — blog-post-quality draft for the 0.1 announce.
- `sig/mlx/rb.rbs` covers the public API including Phase 4 surfaces.
- `bin/setup` now does platform/toolchain checks, builds `mlx-c` if
  missing, and runs a require-time smoke test.
- `bench/` scaffolds for matmul, attention, and Llama-1B generation
  with Python mlx side-by-side comparison.
- `.github/workflows/main.yml` switched to `macos-14` (Apple Silicon
  runners) across Ruby 3.1/3.2/3.3 with a separate rubocop job.
- `.rubocop.yml` tuned for the codebase shape; `bundle exec rubocop
  lib/` passes clean.

## Open items deliberately deferred

- **Quantization-aware training.** Out of scope per the brief; goes to
  Phase 5 or Forge.
- **GPTQ / AWQ conversion.** mlx Python already does this; users
  convert externally and load via the HF loader.
- **bf16/fp16 save symmetry for quantized scales.** Load works; save
  inherits the Phase 3 limitation.
- **Real-hardware memory verification.** The `<6 GB resident on M1
  Ultra for 7B 4-bit` criterion lives in `examples/quantized_llama_7b.rb`
  and the README; CI can't enforce it.
- **Actual `gem push`.** The artifact builds clean and installs from
  the local file. RubyGems push is a one-line `gem push` the maintainer
  runs when ready, not a CI action.
