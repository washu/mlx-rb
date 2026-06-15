# Architecture

This is the half-page tour. The three ADRs (`docs/adr/0001`, `0002`,
`0003`) cover *why* the load-bearing decisions are the way they are;
this document covers *what* the resulting code looks like at a
structural level. Note that ADR 0003 was revised in v0.4.0 — see the
"Substrate" section below.

## Layers

```
┌──────────────────────────────────────────────┐
│  Your Ruby app                               │
├──────────────────────────────────────────────┤
│  MLX::IO, MLX::Models                        │  HF Hub downloader, loader, Llama
├──────────────────────────────────────────────┤
│  MLX::NN, MLX::Optimizers                    │  Modules, LoRA, AdamW/SGD, schedules
├──────────────────────────────────────────────┤
│  MLX::Array, MLX::Transforms, MLX::Quantized │  Tensors, autograd, quantization
├──────────────────────────────────────────────┤
│  MLX::FFI                                    │  Ruby FFI declarations
├──────────────────────────────────────────────┤
│  libmlx_bridge.dylib  (ships in the gem)     │  Rust bridge + statically linked
│   ├─ Rust crate (ext/mlx_bridge/)            │  MLX C++, mlx-c, Metal kernels
│   ├─ mlx-rs                                  │
│   ├─ mlx-sys (bindgen over mlx-c)            │
│   ├─ libmlxc.a                               │
│   └─ libmlx.a + .metallib                    │
└──────────────────────────────────────────────┘
              ↑ only system framework deps (Metal, Accelerate, libc++)
```

Each layer talks only to the layer directly below it. The MLX::FFI
module is the entire C boundary. If a function isn't bound there,
nothing else in the gem can reach it.

## The substrate (v0.4+)

Before v0.4 the gem built `mlx-c` via CMake at install time. v0.4
replaced that path with a Rust bridge crate (`ext/mlx_bridge/`) that
compiles to a single `libmlx_bridge.dylib`. The bridge:

- Depends on `mlx-rs` and `mlx-sys`; `mlx-sys` statically links
  `libmlxc.a` and the MLX C++ core (including Metal `.metallib`
  artifacts) into our cdylib.
- Re-exports the mlx-c C symbol surface that `lib/mlx/ffi.rb` needs.
  `ext/mlx_bridge/exports.txt` is a `-exported_symbols_list` whitelist;
  `build.rs` generates a force-keep function that takes the address of
  every whitelisted symbol so the linker doesn't dead-strip them.
- Defines a small `mlx_rb_bridge_*` surface in `src/lib.rs` for things
  that don't have a direct mlx-c equivalent (ABI-version probe, smoke
  test).

From Ruby's perspective the symbol table is byte-compatible with the
old `libmlxc.dylib` — `lib/mlx/ffi.rb` still calls `attach_function
:mlx_array_new`, `:mlx_quantize`, `:mlx_fast_scaled_dot_product_attention`,
etc.

The platform gem ships the prebuilt dylib (2.9 MB on arm64-darwin).
The source gem runs `cargo build --release` at install time via
`ext/mlx_bridge/extconf.rb`.

## The FFI boundary (lib/mlx/ffi.rb)

mlx-c represents arrays, streams, devices, and closures as
single-pointer "handle" structs. On AArch64 the calling convention
passes a one-pointer struct in a register, identical to a bare pointer,
so Ruby FFI's struct-by-value support marshals them correctly with no
glue.

Lifetimes are handled with `FFI::AutoPointer` — the wrapped `ctx` is
released by an `mlx_*_free` callback when the Ruby object is GC'd.
This is the trick that lets `MLX::Array` look like a normal Ruby
value.

Two patterns recur and are worth knowing:

- **Out parameters.** Most ops return `int` (status) and write the
  result into an `mlx_array*` passed as the first argument. The Ruby
  side allocates a fresh `mlx_array_new`, passes `.pointer`, then wraps
  the returned struct with `from_struct`.
- **Vector-array out params.** Some ops (e.g. `mlx_quantize`) return
  multiple arrays. v0.4-vintage mlx-c moved most of these to discrete
  `mlx_array*` out-pointers; `mlx_quantize(qw_out, scales_out, biases_out,
  w, group_size, bits, stream)` is the canonical shape now.

## The tensor type (MLX::Array)

`MLX::Array` is the universal data type. Constructors include
`new(nested_array)`, `zeros`, `ones`, `random_normal`, `random_uniform`,
`arange`, `full`, and `from_buffer` (used by the safetensors loader).
Every op returns a fresh array.

Per ADR 0001 evaluation is eager. After each op an `mlx_array_eval` is
forced before the value is returned to Ruby. Wrapping a region in
`MLX.lazy { ... }` defers eval to block exit so a tight inner loop can
batch.

## Autograd (MLX::Transforms)

`MLX.value_and_grad(fn)` wraps a Ruby block in an `mlx_closure` and
delegates differentiation to mlx-c. `MLX::NN::Module#value_and_grad`
is the parameter-aware sugar layer: it flattens the model's parameter
hash to a flat vector_array, runs `mlx_value_and_grad`, and re-nests
the gradient hash.

## The module system (MLX::NN)

Per ADR 0002 modules are mutable Ruby objects with a `#forward` method
and ivar-based parameter discovery. Anything that's an `MLX::Array` or
another `Module` (or an `Array` of those) is part of the tree.
Framework-private ivars start with `@_`.

Public API:

```ruby
mod.parameters              # flat list of MLX::Array
mod.named_parameters        # { "fc.weight" => ... } for save/load
mod.update(named)           # in-place replacement (used by optimizers)
mod.freeze / mod.unfreeze   # toggle trainability
```

## Quantization (Phase 4)

`MLX.quantize(weights, bits:, group_size:)` returns the canonical
`[qw, scales, biases]` triple. The `qw` packs `32 / bits` weights per
uint32 along the K axis, so a 4-bit `[out, in]` matrix becomes
`[out, in / 8]`. Scales and biases hold one fp value per group of
`group_size` weights, shape `[out, in / group_size]`.

`QuantizedLinear` wraps that triple plus an optional dense bias. It
exposes `named_buffers` (the quantized state) separately from
`named_parameters` (only the dense bias is trainable) — the dense
weight is gone, not zero-grad. `QuantizedLinear.from_linear(layer)`
quantizes an existing `Linear` in place.

`MLX.quantize_model(module, ...)` is a tree walker that swaps every
matching `Linear` for a `QuantizedLinear`. The optional predicate gates
which paths are eligible; the common pattern is
`{ |path, _| path != "lm_head" }`.

## LoRA (v0.3+)

`MLX::NN::LoRALinear` is the rank-r adapter:
`delta(x) = (x @ A) @ B * (alpha / rank)` with Kaiming-uniform `A` and
zero `B` init so the initial delta is identically zero.

`MLX::NN::LoRAQuantizedLinear` is a composite that wraps a frozen
`Linear` or `QuantizedLinear` with a trainable `LoRALinear`.
`named_parameters` exposes only the LoRA pair, so an optimizer walking
the model touches just the adapter.

`MLX.attach_lora(module, rank:, alpha:, predicate:)` is a tree walker,
sibling to `quantize_model`. `MLX::IO.save_adapter` / `load_adapter`
round-trip the LoRA pairs via safetensors.

## I/O (MLX::IO)

- `MLX::IO::Safetensors` — pure-Ruby reader/writer of the
  [safetensors](https://github.com/huggingface/safetensors) format.
- `MLX::IO::Hub` — native HuggingFace Hub downloader (stdlib
  `Net::HTTP`, no new runtime deps). Cache layout matches
  `huggingface_hub` byte-for-byte: `blobs/<oid>`, `snapshots/<commit>/`,
  `refs/<rev>`.
- `MLX::IO.load_huggingface(path_or_repo)` — accepts either a local
  directory or an `org/name` slug. If the slug isn't cached, the
  loader calls `Hub.download` automatically; pass `download: false`
  to opt out. Sharded checkpoints handled. A `quantization` block in
  `config.json` triggers the quantized-load path.

## The ADRs at a glance

- **0001 — eager vs lazy.** Eager by default. Lazy is opt-in via a
  block so the failure mode of "I forgot to call .eval" doesn't exist.
- **0002 — module system.** PyTorch-style mutable modules over mlx's
  own tree-of-dataclasses approach. Ivar walking finds parameters;
  arrays of submodules are supported.
- **0003 — binding strategy.** *Revised in v0.4.* Originally: Ruby FFI
  directly against mlx-c, built via CMake at install time. Now: Ruby
  FFI against a Rust bridge crate (`ext/mlx_bridge/`) that statically
  links MLX C++ and re-exports the mlx-c symbol surface. Users get a
  precompiled `arm64-darwin` platform gem with no toolchain
  requirements.
