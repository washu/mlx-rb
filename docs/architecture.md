# Architecture

This is the half-page tour. The three ADRs (`docs/adr/0001`, `0002`,
`0003`) cover *why* the load-bearing decisions are the way they are; this
document covers *what* the resulting code looks like at a structural
level.

## Layers

```
┌──────────────────────────────────────────────┐
│  Your Ruby app                               │
├──────────────────────────────────────────────┤
│  MLX::IO, MLX::Models                        │  HF loader, reference Llama
├──────────────────────────────────────────────┤
│  MLX::NN, MLX::Optimizers                    │  Modules, AdamW/SGD, schedules
├──────────────────────────────────────────────┤
│  MLX::Array, MLX::Transforms, MLX::Quantized │  Tensors, autograd, quant
├──────────────────────────────────────────────┤
│  MLX::FFI                                    │  Raw mlx-c bindings (FFI)
├──────────────────────────────────────────────┤
│  libmlxc.dylib (Apple, C API)                │
├──────────────────────────────────────────────┤
│  libmlx.dylib (Apple, C++ + Metal)           │
└──────────────────────────────────────────────┘
```

Each layer talks only to the layer directly below it. The MLX::FFI module
is the entire C boundary. If a function isn't bound there, nothing else
in the gem can reach it.

## The FFI boundary (lib/mlx/ffi.rb)

mlx-c represents arrays, streams, devices, and closures as single-pointer
"handle" structs. On AArch64 the calling convention passes a one-pointer
struct in a register, identical to a bare pointer, so Ruby FFI's
struct-by-value support marshals them correctly with no glue.

Lifetimes are handled with `FFI::AutoPointer` — the wrapped `ctx` is
released by an `mlx_*_free` callback when the Ruby object is GC'd. This
is the trick that lets `MLX::Array` look like a normal Ruby value.

Two patterns recur and are worth knowing:

- **Out parameters.** Most ops return `int` (status) and write the result
  into an `mlx_array*` passed as the first argument. The Ruby side
  allocates a fresh `mlx_array_new`, passes `.pointer`, then wraps the
  returned struct with `from_struct`.
- **Optional scalars.** `mlx_optional_int` and `mlx_optional_dtype` are
  two-field structs `{ int value; bool has_value; }`. The `FFI.opt_int`
  / `FFI.opt_dtype` helpers construct them.

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
delegates differentiation to mlx-c. `MLX::NN::Module#value_and_grad` is
the parameter-aware sugar layer: it flattens the model's parameter
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

## I/O (MLX::IO)

- `MLX::IO::Safetensors` — pure-Ruby reader/writer of the
  [safetensors](https://github.com/huggingface/safetensors) format. The
  loader maps the JSON header byte-ranges directly through FFI; mlx-c
  copies into its own buffer on the way in.
- `MLX::IO.load_huggingface(path)` — accepts either a local directory or
  a `org/name` slug resolved against the standard `huggingface_hub`
  cache. Sharded checkpoints (`model.safetensors.index.json` + multiple
  shards) are handled. A `quantization` block in `config.json` triggers
  the Phase-4 quantized-load path.

There is no Hub HTTP client in 0.1 — users invoke
`huggingface-cli download` first.

## The ADRs at a glance

- **0001 — eager vs lazy.** Eager by default. Lazy is opt-in via a block
  so the failure mode of "I forgot to call .eval" doesn't exist.
- **0002 — module system.** PyTorch-style mutable modules over mlx's own
  tree-of-dataclasses approach. Ivar walking finds parameters; arrays of
  submodules are supported.
- **0003 — binding strategy.** Ruby FFI against mlx-c. No C++ extension,
  no Rust extension. The C++ wrapping work has already been done
  upstream in mlx-c; we don't repeat it.
