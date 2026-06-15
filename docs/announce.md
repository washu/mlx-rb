# mlx-rb 0.1.0 — Ruby on Apple Silicon, finally a real ML substrate

> Draft. Not yet published. Trim where it drags.

I just tagged the first release of [mlx-rb](https://github.com/washu/mlx-rb),
a Ruby binding for Apple's [MLX](https://github.com/ml-explore/mlx)
framework. It's the substrate I always wanted on my M-series Mac and
couldn't find: tensors, autograd, neural-network modules, optimizers,
HuggingFace model loading, and 4/8-bit quantization, all in
idiomatic Ruby, running on Metal through Apple's own `mlx-c` C API.

If you've been doing ML in Ruby on a Mac and quietly accepting that the
tooling story is "fine, just use Python," this is the gem that flips
that. On an M-series machine MLX is the fast path: unified memory means
your full 32 GB / 64 GB / 128 GB acts as effective VRAM with no
host-device transfers, and Apple's Metal kernels are tuned for it. A
4-bit quantized Llama-7B fits in under 6 GB resident on an M1 Ultra.

## What's in 0.1

The whole back-half of a modern ML stack:

- **Tensors.** `MLX::Array` with the usual constructors, elementwise
  ops, broadcasting, slicing, reductions. Eager by default with a
  `MLX.lazy { ... }` block for tight inner loops.
- **Autograd.** `MLX.value_and_grad` against a Ruby block, parameter-aware
  on `MLX::NN::Module`.
- **NN modules.** `Linear`, `LayerNorm`, `RMSNorm`, `Embedding`,
  `Dropout`, `MultiHeadAttention`. Modules are mutable; parameters are
  discovered by walking instance variables.
- **Optimizers.** `AdamW`, `SGD`, `CosineSchedule`, `LinearWarmup`.
- **safetensors I/O + HuggingFace loading.** Sharded checkpoints,
  per-arch weight remapping, partial-load reporting.
- **A reference Llama 3 implementation** with KV cache and greedy
  decoding.
- **4-bit / 8-bit quantization** with a single-line model swap and a
  Linear → QuantizedLinear walker.

The full surface is around 1500 lines of Ruby on top of `mlx-c`. There
is no C++ or Rust extension in this gem — by design.

## Three reasons to look at it

**1. Performance you can verify.** Inference and training-step
throughput on Apple Silicon match Python MLX within a few percent for
typical workloads, because the Ruby layer is doing what it should be
doing: orchestrating Metal kernels. The `bench/` directory has matmul,
attention, and Llama-1B generation comparisons.

**2. APIs that read like Ruby.** I deliberately did not chase Python
MLX's tree-of-dataclasses module style. Modules are mutable, you write
plain Ruby classes with a `#forward`, you call `.parameters` and you
get an array.

**3. The right hardware story.** No CUDA, no fallback to a stubbed-out
Metal backend, no host/device tensor split. Apple Silicon only — and
the platform check fires at `require` time so there are no surprises.

## What's *not* in 0.1

- No tokenizer. Examples emit token ids and you bring your own
  (sentencepiece, the `tokenizers` gem, anything that produces ints).
- No HF Hub HTTP downloader. Run `huggingface-cli download` first.
- No quantization-aware training. Quantization is post-training.
- No distributed / multi-node.

These are the obvious 0.2 candidates.

## Try it

```bash
# Apple Silicon, Xcode CLT, CMake, Ruby ≥ 3.1
gem install mlx-rb
```

```ruby
require "mlx"

a = MLX::Array.new([[1, 2], [3, 4]])
b = MLX::Array.ones([2, 2])
puts a.matmul(b).to_a
# => [[3.0, 3.0], [7.0, 7.0]]
```

[Repo](https://github.com/washu/mlx-rb) ·
[Architecture](https://github.com/washu/mlx-rb/blob/main/docs/architecture.md) ·
[Changelog](https://github.com/washu/mlx-rb/blob/main/CHANGELOG.md)

If you want one-shot quantized-checkpoint production from a dense HF
model, the sibling [`mlx-convert`](https://github.com/washu/mlx-convert)
CLI does that in a single command.
