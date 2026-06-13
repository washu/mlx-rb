# mlx-rb

Ruby bindings for [Apple's MLX](https://github.com/ml-explore/mlx) machine
learning framework.

> **Status: 0.1.0 (pre-stable), Apple Silicon only.** All five build
> phases have shipped: tensors, autograd, nn modules, optimizers, model
> loading, and 4/8-bit quantization. APIs may still change before 1.0.

## What this is

`mlx-rb` is a Ruby FFI binding over [mlx-c](https://github.com/ml-explore/mlx-c),
Apple's official C API for MLX. It exposes MLX's tensor operations, automatic
differentiation, neural-network modules, optimizers, model loading, and
quantization to Ruby.

It's the substrate gem. Higher-level training, fine-tuning, adapter
lifecycle, and CLI tooling live in [Forge](https://github.com/washu/forge)
(or whatever the orchestration gem is called by the time you read this).

```
┌─────────────────────────────────────────┐
│  Your Ruby app / Forge / CLI            │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  mlx-rb  (this gem — idiomatic Ruby)    │
└──────────────────┬──────────────────────┘
                   │  Ruby FFI
┌──────────────────▼──────────────────────┐
│  mlx-c (Apple, C API)                   │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  libmlx.dylib (Apple, C++ + Metal)      │
└─────────────────────────────────────────┘
```

## Why MLX, why now

MLX is designed from scratch for Apple Silicon. Unified memory, Metal
kernels, no host/device split. On an M-series Mac it gives you access to
all of system memory as effective "VRAM" — a 128GB Mac Studio can hold
a 70B-parameter model in 4-bit quantization comfortably.

Existing options for ML in Ruby:
- **Torch.rb** wraps LibTorch. Cross-platform, good op coverage, but its
  MPS backend is partial and not designed for unified memory.
- **red-candle** wraps the Rust `candle` framework. Cross-platform,
  growing, but Metal support is less mature than MLX's.
- **rllama** wraps `llama.cpp`. Best-in-class for GGUF inference, but
  inference-only.

`mlx-rb` is the Apple-Silicon-first option. It's not a replacement for
those gems; it's the right substrate when your hardware is M-series.

## Requirements

- **macOS** with **Apple Silicon (M1 or later)**
- **Xcode Command Line Tools**
- **CMake** (for building `mlx-c`)
- **Ruby ≥ 3.3**

Linux and Intel Mac are not supported and won't be. The gem fails fast
at load time with a clear message on unsupported platforms.

## Installation

```ruby
# Gemfile
gem "mlx-rb", "~> 0.1"
```

```bash
bundle install
bundle exec bin/setup    # verifies macOS arm64, builds mlx-c, runs smoke test
```

On a fresh checkout you can also install straight from a built gem:

```bash
gem build mlx-rb.gemspec
gem install mlx-rb-0.1.0-arm64-darwin.gem
```

If `libmlxc.dylib` isn't on your loader path, point `MLX_C_LIB` at it
directly.

## Three worked examples

### 1. Inference — load Llama and generate

```ruby
require "mlx"

# Auto-downloads on first use, then loads from the local HF cache.
model = MLX::IO.load_huggingface("NousResearch/Llama-2-7b-hf")

prompt = [1, 2, 3, 4]                        # token ids; bring your own tokenizer
tokens = model.generate(prompt, max_new_tokens: 32)
puts tokens.inspect
```

The HF loader handles sharded checkpoints transparently and reports
which tensors were missing or unexpected via
`model.instance_variable_get(:@_load_report)`. Use the CLI for explicit
downloads:

```bash
mlx-rb download NousResearch/Llama-2-7b-hf
```

### 2. Fine-tuning — small MLP, AdamW + cosine schedule

```ruby
require "mlx"

class MLP < MLX::NN::Module
  def initialize
    super()
    @l1 = MLX::NN::Linear.new(64, 128)
    @l2 = MLX::NN::Linear.new(128, 1)
  end

  def forward(x)
    h = MLX::NN.relu(@l1.call(x))
    @l2.call(h)
  end
end

model  = MLP.new
optim  = MLX::Optimizers::AdamW.new(model, lr: 3e-3)
sched  = MLX::Optimizers::CosineSchedule.new(optim, total_steps: 100, warmup_steps: 10)

loss_fn = ->(m, x, y) { ((m.call(x) - y) ** MLX::Array.new(2.0)).mean }

100.times do |step|
  x = MLX::Array.random_normal([16, 64])
  y = MLX::Array.random_normal([16, 1])
  loss, grads = MLX.value_and_grad(model) { |m| loss_fn.call(m, x, y) }
  optim.step(grads)
  sched.step
  puts "step=#{step} loss=#{loss.to_a}" if (step % 10).zero?
end
```

A full version lives in [`examples/train_mlp.rb`](examples/train_mlp.rb).

### 3. Quantized inference — 4-bit weights, no accuracy bath

```ruby
require "mlx"

model = MLX::IO.load_huggingface("./llama3")

# Quantize every Linear *except* the language-model head.
MLX.quantize_model(model, bits: 4, group_size: 64) { |path, _| path != "lm_head" }

tokens = model.generate([1, 2, 3, 4], max_new_tokens: 32)
puts tokens.inspect
```

Memory: a 7B model in fp16 is ~13 GB; 4-bit / group_size=64 brings the
weights to ~3.7 GB, leaving headroom on a 16 GB M-series machine. The
end-to-end script with a synthetic checkpoint is in
[`examples/quantized_llama.rb`](examples/quantized_llama.rb); the real
7B variant is in
[`examples/quantized_llama_7b.rb`](examples/quantized_llama_7b.rb).

## Constraints by design

- **Apple Silicon only.** No CUDA, no Linux, no Intel Mac.
- **Inference and single-node training.** No multi-node distributed.
  See the project ADRs for the explicit non-goals.
- **mlx-c as the boundary.** If Apple doesn't expose an op through
  `mlx-c`, we don't expose it from Ruby. Upstream first.
- **No model zoo in this gem.** One reference architecture (Llama) ships
  here. Others live in separate gems

## Project shape

```
mlx-rb/
├── lib/mlx/
│   ├── array.rb         # MLX::Array tensor type
│   ├── ffi.rb           # FFI declarations against mlx-c
│   ├── nn/              # Linear, LayerNorm, attention, etc.
│   ├── optimizers/      # AdamW, SGD, schedulers
│   ├── io/              # safetensors, HuggingFace loading
│   ├── models/          # reference Llama implementation
│   ├── quantized.rb     # 4-bit / 8-bit quantization
│   └── quantize_model.rb # walker that swaps Linear → QuantizedLinear
├── ext/mlx_c/           # mlx-c source (vendored)
├── bench/               # benchmark scripts vs Python mlx
├── sig/                 # RBS types
└── docs/                # ADRs, architecture, announce
```

## Design decisions

The load-bearing choices are written down as ADRs:

- [`0001-eager-vs-lazy.md`](docs/adr/0001-eager-vs-lazy.md) — eager by
  default, lazy via `MLX.lazy { ... }` block
- [`0002-module-system.md`](docs/adr/0002-module-system.md) —
  PyTorch-style modules with `#forward`
- [`0003-binding-strategy.md`](docs/adr/0003-binding-strategy.md) —
  Ruby FFI over `mlx-c`, no C++ or Rust extension

Read these before contributing. They explain why things are the way
they are.

## Roadmap

mlx-rb was built in five phases. All five are now in `main` and shipped
in 0.1.0:

| Phase | Scope | Status |
|---|---|---|
| 1 | Tensor primitives, basic ops, FFI plumbing | Shipped |
| 2 | Autograd, `MLX::NN::Module`, common layers | Shipped |
| 3 | Optimizers, safetensors, HuggingFace loading, reference Llama | Shipped |
| 4 | 4-bit / 8-bit quantization, `QuantizedLinear` | Shipped |
| 5 | Audit, docs, benchmarks, v0.1.0 release | Shipped |

Post-0.1 work is sized into 0.2 (HF Hub downloader) and 0.3 (LoRA
adapter API). See [`docs/roadmap.md`](docs/roadmap.md) for the briefs.
Tokenizers, GPTQ/AWQ conversion, training loops, and multi-node stay
out of scope for mlx-rb proper — those belong in Forge or external CLIs.

## Non-goals

- Linux or Intel Mac support
- Multi-node distributed training
- Bleeding-edge CUDA kernels (Flash Attention NVIDIA variants, FP8 on
  Hopper, etc. — those belong on other hardware)
- Replicating Python mlx's entire surface area in v0.x — we cover the
  subset needed for transformer training and inference
- A model zoo. One reference architecture lives here; others go in
  separate gems

## Relationship to other Ruby ML gems

- **`torch-rb`** — different substrate (LibTorch). Use when you need
  cross-platform or CUDA. `mlx-rb` is the better choice on Apple Silicon.
- **`red-candle`** — different substrate (candle/Rust). Use when you
  want a single gem that works across CPU/CUDA/Metal. `mlx-rb` is the
  better choice when Apple Silicon performance is the priority.
- **`rllama`** — different scope (inference only via llama.cpp).
  Complementary; use rllama for serving GGUF models, `mlx-rb` for
  training.

## Contributing

Not yet open for contributions. The API is changing too fast. Once v0.1
ships, contribution guidelines will land in `CONTRIBUTING.md`.

If you find a bug or a missing op in pre-release, open an issue with:
- macOS version + Apple Silicon generation
- Ruby version
- `mlx-c` commit hash you built against
- Minimal reproducer

## License

MIT. See [`LICENSE.txt`](LICENSE.txt).

MLX itself is also MIT, copyright Apple. `mlx-c` is MIT.

## Acknowledgements

- The MLX team at Apple for building a framework that takes Apple
  Silicon's architecture seriously
- The `torch-rb` and `red-candle` projects for proving Ruby can host
  serious ML substrates
- The `ankane`-pattern Ruby ML ecosystem for the design conventions
  this gem borrows from
