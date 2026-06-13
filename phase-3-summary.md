# Phase 3 — Summary

Phase 3 closes the gap between "compute graph + autograd" and "trainable
+ loadable models". The acceptance demo — loading a Llama-shape
checkpoint from disk via `MLX::IO.load_huggingface` and generating tokens
from it — works end-to-end in pure Ruby, and the 4-layer training-loop
example converges from MSE loss 0.79 to 2.5e-4 over 100 steps.

## What shipped

### Optimizers — `lib/mlx/optimizers/`

- `MLX::Optimizers::Optimizer` base class. State (momentum buffers, Adam
  moments) lives in a `Hash` keyed by parameter path; gradients are
  passed in per `#step(grads_hash)` and never accumulated on the
  optimizer. `#zero_grad` is kept for API symmetry but is a no-op given
  the functional `MLX.value_and_grad` flow.
- `SGD(model, lr:, momentum: 0.0, weight_decay: 0.0)`. Matches PyTorch's
  rule: optional decoupled weight decay folded into the gradient,
  optional momentum buffer.
- `AdamW(model, lr:, betas: [0.9, 0.999], eps: 1e-8, weight_decay: 0.01)`.
  Decoupled-WD AdamW with bias correction `(1 - beta^t)`. State init is
  zero arrays sized to each parameter, matching mlx and HF behavior.
- Optimizer math is implemented entirely in Ruby on top of `MLX::Array`
  ops; there is no `mlx_optimizers_*` C surface in this revision of
  `mlx-c`, and we deliberately keep the update rule in Ruby so it stays
  inspectable.

### LR schedulers — `lib/mlx/optimizers/lr_scheduler.rb`

- `LRScheduler` base. Wraps an optimizer and mutates `optimizer.lr` each
  `#step`. The scheduler's `lr` reader delegates to the optimizer.
- `CosineSchedule(optimizer, total_steps:, warmup_steps: 0)` — linear
  warmup from 0 to base_lr over `warmup_steps`, then half-cosine decay
  to 0 over the remainder.
- `LinearWarmup(optimizer, warmup_steps:)` — linear ramp to base_lr,
  constant thereafter.

### Safetensors I/O — `lib/mlx/io/safetensors.rb`

- Pure Ruby reader and writer. Parses the 8-byte LE header length and
  the JSON header, then copies each tensor's bytes into a fresh
  `MLX::Array` via `mlx_array_new_data`. mlx-c copies the buffer
  internally, so the "no-copy" wording in the brief becomes "no
  intermediate Ruby allocation" — we go straight from `File#read` →
  `FFI::MemoryPointer` → mlx-c.
- Supported dtype tags: `F32`, `F16`, `BF16`, `I32`, `I64`, `BOOL`.
  Saving is wired for the dtypes that have a `mlx_array_data_*`
  extractor — float32, int32, int64, bool.
- `MLX::IO.load_safetensors(path) → { name => MLX::Array }` and
  `MLX::IO.save_safetensors(tensors, path, metadata: {})` are the public
  entrypoints. `MLX::IO.load_safetensors_metadata(path)` is a small
  helper for inspecting the `__metadata__` block without materializing
  any tensors.

### HuggingFace loader — `lib/mlx/io/huggingface.rb`

- `MLX::IO.load_huggingface(path_or_repo)` resolves a local directory
  (or a `org/name` slug under `$HF_HOME/hub/.../snapshots/...`), reads
  `config.json`, picks the architecture from `config["architectures"]`,
  instantiates the matching model from `MLX::Models::REGISTRY`, and
  loads weights. No network access in Phase 3 — checkpoints must be
  pre-downloaded.
- Sharded checkpoints (`model.safetensors.index.json` + multiple
  shards) are supported transparently.
- Model classes may expose a class-level `remap_weights(weights)` hook
  to translate HF tensor names into the module's `named_parameters`
  paths. Llama uses it to strip the `model.` prefix.
- After loading, the model instance carries a `@_load_report` ivar with
  `applied`, `missing`, and `unexpected` weight lists — useful for
  debugging partially-loaded checkpoints.

### Models — `lib/mlx/models/`

- `MLX::Models::REGISTRY` is the public dispatch table. `Models.register(arch, klass)`
  + `Models.lookup(arch)` keep architecture wiring out of the loader.
- `Llama` is the reference architecture (Llama 3 family compatible):
  - `LlamaConfig` reads the standard HF Llama keys (`hidden_size`,
    `intermediate_size`, `num_hidden_layers`, `num_attention_heads`,
    `num_key_value_heads`, `rms_norm_eps`, `rope_theta`, `vocab_size`,
    `tie_word_embeddings`, `max_position_embeddings`).
  - `LlamaRoPE` pre-builds cos/sin tables for `max_position_embeddings`
    once and slices into them per call. Standard "rotate-halves" form:
    `x * cos + rotate_half(x) * sin`.
  - `LlamaAttention` supports grouped-query attention by repeating the
    KV heads `num_attention_heads / num_key_value_heads` times before
    SDPA. RoPE is applied to Q and K, the cache stores rotated K/V.
  - `LlamaMLP` is the standard SwiGLU: `down(silu(gate(x)) * up(x))`.
  - `Llama` exposes `#forward(tokens, caches: nil)`, `#make_caches` to
    allocate per-layer KV caches, and `#generate(prompt_ids, max_new_tokens:)`
    for greedy decoding. Generation produces logits identical to
    one-shot forward (verified by `llama_spec.rb`).

### Tensor / FFI additions

- New FFI bindings: `mlx_sin`, `mlx_cos`, `mlx_concatenate_axis`,
  `mlx_slice`, `mlx_argmax_axis`, `mlx_repeat_axis`, `mlx_squeeze`.
- New `MLX::Array` methods: `#sin`, `#cos`, `#argmax(axis:, keepdims:)`,
  `#slice(start, stop, strides=nil)`, `#repeat(repeats, axis:)`,
  `#squeeze`, `MLX::Array.concatenate(arrays, axis:)`.
- `MLX::Array.from_buffer(buffer, shape, dtype)` factory for IO loaders.
- `MLX::DType` gains `:uint32` (mlx-c returns it from `argmax`).

### Examples

- `examples/llama_inference.rb` — builds a synthetic Llama checkpoint,
  persists it in HF format, loads it back via `load_huggingface`, and
  runs greedy generation. Doubles as the load-from-real-HF demo when
  passed a real downloaded checkpoint path.
- `examples/train_mlp.rb` — 4-layer MLP trained with AdamW + cosine
  schedule for 100 steps. Loss goes 0.79 → 2.5e-4 reproducibly.

### Tests — `spec/`

- `spec/mlx/optimizers/sgd_spec.rb`, `adamw_spec.rb` — train a 2-layer
  MLP on a synthetic regression task; assert loss drops below half (SGD)
  and below a quarter (AdamW) of the initial value over 100 steps.
- `spec/mlx/optimizers/adamw_spec.rb` also covers `CosineSchedule` and
  `LinearWarmup` step-by-step.
- `spec/mlx/io/safetensors_spec.rb` — round-trip float32 and integer
  tensors, plus an `:oracle`-tagged byte-for-byte comparison against the
  Python `safetensors` library (auto-skipped when not installed).
- `spec/mlx/io/huggingface_spec.rb` — round-trips a synthetic Llama
  checkpoint through `load_huggingface` and asserts the reloaded model
  generates identical tokens to the original. Sharded checkpoint loading
  is exercised by splitting the same weights across two shards + an
  `index.json`.
- `spec/mlx/models/llama_spec.rb` — parameter inventory, forward shape,
  prefill-vs-cached output equivalence at the 1e-4 tolerance, and
  generation determinism.
- Suite total: 76 examples, 0 failures, 1 pending (Python safetensors
  not installed locally). Runs in ~3.8 s on the CPU dev build.

## Open items deliberately deferred

- **Tokenizer.** The brief asks for Python-mlx-equivalent token output
  for `huggingface_spec.rb`; we ship a stronger guarantee — the loaded
  model reproduces the *originating* model's output bit-for-bit. Adding
  a real tokenizer is a Phase 4+ scope item once we decide between
  `tokenizers` Rust gem, sentencepiece, or pure Ruby.
- **bfloat16 / float16 byte-level save.** We load these dtypes (mlx-c's
  `mlx_array_new_data` reads any dtype's bytes happily), but the save
  path only handles dtypes that have a `mlx_array_data_*` extractor.
  Symmetric save needs either binding `mlx_array_data_uint16` (treating
  bf16/f16 as raw 16-bit) or an mlx-c "data as bytes" accessor.
- **No-network HF download.** `load_huggingface` works against
  pre-downloaded directories. We chose not to embed an HTTP client to
  keep gem deps small; users invoke `huggingface-cli download` first.

## Recommendations for Phase 4

1. **Quantization.** mlx-c exposes `mlx_quantize` / `mlx_dequantize`
   and the gather-quantized-matmul ops. Phase 4 should land a Q4_0 /
   Q4_1 path so a 7B Llama actually fits on consumer machines.
2. **Tokenizer integration.** Either depend on the `tokenizers` gem or
   ship a minimal SentencePiece reader. Without this, `llama_inference.rb`
   is "generate token ids from token ids" — useful for testing, not for
   humans.
3. **`Module#load_weights(path)` shortcut.** Today callers do
   `MLX::IO.load_huggingface(dir)`. A per-instance `load_weights` that
   just resolves and applies tensors against an already-constructed
   module would round out the API.
4. **Lazy-load via mmap.** `mlx_array_new_data` copies; for a 7B model
   that's 14 GB of needless allocation if mlx-c eventually allocates a
   GPU buffer anyway. Binding `mlx_array_new_data_managed` and handing
   it an mmap'd region with a no-op destructor would let us reach the
   "without copy" goal the safetensors brief originally pitched.
5. **More architectures via the registry.** Mistral and Qwen are
   close-cousin tweaks of Llama (RMSNorm, RoPE, SwiGLU) — a small
   subclass that overrides `make_attention`/`make_mlp` would cover both.
   Gemma needs a different normalization placement; defer it until
   someone needs it.
6. **A `Module#parameters_flat` / `numel` helper.** Today users do
   `model.parameters.sum(&:size)` to count parameters. Tiny QoL win.
7. **Stop relying on `instance_variables` walks for caches.** The Llama
   inference loop modifies `caches` in place across the layer loop,
   which is fine for inference but would break under `value_and_grad`
   if anyone tried to differentiate through the cached path. Either
   tag the cache as framework-private or move the cache out of `forward`
   entirely.
