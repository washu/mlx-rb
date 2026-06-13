# Phase 2 — Summary

Phase 2 lands autograd and the neural-network module system. The
acceptance demo — a 6-layer transformer block written in pure Ruby that
forwards and backprops through mlx-c — works end to end. A 1-block
transformer with hand-set weights matches Python mlx to ~1.2e-7 absolute
diff (fp32 machine epsilon) on the output, and the autograd MLP test
matches the Python `value_and_grad` oracle on every gradient.

## What shipped

### Autograd — `lib/mlx/transforms.rb`

- `MLX.grad(fn, argnums: nil)` and `MLX.value_and_grad(fn, argnums: nil)`.
  Both accept a `Proc` or a block. `argnums` defaults to `[0]` and
  accepts negative indices.
- Behind the scenes we wrap the Ruby block in an
  `FFI::Function(:int, [:pointer, MlxVectorArray.by_value])` matching
  mlx-c's `mlx_closure` signature, hand it to `mlx_closure_new_func`,
  feed the resulting closure into `mlx_value_and_grad`, and apply.
- Inputs to the returned callable must be `MLX::Array`. Passing a Ruby
  number raises `MLX::TypeError`. (Phase brief required this.)
- The callback uses Ruby exception capture to propagate errors:
  `callback_err` is set, `1` is returned, and the outer Ruby method
  re-raises after teardown so we don't try to throw across the
  Ruby↔mlx-c boundary.

### Module system — `lib/mlx/nn/module.rb`

- `MLX::NN::Module#forward(*inputs)` to subclass, `#call(*inputs,
  **kwargs)` as the PyTorch-style entry. `**kwargs` is forwarded so
  modules like `Dropout` can take `training:` flags.
- Parameter detection walks `instance_variables` on demand:
  `MLX::Array` → parameter, child `Module` → submodule, `Array`
  containing either → flattened with numeric indices (so `@blocks =
  Array.new(6) { Block.new }` works as expected).
- Framework-private ivars (anything starting with `@_`) are hidden from
  the walk; `@_frozen` carries the freeze flag.
- `parameters`, `named_parameters("prefix")`, `children`, `freeze`,
  `unfreeze`, `frozen?`, and bulk `update("path.to.param" => array)`.
- No metaclass magic. Per ADR 0002 this is deliberately closer to
  Torch.rb / PyTorch than to Python mlx's tree-of-dataclasses.

### Layers — `lib/mlx/nn/`

- `Linear(in, out, bias: true)` — Kaiming-uniform init, matmul + bias.
- `LayerNorm(dim, eps: 1e-5, affine: true)` — wraps `mlx_fast_layer_norm`.
- `RMSNorm(dim, eps: 1e-5)` — wraps `mlx_fast_rms_norm`.
- `Embedding(num_embeddings, embedding_dim)` — `take` row lookup.
- `Dropout(p)` — inverted dropout via `mlx_random_bernoulli`. No-op when
  `training: false` or `p == 0`.
- `MultiHeadAttention(dim, num_heads, bias: false)` — q/k/v/out
  projections + `mlx_fast_scaled_dot_product_attention`. Supports
  `mask: :causal`.

### Functional — `lib/mlx/nn/functional.rb`

- `MLX::NN::F.relu`, `.silu`, `.gelu` (tanh-approx, matching mlx's
  `gelu_approx`), `.softmax`, `.log_softmax`, `.cross_entropy`,
  `.mse_loss`. Loss reductions: `:mean | :sum | :none`.
- `log_softmax` is derived from `logsumexp` because mlx-c at this
  version does not expose a `mlx_log_softmax` C function.

### Tensor / FFI surface expansion

- New ops on `MLX::Array`: `-@`, `**`, `maximum`, `equal`, `exp`, `log`,
  `sqrt`, `rsqrt`, `square`, `abs`, `sigmoid`, `tanh`, `erf`,
  `stop_gradient`, `sum/mean/logsumexp/softmax` (with `axes:` and
  `keepdims:`), `broadcast_to`, `expand_dims`, `astype`, `take`. Plus
  `coerce` so `2.0 * arr` works without manual scalar wrapping.
- New constructors: `MLX::Array.random_normal`, `.random_uniform`. A
  process-global RNG can be seeded via `MLX.random_seed(seed)`.
- `#to_a` / scalar extraction now also wired for `:int32`, `:int64`,
  and `:bool` (closes Phase 1 known-gap).

### Example

- `examples/tiny_transformer.rb` builds a 6-layer transformer block
  stack with causal attention, runs `value_and_grad` over every
  parameter (74 of them at dim=16, depth=6, heads=4), and reports
  shape + loss. All gradients are finite.

### Tests — `spec/`

- `spec/mlx/transforms_spec.rb` covers the three deliverables called out
  in the brief (x² at x=3 → 6, sum(x·w) wrt w, two-layer MLP fwd/bwd
  vs Python oracle).
- Per-layer specs (`spec/mlx/nn/*_spec.rb`) round-trip each Module and,
  where applicable, diff against the Python mlx oracle. The oracle
  helper grew a `PythonOracle.run_script` variant that takes an inline
  Python snippet so we don't have to extend the central `case op`
  switch for every new op.
- Suite total: 63 examples, 0 failures, ~3.2 s on the CPU dev build.

## Surprises and mlx-c API gaps

1. **The closure callback's output `mlx_vector_array` has a NULL ctx.**
   The C trampoline in `closure.cpp` builds the output vector with
   `mlx_vector_array_new_()` (private, returns `{nullptr}`), then
   passes `&res` to the user function. If you naively wrap that pointer
   in a Ruby struct and call `mlx_vector_array_append_value`, mlx-c
   throws `expected a non-empty mlx_vector_array` from the private
   accessor in `vector.h`. The fix is to allocate a fresh public
   `mlx_vector_array_new()` inside the callback, append outputs to it,
   and write that vector's `ctx` back through the out pointer. This is
   not documented anywhere in the headers; it took an instrumented run
   and a read of `private/vector.h` to spot.
2. **`mlx_fast_scaled_dot_product_attention` crashes on a NULL
   `mask_mode`.** The implementation does `std::string(mask_mode)`
   without a NULL check, so passing a NULL `const char*` is UB.
   Pass `""` for "no mode" instead. Worth either a NULL-check in
   mlx-c or documenting the contract.
3. **`mlx_log_softmax` is not exposed by mlx-c at this revision.** The
   Python `mlx.nn.log_softmax` exists, but the C surface stops at
   `mlx_softmax` and `mlx_logsumexp_axes`. We derive log_softmax in
   Ruby. If a `mlx_log_softmax` lands later we should switch.
4. **`mlx_vector_array_new` returns a non-null ctx, but the private
   underscore variant doesn't.** Two functions with nearly-identical
   names, opposite null behavior. The closure header gives no hint
   about which the trampoline uses. Documentation would help.
5. **mlx-c does not expose a "set parameter" helper for the C
   abstractions** — but that's fine; our Ruby-side `Module#update`
   does the right thing without going through C.
6. **`take_along_axis` (general N-D row gather)** is exposed as
   `mlx_take_along_axis`, but Phase 2 only needs the 2-D
   classification path (cross-entropy). We special-case it inside
   `MLX::NN::F.take_along_axis`; a real N-D implementation can wait
   until something needs it.
7. **`MLX::Array#==` is still not overridden** — `update` and other
   parameter-replacement paths use string-keyed hashes, so this hasn't
   bitten us yet, but tests still compare via `#to_a`.

## Recommendations for Phase 3

1. **Optimizers (`MLX::Optim::SGD`, `Adam`, `AdamW`).** With
   `Module#update` and `MLX.value_and_grad` in place this is a small
   amount of code: optimizer carries state arrays keyed by parameter
   path, applies per-step update rules, and calls `model.update(...)`.
   Per the scope guard for Phase 2 we deferred this entirely.
2. **`take_along_axis` on `MLX::Array`.** Bind `mlx_take_along_axis`
   properly. Then `F.cross_entropy` works for any rank and we can
   delete the 2-D special case.
3. **Model serialization.** `mlx-c` has `io.h` with safetensors/gguf
   readers. A `Module#load_weights(path)` that walks `named_parameters`
   and pulls matching tensors out of a safetensors archive is the
   right Phase 3 shape.
4. **Random key splitting.** Currently we use the global RNG via
   `mlx_random_seed`. For reproducible Dropout / data augmentation we
   should expose `mlx_random_key`, `mlx_random_split`, and let
   `Dropout` accept an optional `key:` argument.
5. **A `value_and_grad` that takes the model directly.** PyTorch users
   are used to `loss.backward()`; the closer ergonomic in mlx is
   `MLX.value_and_grad(model.method(:forward))` returning grads as a
   `{name => grad}` hash that lines up with `named_parameters`. Today
   callers do this remap manually (see `examples/tiny_transformer.rb`).
6. **Drop the FFI `FFI::Function` retain-by-reference hack.** Today the
   closure callback is held alive only by a local reference; we lean
   on `ensure ... cb` to keep it from being GC'd. If a future
   `mlx-c` ever calls the closure asynchronously, this falls apart.
   A small `ThreadLocal` of live callbacks or a payload-carrying
   closure variant would harden this.
7. **`Module` ergonomics: `each_param` block form, parameter count
   helper.** Small QoL improvements once optimizers land.
8. **Try a GPU-enabled mlx-c build.** Per Phase 1's open-task: still
   pending. Phase 2's fast_* paths (layer_norm, rms_norm, sdpa) really
   want Metal. Everything wired here should pick up GPU automatically
   thanks to `MLX.default_device`'s probe.
