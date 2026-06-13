
---

### `docs/adr/0003-binding-strategy.md`

```markdown
# ADR 0003: Ruby FFI over mlx-c, not a C++ or Rust extension

## Status

Accepted, 2026-06-10.

## Context

Three paths to bind MLX from Ruby:

1. **Ruby FFI against `mlx-c`** — Apple ships a C API specifically for
   non-C++ language bindings. Ruby's `ffi` gem consumes C ABIs directly.
2. **C++ extension via rice or rb-sys** — wrap `libmlx.dylib` (C++ ABI)
   directly. Pattern used by Torch.rb (`rice` wrapping LibTorch).
3. **Rust extension via magnus + `mlx-rs`** — bind Rust's MLX wrapper,
   expose to Ruby. Pattern used by `red-candle` over `candle`.

Each has different cost, distribution, and performance characteristics.

## Decision

**Ruby FFI against `mlx-c`. No C++ or Rust extension in the gem.**

The FFI layer lives in `lib/mlx/ffi.rb`. Idiomatic Ruby wrapping lives in
`lib/mlx/array.rb`, `lib/mlx/nn/*.rb`, etc. The gem has no compiled Ruby
extension of its own.

`mlx-c` itself must be available as a system library. The gem ships a
`bin/setup` that builds it from vendored source if not present.

## Consequences

**Positive:**
- No C++ toolchain required of gem consumers beyond what mlx-c needs.
- `mlx-c` is Apple's supported boundary. We're aligned with where Apple
  invests testing and stability effort.
- Smaller maintenance surface: no `extconf.rb` for our own C code, no
  Rust toolchain to manage.
- FFI declarations are readable and easy to extend op-by-op.
- Faster initial development. Phase 1 ships sooner.

**Negative:**
- Per-call FFI overhead is higher than an inline C++ extension. Measured
  in low microseconds per call. Negligible for ML workloads where each
  op dispatches a Metal kernel that runs for milliseconds, but real for
  scalar-heavy code.
- We depend on `mlx-c`'s API surface. If Apple doesn't expose an op
  through the C API, we can't reach it (or we have to vendor a patch).
- Garbage-collection coordination: every `MLX::Array` needs an
  `FFI::AutoPointer` with the right release function. Easy to get wrong.

**Mitigations:**
- ADR is revisitable in v0.2+ if profiling shows FFI overhead matters
  for our actual workloads. The wrapping layer above FFI is unchanged
  if we swap the implementation.
- If `mlx-c` lags behind `mlx`, file upstream issues and pin to known-good
  versions in our gemspec.
- Memory-safety tests in CI: tight allocate/free loops with `GC.stat`
  assertions catch pointer-management regressions.

## Alternatives considered

1. **C++ extension via rice (Torch.rb pattern).** Rejected for v0.x:
   more upfront work, requires C++ toolchain on every install, and
   `mlx-c` makes it unnecessary. Reconsider if hot-path overhead
   becomes a real bottleneck.
2. **Rust extension via magnus + `mlx-rs` (red-candle pattern).**
   Rejected: adds Rust toolchain dependency for install, and `mlx-rs`
   itself is another moving piece in the stack. `mlx-c` cuts that out.
3. **Hybrid: FFI for most ops, a small C extension for hot paths.**
   Possibly in v0.2+. Out of scope for v0.1 to keep the build story simple.

## Validation milestones

- **End of Phase 1:** confirm FFI overhead is <5% on a synthetic
  matmul-heavy benchmark vs. Python mlx.
- **End of Phase 3:** confirm Llama-1B inference throughput in Ruby
  is within 10% of Python mlx.

If either milestone fails, reopen this ADR.
