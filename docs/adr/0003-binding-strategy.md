
---

### `docs/adr/0003-binding-strategy.md`

```markdown
# ADR 0003: Ruby FFI over mlx-c, not a C++ or Rust extension

## Status

Accepted, 2026-06-10. **Revised, 2026-06-15** — see "Revision (v0.4)"
at the bottom. The original decision is preserved above for historical
context; the v0.4 revision applies to current code.

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

## Revision (v0.4)

The "no extension" promise survived v0.1–v0.3 but ran into a real
install-UX problem: every user needed CMake + Xcode CLT to build
`mlx-c` from vendored source. For a gem whose value prop is "ML on
Apple Silicon in Ruby," forcing every consumer to install and run
CMake at gem-install time is friction we can't justify.

v0.4 swaps the build path to a Rust bridge crate
(`ext/mlx_bridge/`) that:

- depends on `mlx-rs` and `mlx-sys`,
- statically links MLX C++, mlx-c, and the Metal `.metallib` artifacts
  into a single `libmlx_bridge.dylib`,
- re-exports the mlx-c C symbol surface via a linker whitelist plus a
  force-keep slice generated at build time,
- ships prebuilt inside the `arm64-darwin` platform gem (~2.9 MB).

This *adds* the very Rust toolchain dependency the original ADR
rejected — but only for *us* (gem maintainers), not for users. End
users install `mlx-rb` and get a self-contained dylib with no
toolchain on PATH. The source gem fallback runs `cargo build --release`
at install time for developers on dev checkouts.

The Ruby `MLX::FFI` surface is unchanged: every `attach_function`
points at a mlx-c symbol the bridge re-exports. The wrapping layers
above FFI (`MLX::Array`, `MLX::NN`, etc.) didn't change at all.

### What stays true from the original ADR

- The C boundary is still `lib/mlx/ffi.rb`. Adding an op is still a
  one-line `attach_function` declaration; if mlx-c doesn't expose it,
  we don't either.
- We still don't have a Ruby-side C++ extension. The C++ wrapping
  work that mlx-c does isn't repeated in our codebase.
- Garbage-collection coordination via `FFI::AutoPointer` is unchanged.

### What changed

- The "no Rust toolchain" promise applies to *users* only. Maintainers
  building a release need Rust + Xcode (full app, for the Metal
  compiler).
- Install UX improves dramatically: from "install CMake, install Xcode
  CLT, wait 10+ minutes for mlx-c to build" to "`gem install mlx-rb`
  with no toolchain on PATH at all."
- We pick up mlx-c API changes through mlx-sys's bindgen output instead
  of pinning a vendored mlx-c commit. v0.4 absorbed the upstream
  reshaping of `mlx_quantize`, `mlx_fast_scaled_dot_product_attention`,
  and the removal of `mlx_device_count` as part of the swap.
