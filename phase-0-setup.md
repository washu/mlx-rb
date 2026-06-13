# Phase 0 — Setup

This file records the state of the repo when Phase 1 begins. Read it
along with the README and the three ADRs before starting Phase 1 work.

## What shipped

### Repo structure

Standard `bundle gem` layout, with these additions:

- `docs/adr/` — three load-bearing ADRs (read these)
- `ext/mlx_c/` — vendoring location for mlx-c source
- `spec/support/python_oracle.rb` — stub for the Python-mlx oracle
  test helper (Phase 1 implements the body)
- `bin/setup` — extended to build mlx-c and run a smoke test

### Architecture decisions

Recorded as ADRs in `docs/adr/`. Summary:

| ADR | Decision |
|---|---|
| 0001 | Eager evaluation by default; `MLX.lazy { ... }` block for lazy |
| 0002 | PyTorch-style modules with `#forward`; not MLX-Python tree pattern |
| 0003 | Ruby FFI over `mlx-c`; no C++ or Rust extension |

These are not up for debate inside a phase. If a phase finds reason to
revisit, write a new ADR superseding the old one — don't drift silently.

### Gemspec

- `spec.platform = "arm64-darwin"`
- `spec.required_ruby_version = ">= 3.1.0"`
- Dependencies: `ffi ~> 1.16`, `zeitwerk ~> 2.6`
- Dev dependencies: `rspec`, `rubocop`, `yard`, `rake-compiler`
- `metadata["rubygems_mfa_required"] = "true"`

### CI

GitHub Actions matrix runs on `macos-14` (Apple Silicon runners), Ruby
3.1 / 3.2 / 3.3. Steps:
1. Build `mlx-c` from vendored source
2. `bundle install`
3. `bundle exec rspec`
4. `bundle exec rubocop`

Oracle tests are skipped in CI for Phase 1 (Python mlx not yet installed
in the CI image). They run locally during development.

### License

MIT. Confirmed compatible with MLX and mlx-c (both MIT, Apple).

## What was verified

- `mlx-c` builds cleanly from source on the dev machine (M1 Ultra, macOS
  14.x, Xcode CLT installed)
- `python3 -m pip install mlx` succeeds; `import mlx.core` works
- Skeleton `bundle exec rspec` runs (no specs yet, exits 0)
- `bin/console` boots, `require "mlx"` succeeds
- Top-level `MLX.platform_supported?` returns true on dev machine

## What was deliberately not done

The skeleton ships **none** of the following — they belong to Phase 1
or later. Do not assume they exist:

- No `MLX::Array` class. Only the namespace exists.
- No FFI declarations against mlx-c. `lib/mlx/ffi.rb` is empty.
- No tests. `spec/mlx/` is empty.
- No examples in `examples/`.
- No YARD docs beyond placeholder module-level comments.
- No version bump beyond `0.0.1.pre`.

## Pinned versions

Pin these in your work and don't drift without writing an ADR:

| Component | Version | Why |
|---|---|---|
| `mlx-c` | (commit SHA, e.g. `abc1234`) | First version we validated against |
| `mlx` (Python, oracle) | (matching version) | Must match mlx-c for oracle tests to be valid |
| Ruby (minimum) | 3.1 | Pattern matching, in_pattern, etc. |
| FFI gem | ~> 1.16 | Stable; no known incompatibilities |

If you bump `mlx-c` mid-phase, update the matching Python `mlx` version
in the same commit. The oracle is only valid when they match.

## Environment expected at start of Phase 1

- macOS arm64 (M1 or later)
- Xcode CLT installed
- CMake on PATH
- Ruby 3.1+
- `mlx-c` source vendored under `ext/mlx_c/upstream/`
- `mlx-c` built; `libmlxc.dylib` present at the expected path
- Python 3 with `mlx` installed (for oracle tests, optional but expected
  during dev)

`bin/setup` validates all of the above and exits with a clear message if
anything is missing. Run it first if anything feels wrong.

## Open questions deferred to Phase 1

These are decisions the skeleton doesn't make. Phase 1 must settle them
and document the choice (either inline in the relevant file or as a new
ADR if it's load-bearing):

1. **Dtype enum representation in Ruby.** Symbol-based (`:float32`,
   `:bfloat16`) or constant-based (`MLX::Float32`)? ADR 0002 implies
   symbol-based for ergonomics; confirm and commit to it in `lib/mlx/dtype.rb`.

2. **Default device on M-series.** ADR doesn't specify. Phase 1 should
   pick: probably `:gpu` (which on Apple Silicon is the unified Metal
   device) and document that there is no `:cpu` vs `:gpu` distinction
   in the user-facing API the way there is on CUDA. Single device, name
   it `:gpu`, move on.

3. **`MLX::Array#to_a` shape.** Always returns nested Ruby arrays?
   Always returns a flat array with a separate shape accessor? PyTorch
   convention is nested-for-Tensor, flat-for-Storage. Pick nested for
   `#to_a` and add `#to_flat_a` if needed.

4. **Error class hierarchy.** Phase 1 will encounter mlx-c error codes.
   Decide on `MLX::Error` base, then specific subclasses (`MLX::ShapeError`,
   `MLX::DTypeError`, `MLX::FFIError`). Don't proliferate — three or four
   total for the whole gem is enough.

## Recommendations for Phase 1

1. **Build `spec/support/python_oracle.rb` first**, before any FFI work.
   Without a working oracle, you can't verify FFI calls are returning
   correct results. The oracle is the most important infrastructure in
   the whole project. Get it right, get it stable, then build on it.

2. **Start with `mlx_zeros` and `mlx_array_data_float32`.** Smallest
   useful surface that proves the FFI plumbing end-to-end. Once that
   round-trips, the rest of Phase 1 is mechanical.

3. **Write the GC test early.** A tight loop allocating + freeing 100K
   arrays must not grow RSS unboundedly. If `FFI::AutoPointer` is wired
   wrong, you'll find out at 10K iterations rather than at production
   training-loop time. The earlier this fails, the cheaper it is to fix.

4. **Resist scope creep into autograd.** Phase 1 is tensor primitives
   only. Every hour spent on autograd in Phase 1 is an hour Phase 2
   spends untangling.

## Pointers

- README.md — project overview and roadmap
- docs/adr/0001-eager-vs-lazy.md — eager-default decision
- docs/adr/0002-module-system.md — PyTorch-style modules decision
- docs/adr/0003-binding-strategy.md — FFI-over-mlx-c decision
- ext/mlx_c/upstream/mlx/c/ — the C headers Phase 1 binds against
- bin/setup — environment validator
