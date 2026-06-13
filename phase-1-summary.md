# Phase 1 — Summary

Phase 1 wires the FFI boundary against `mlx-c` and ships a working
`MLX::Array` for tensor primitives. Everything below is committed at the
end of Phase 1 and matches the in-tree behavior.

## What shipped

### FFI plumbing — `lib/mlx/ffi.rb`

- Single-pointer struct wrappers for `mlx_array`, `mlx_stream`,
  `mlx_device`, `mlx_string`. Passed `by_value` to mlx-c functions;
  AArch64 ABI marshals these as bare registers.
- Candidate-list library loader: respects `MLX_C_LIB`, then the vendored
  build path, then a small set of system locations. `ffi_lib` treats
  multiple arguments as "all-or-nothing", so we iterate manually and
  stop on the first dylib that opens.
- `attach_function` for the Phase 1 surface:
  - array: `mlx_array_new`, `mlx_array_new_data`, `mlx_array_free`,
    `mlx_array_ndim`, `mlx_array_shape`, `mlx_array_size`,
    `mlx_array_dtype`, `mlx_array_eval`,
    `mlx_array_item_float32`, `mlx_array_data_float32`
  - ops: `mlx_add`, `mlx_subtract`, `mlx_multiply`, `mlx_divide`,
    `mlx_matmul`, `mlx_reshape`, `mlx_transpose`,
    `mlx_transpose_axes`, `mlx_contiguous`, `mlx_zeros`, `mlx_ones`,
    `mlx_arange`, `mlx_full`
  - device/stream: `mlx_device_new_type`, `mlx_device_free`,
    `mlx_get_default_device`, `mlx_set_default_device`,
    `mlx_device_get_type`, `mlx_device_count`,
    `mlx_default_cpu_stream_new`, `mlx_default_gpu_stream_new`,
    `mlx_stream_free`, `mlx_synchronize`
- `FFI::AutoPointer` GC integration: a releaser proc reconstitutes the
  single-pointer `MlxArray` struct from the wrapped `ctx` pointer and
  calls `mlx_array_free` on finalization.

### `MLX::Array` — `lib/mlx/array.rb`

- Constructors: `.new(nested_ruby_array, dtype:)`, `.zeros`, `.ones`,
  `.arange(start, stop=nil, step=1, dtype:)` (Range-ish single-arg form
  supported), `.full(shape, value, dtype:)`.
- Introspection: `#shape`, `#ndim`, `#size`, `#dtype`.
- Arithmetic with broadcasting: `#+`, `#-`, `#*`, `#/`. Numeric and
  boolean scalars are coerced into 0-D MLX arrays via
  `mlx_array_new_data` so broadcasting falls out of mlx-c naturally.
- Linear algebra: `#matmul`.
- Shape ops: `#reshape`, `#transpose(axes=nil)` (delegates to either
  `mlx_transpose` or `mlx_transpose_axes`).
- Extraction: `#eval!`, `#to_a`, `#to_flat_a`, `#inspect`.
- Inputs are validated: non-rectangular nested arrays raise
  `MLX::ShapeError`.

### Top-level — `lib/mlx.rb`

- Error hierarchy: `MLX::Error < StandardError` with four specialized
  subclasses (`FFIError`, `ShapeError`, `DTypeError`, `TypeError`) plus
  `PlatformError` for require-time failures. Kept narrow per
  phase-0-setup.md's open-question note.
- `MLX.platform_supported?` checks `RbConfig::CONFIG["host_cpu"]` and
  `host_os`. A hard `MLX::PlatformError` is raised at require time on
  non-Darwin-arm64 hosts, and a second one is raised if libmlxc.dylib
  itself cannot be located.
- `MLX.default_device` probes `mlx_device_count(MLX_GPU)` and returns
  `:gpu` when Metal is built into mlx-c, falling back to `:cpu`
  otherwise. The chosen stream is memoized.
- `MLX.eval(*arrays)` and `MLX.lazy { ... }` are implemented with a
  thread-local stack. Inside `MLX.lazy`, eager evaluation is suppressed
  and any `MLX::Array` returned from the block (including those inside
  a returned Array) is `eval!`ed when the outermost block exits.

### `MLX::DType` — `lib/mlx/dtype.rb`

- Symbol-keyed enum: `:bool`, `:int32`, `:int64`, `:float16`,
  `:float32`, `:bfloat16`. Phase 0 left the symbol-vs-constant question
  open; Phase 1 commits to symbols (Torch.rb's idiom).
- `to_c`, `from_c`, and `bytesize` accessors. `from_c` raises a
  `DTypeError` rather than silently returning `nil` for unsupported
  codes.

### Tests — `spec/`

- `spec/support/python_oracle.rb`: shells out to `python3 -c` with a
  single self-contained script; inputs are JSON over stdin, results are
  JSON over stdout. `Open3.capture3` keeps the script and result paths
  out of the process arglist.
- `spec/mlx/array_spec.rb`: every op listed in the Phase 1 deliverables
  gets both a round-trip test (Ruby → MLX → `#to_a`) and a `:oracle`-
  tagged diff against Python `mlx`. Oracle tests are auto-skipped when
  `python3 -c "import mlx.core"` fails so CI without Python `mlx`
  installed still passes.
- `expect_close` helper in `spec_helper.rb` compares nested arrays with
  a configurable float tolerance.

### Acceptance check results

- `bundle exec rspec`: **30 examples, 0 failures**, including 12 oracle
  diffs. Runs in ~1.4s.
- `bin/console`: boots and
  `MLX::Array.new([[1,2],[3,4]]).matmul(MLX::Array.new([[1,0],[0,1]]))
  .to_a` returns `[[1.0, 2.0], [3.0, 4.0]]`.
- GC stress: 100,000 iterations each allocating three arrays (two
  inputs + one result), `GC.start` every 10k. RSS grew from 25.5 MB to
  28.9 MB across the whole run — a 3.4 MB ceiling that did not
  monotonically increase past iteration 50k, confirming
  `FFI::AutoPointer` is releasing mlx-c arrays as expected.

## Known gaps (intentional)

- **`#to_a` only materializes `:float32`.** Other dtypes parse through
  on construction (int32/int64/bool) but raise `NotImplementedError`
  on extraction. The deliverables only required
  `mlx_array_data_float32`; we'll wire the others in Phase 2.
- **`MLX::Array#@`.** The README example uses `a.matmul(b)`. The Phase
  1 brief mentions `#@(other)` as an alternative spelling. Ruby
  reserves the `@` sigil for ivars, so it cannot be a method name; only
  `#matmul` is available. Worth pinning in an ADR if a future phase
  wants operator-style matmul (`a * b` is already taken — we'd need
  `**` or similar).
- **Mixed-dtype binary ops** rely on mlx-c's type promotion. The Ruby
  side does not do its own promotion; if mlx-c rejects a combination
  we forward the error code.
- **Streams are global.** Phase 1 uses a single memoized default stream
  per process. Per-array stream selection / per-block `MLX.on(device)`
  is deferred.
- **`MLX_BUILD_METAL=OFF` is forced in our dev build** because the dev
  machine has Xcode CLT only, not the full Xcode (no `metal`
  compiler). The Ruby code works against either build flavor — GPU is
  picked when available — but Phase 1 was actually verified on a CPU
  build. Sanity-check on a Metal-enabled mlx-c is a Phase 2 entry
  task.
- **`MLX::Array.arange(stop)` single-arg form** mimics Ruby's `Range`
  (`[0, stop)`) rather than Python's signature. Documenting this
  divergence in the wrapper alone for now; revisit if it surprises
  users.
- **`#inspect` preview** is float32-aware; for other dtypes it prints
  `(non-float32 preview not implemented)`.

## mlx-c friction encountered

- `ffi_lib(*paths)` requires every path to load; you cannot pass it a
  fallback list. Worked around by walking the list manually. Worth
  flagging upstream to `ffi` or documenting in our README.
- `mlx_default_gpu_stream_new` on a CPU-only mlx-c build prints to
  stderr and aborts (`MLX error: [default_stream] Cannot get gpu stream
  without gpu backend.`) rather than returning a non-zero status that
  Ruby could rescue. We probe `mlx_device_count(MLX_GPU)` first. A
  recoverable error code from mlx-c would be friendlier.
- `mlx_array_data_float32` returns the raw underlying storage even
  after `mlx_array_eval`, which means a transpose view reads back its
  pre-transpose buffer. We force a `mlx_contiguous` copy before
  reading. This is correct but doubles memory for views going through
  `#to_a`. Not a bug in mlx-c, just a sharp edge.
- The C ABI requires that `mlx_array*` out-parameters point to an
  already-allocated `mlx_array` (typically from `mlx_array_new()`); the
  op internally calls `mlx_array_set`, which frees the previous
  context. Allocating a fresh empty `mlx_array_new()` per op is a real
  per-call cost (one extra alloc + one free). If hot paths show up in
  Phase 2 profiling, an "scratch" pool would help.

## Recommendations for Phase 2

1. **Build mlx-c with Metal enabled and re-run the suite.** All paths
   should already work but the verification matters. Block on it
   before Phase 2 implementation.
2. **Wire the remaining dtype extraction paths** (`mlx_array_data_int32`,
   `_int64`, `_uint8` for bool, `_float16`, `_bfloat16`). `#to_a` should
   stop raising NotImplementedError for any Phase 1-supported dtype.
3. **Autograd via `mlx_value_and_grad`** is the headline Phase 2 work.
   Ruby blocks → mlx-c closures is the interesting wiring. Start with
   a tiny end-to-end test: `MLX.grad(->(x) { (x * x).sum })`.
4. **`MLX::NN::Module`.** Phase 2 should land the auto-parameter
   registration. ADR 0002 is the spec; mirror Torch.rb's
   `instance_variable_set` hook rather than Python MLX's tree pattern.
5. **Drop `MLX::Array.allocate` + `send(:adopt!, ...)` dance** in
   favor of a private factory module method. Currently fine but the
   `send` is a code smell.
6. **Move libmlxc discovery into `bin/setup`** so users don't have to
   set `MLX_C_LIB`/`DYLD_LIBRARY_PATH` manually. Either install the
   built dylib into the gem's install tree, or generate a small Ruby
   config file with the resolved absolute paths at build time.
7. **Add `MLX::Array#==` and friends.** Not needed for Phase 1 but
   tests already want them; right now we compare `#to_a` results.
8. **Profile FFI overhead vs Python mlx** on a matmul-heavy
   microbenchmark per ADR 0003's validation milestone. If it lands
   under 5% we're free to keep the FFI-only strategy; if not, reopen
   the ADR.
