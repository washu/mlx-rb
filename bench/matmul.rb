# frozen_string_literal: true

# Head-to-head matmul benchmark: mlx-rb vs Python mlx.
#
# Reports best-of-N wall time in ms/op. Python column shows "n/a" if
# Python mlx isn't importable.

require "mlx"
require "json"
require "open3"

SIZE  = Integer(ENV["BENCH_SIZE"] || 2048)
ITERS = Integer(ENV["BENCH_ITERS"] || 20)

a = MLX::Array.random_normal([SIZE, SIZE])
b = MLX::Array.random_normal([SIZE, SIZE])
# Warm-up.
3.times { a.matmul(b).eval! }

times = []
ITERS.times do
  t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  c = a.matmul(b)
  c.eval!
  times << (Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0)
end
rb_best = times.min * 1000.0

py_best = begin
  script = <<~PY
    import time, sys, mlx.core as mx
    n = #{SIZE}
    a = mx.random.normal([n, n])
    b = mx.random.normal([n, n])
    for _ in range(3): mx.eval(a @ b)
    times = []
    for _ in range(#{ITERS}):
        t0 = time.perf_counter()
        c = a @ b
        mx.eval(c)
        times.append(time.perf_counter() - t0)
    print(min(times) * 1000.0)
  PY
  stdout, _, status = Open3.capture3("python3", "-c", script)
  status.success? ? Float(stdout.strip) : nil
end

if py_best
  overhead = ((rb_best - py_best) / py_best * 100).round(1)
  printf "matmul %dx%d  mlx-rb=%6.2f ms/op  mlx-py=%6.2f ms/op  overhead=%+.1f%%\n",
         SIZE, SIZE, rb_best, py_best, overhead
else
  printf "matmul %dx%d  mlx-rb=%6.2f ms/op  mlx-py=   n/a\n", SIZE, SIZE, rb_best
end
