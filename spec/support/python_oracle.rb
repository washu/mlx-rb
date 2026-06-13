# frozen_string_literal: true

require "json"
require "open3"
require "shellwords"

# Compute the same op via Python `mlx` and return the result as a Ruby array.
#
# Usage:
#   PythonOracle.run(:add, [[1, 2], [3, 4]], [[5, 6], [7, 8]])
#   PythonOracle.run(:matmul, [[1, 2], [3, 4]], [[1, 0], [0, 1]])
#   PythonOracle.run(:arange, args: [0, 5, 1])
#   PythonOracle.run(:zeros, args: [[2, 3]])
#
# We shell out to `python3 -c` with a minimal script and pass the inputs as
# JSON on stdin. The oracle is intentionally narrow: one op per call, only
# the ops Phase 1 needs.
module PythonOracle
  class Error < StandardError; end
  class Unavailable < Error; end

  PYTHON_SCRIPT = <<~PY
    import json, sys
    try:
        import mlx.core as mx
    except Exception as e:
        sys.stderr.write("oracle:import-failed:" + str(e))
        sys.exit(2)

    payload = json.load(sys.stdin)
    op = payload["op"]
    args = payload.get("args", [])
    inputs = payload.get("inputs", [])
    dtype = getattr(mx, payload.get("dtype", "float32"))

    arrs = [mx.array(x, dtype=dtype) for x in inputs]

    def to_list(a):
        mx.eval(a)
        return a.tolist()

    if op == "add":        out = arrs[0] + arrs[1]
    elif op == "subtract": out = arrs[0] - arrs[1]
    elif op == "multiply": out = arrs[0] * arrs[1]
    elif op == "divide":   out = arrs[0] / arrs[1]
    elif op == "matmul":   out = arrs[0] @ arrs[1]
    elif op == "reshape":  out = mx.reshape(arrs[0], args[0])
    elif op == "transpose":
        out = mx.transpose(arrs[0], args[0]) if args else mx.transpose(arrs[0])
    elif op == "zeros":    out = mx.zeros(args[0], dtype=dtype)
    elif op == "ones":     out = mx.ones(args[0], dtype=dtype)
    elif op == "arange":
        start, stop, step = args
        out = mx.arange(start, stop, step, dtype=dtype)
    elif op == "full":     out = mx.full(args[0], args[1], dtype=dtype)
    else:
        sys.stderr.write("oracle:unknown-op:" + op)
        sys.exit(3)

    json.dump(to_list(out), sys.stdout)
  PY

  module_function

  def available?
    return @available unless @available.nil?

    _, status = Open3.capture2e("python3", "-c", "import mlx.core")
    @available = status.success?
  end

  def run(op, *inputs, args: [], dtype: "float32")
    raise Unavailable, "python3 with `mlx` is not importable" unless available?

    payload = JSON.dump(op: op.to_s, inputs: inputs, args: args, dtype: dtype)
    stdout, stderr, status = Open3.capture3("python3", "-c", PYTHON_SCRIPT, stdin_data: payload)
    unless status.success?
      raise Error, "python oracle failed (#{status.exitstatus}): #{stderr.strip}"
    end

    JSON.parse(stdout)
  end

  # Run an arbitrary Python snippet against `mlx.core`. The snippet should
  # write a JSON-serialisable result to stdout via the helper `emit(obj)`.
  # `inputs` is passed in via `INPUTS` (a Python list). `args` is `ARGS`.
  def run_script(snippet, inputs: [], args: [])
    raise Unavailable, "python3 with `mlx` is not importable" unless available?

    payload = JSON.dump(inputs: inputs, args: args)
    wrapper = <<~PY
      import json, sys
      try:
          import mlx.core as mx
          import mlx.nn as mxnn
      except Exception as e:
          sys.stderr.write("oracle:import-failed:" + str(e))
          sys.exit(2)

      payload = json.load(sys.stdin)
      INPUTS = payload.get("inputs", [])
      ARGS = payload.get("args", [])

      def emit(obj):
          if hasattr(obj, "tolist"):
              mx.eval(obj)
              obj = obj.tolist()
          json.dump(obj, sys.stdout)

      #{snippet}
    PY
    stdout, stderr, status = Open3.capture3("python3", "-c", wrapper, stdin_data: payload)
    unless status.success?
      raise Error, "python oracle script failed (#{status.exitstatus}): #{stderr.strip}"
    end

    JSON.parse(stdout)
  end
end
