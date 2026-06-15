# frozen_string_literal: true

require "rbconfig"

# Top-level MLX namespace.
#
# `require "mlx"` validates the platform up front (per ADR 0003 we ship for
# Apple Silicon only) and loads the FFI bindings. Public API:
#
#   MLX.platform_supported?  -> Boolean
#   MLX.default_device       -> :gpu | :cpu
#   MLX.eval(*arrays)        -> forces evaluation of each MLX::Array
#   MLX.lazy { ... }         -> defers eager eval until the block returns
module MLX
  class Error < StandardError; end
  class FFIError       < Error; end
  class ShapeError     < Error; end
  class DTypeError     < Error; end
  class TypeError      < Error; end
  class PlatformError  < Error; end

  PLATFORM_MESSAGE = <<~MSG
    mlx-rb requires macOS on Apple Silicon (arm64-darwin). MLX is built on
    Metal and unified memory; there is no port to other platforms. Detected
    platform: %<host>s.
  MSG

  module_function

  def platform_supported?
    host = RbConfig::CONFIG["host_os"].to_s
    cpu  = RbConfig::CONFIG["host_cpu"].to_s
    host.start_with?("darwin") && %w[arm64 aarch64].include?(cpu)
  end

  def assert_platform_supported!
    return if platform_supported?

    raise PlatformError, format(PLATFORM_MESSAGE, host: RbConfig::CONFIG["host"])
  end

  # Re-raise an mlx-c return code as an exception. mlx-c uses 0 = success.
  def check!(rc, where)
    return if rc.zero?

    raise FFIError, "#{where} returned non-zero code #{rc}"
  end

  # Default device for ops. On a Metal-capable mlx-c build this is :gpu (the
  # unified Metal backend); on a CPU-only build we fall back to :cpu so the
  # gem stays usable for development on machines without the Metal toolchain.
  # Memoized after stream_struct is built.
  def default_device
    stream_struct # force resolution
    @default_device || :cpu
  end

  # Force evaluation of one or more MLX::Array. No-op for non-arrays so
  # callers can splat heterogeneous collections.
  def eval(*arrays)
    arrays.flatten.each do |arr|
      next unless arr.is_a?(MLX::Array)

      arr.eval!
    end
    nil
  end

  # Block form. Inside the block, ops construct arrays but skip the eager
  # `mlx_array_eval`. Arrays returned (or registered via the block result)
  # are evaluated at block exit. Nested lazy blocks are flattened — only the
  # outermost block performs the eval.
  def lazy
    state = (Thread.current[:mlx_lazy] ||= [])
    outer = state.empty?
    pending = []
    state << pending
    begin
      result = yield
      pending << result if result.is_a?(MLX::Array)
      if result.is_a?(::Array)
        result.each { |r| pending << r if r.is_a?(MLX::Array) }
      end
      result
    ensure
      state.pop
      if outer
        eval(*pending) # rubocop:disable Security/Eval — this is MLX.eval, not Kernel#eval
        Thread.current[:mlx_lazy] = nil
      end
    end
  end

  def lazy?
    state = Thread.current[:mlx_lazy]
    !state.nil? && !state.empty?
  end

  # Called by MLX::Array on every constructed array. Honors the current
  # lazy stack: under MLX.lazy we just register the array for later eval.
  def auto_eval(arr)
    state = Thread.current[:mlx_lazy]
    if state && !state.empty?
      state.last << arr
    else
      arr.eval!
    end
    arr
  end

  # The default mlx_stream we dispatch ops on. Built lazily on first use to
  # avoid touching libmlxc during require for unsupported platforms.
  def stream_struct
    @stream_struct ||= build_default_stream
  end

  def build_default_stream
    # Probe mlx-c's device count before asking for a GPU stream. Calling
    # mlx_default_gpu_stream_new on a CPU-only build prints to stderr and
    # aborts inside mlx-c, so we steer clear unless we're sure.
    if gpu_available?
      @default_device = :gpu
      MLX::FFI.mlx_default_gpu_stream_new
    else
      @default_device = :cpu
      MLX::FFI.mlx_default_cpu_stream_new
    end
  end

  def random_seed(seed)
    check!(MLX::FFI.mlx_random_seed(seed.to_i), "mlx_random_seed")
    seed.to_i
  end

  def gpu_available?
    return false unless MLX::FFI.respond_to?(:mlx_metal_is_available)

    out = ::FFI::MemoryPointer.new(:bool, 1)
    rc = MLX::FFI.mlx_metal_is_available(out)
    rc.zero? && out.read_uint8.positive?
  rescue StandardError
    false
  end
end

assert_load = lambda do
  MLX.assert_platform_supported!
  require_relative "mlx/version"
  require_relative "mlx/ffi"
  if MLX::FFI.load_error
    raise MLX::PlatformError,
          "failed to load libmlxc.dylib: #{MLX::FFI.load_error.message}. " \
          "Set MLX_C_LIB to the path of your built libmlxc.dylib, or run bin/setup."
  end
  require_relative "mlx/dtype"
  require_relative "mlx/array"
  require_relative "mlx/transforms"
  require_relative "mlx/quantized"
  require_relative "mlx/nn"
  require_relative "mlx/optimizers"
  require_relative "mlx/quantize_model"
  require_relative "mlx/attach_lora"
  require_relative "mlx/io"
  require_relative "mlx/models"
end

assert_load.call
