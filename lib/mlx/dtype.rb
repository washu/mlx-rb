# frozen_string_literal: true

module MLX
  # Symbol-keyed dtype enum.
  #
  # Phase 0 left this as an open question; Phase 1 commits to symbols
  # (:float32 etc.) over constants (MLX::Float32) — it matches Torch.rb's
  # convention and reads cleanly in keyword arguments.
  #
  # Phase 1 exposes a subset of mlx_dtype: float32/float16/bfloat16/int32/
  # int64/bool. The rest of mlx_dtype is reachable via DType.to_c but is not
  # wired through MLX::Array yet.
  module DType
    NAME_TO_CODE = {
      bool:     MLX::FFI::MLX_BOOL,
      uint32:   MLX::FFI::MLX_UINT32,
      int32:    MLX::FFI::MLX_INT32,
      int64:    MLX::FFI::MLX_INT64,
      float16:  MLX::FFI::MLX_FLOAT16,
      float32:  MLX::FFI::MLX_FLOAT32,
      bfloat16: MLX::FFI::MLX_BFLOAT16
    }.freeze

    CODE_TO_NAME = NAME_TO_CODE.invert.freeze

    # Element width in bytes for the dtypes we support.
    BYTES = {
      bool:     1,
      uint32:   4,
      int32:    4,
      int64:    8,
      float16:  2,
      float32:  4,
      bfloat16: 2
    }.freeze

    module_function

    def to_c(dtype)
      NAME_TO_CODE.fetch(dtype) do
        raise MLX::DTypeError, "unknown dtype: #{dtype.inspect}"
      end
    end

    def from_c(code)
      CODE_TO_NAME.fetch(code) do
        raise MLX::DTypeError, "unsupported mlx_dtype code in Phase 1: #{code}"
      end
    end

    def bytesize(dtype)
      BYTES.fetch(dtype) do
        raise MLX::DTypeError, "unknown dtype: #{dtype.inspect}"
      end
    end
  end
end
