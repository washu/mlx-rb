# frozen_string_literal: true

module MLX
  # Phase 4 — quantization. Thin Ruby wrappers around mlx-c's
  # `mlx_quantize`, `mlx_dequantize`, and `mlx_quantized_matmul`.
  #
  # mlx-c packs the K-axis of weights into uint32 words: for `bits=4` each
  # uint32 holds 8 weights, so a weight matrix of shape `[out, in]` becomes a
  # packed matrix of shape `[out, in * bits / 32]`. `scales` and `biases` are
  # one fp16/fp32 value per group of `group_size` weights along K, shape
  # `[out, in / group_size]`. We don't try to hide this — the three returned
  # arrays are the canonical mlx representation, and downstream ops
  # (`quantized_matmul`, `QuantizedLinear`) consume them in that shape.
  module Quantized
    AFFINE = "affine"

    module_function

    # Quantize a 2-D weight matrix.
    #
    # @param weights [MLX::Array] shape `[out_features, in_features]`, fp32 or fp16.
    # @param bits [Integer] 2, 3, 4, 6, or 8. Default 4.
    # @param group_size [Integer] number of weights sharing one scale/bias.
    #   Default 64. Must divide `in_features`.
    # @return [Array(MLX::Array, MLX::Array, MLX::Array)] `[qw, scales, biases]`.
    def quantize(weights, bits: 4, group_size: 64)
      raise MLX::TypeError, "quantize expects MLX::Array" unless weights.is_a?(MLX::Array)

      vec = MLX::FFI.mlx_vector_array_new
      rc = MLX::FFI.mlx_quantize(
        vec.pointer,
        weights.struct,
        MLX::FFI.opt_int(group_size),
        MLX::FFI.opt_int(bits),
        AFFINE,
        MLX::FFI.null_array,
        MLX.stream_struct
      )
      MLX.check!(rc, "mlx_quantize")

      size = MLX::FFI.mlx_vector_array_size(vec)
      raise MLX::FFIError, "mlx_quantize returned #{size} arrays, expected 3" unless size == 3

      results = (0...size).map do |i|
        out = MLX::FFI.mlx_array_new
        MLX.check!(MLX::FFI.mlx_vector_array_get(out.pointer, vec, i), "mlx_vector_array_get")
        MLX::Array.from_struct(out)
      end
      MLX::FFI.mlx_vector_array_free(vec)
      results
    end

    # Dequantize back to a dense fp16/fp32 array.
    #
    # @return [MLX::Array]
    def dequantize(qw, scales, biases, bits: 4, group_size: 64)
      [qw, scales, biases].each do |a|
        raise MLX::TypeError, "dequantize expects MLX::Array" unless a.is_a?(MLX::Array)
      end

      out = MLX::FFI.mlx_array_new
      rc = MLX::FFI.mlx_dequantize(
        out.pointer,
        qw.struct, scales.struct, biases.struct,
        MLX::FFI.opt_int(group_size),
        MLX::FFI.opt_int(bits),
        AFFINE,
        MLX::FFI.null_array,
        MLX::FFI.opt_dtype(nil),
        MLX.stream_struct
      )
      MLX.check!(rc, "mlx_dequantize")
      MLX::Array.from_struct(out)
    end

    # x @ dequantize(qw, scales, biases)^T.
    #
    # `transpose=true` (the default) matches the orientation used by
    # `QuantizedLinear`: weights are stored `[out, in]` and the matmul is
    # `x @ W^T`. mlx-c fuses the dequant+matmul in one Metal kernel.
    def quantized_matmul(x, qw, scales, biases, bits: 4, group_size: 64, transpose: true)
      [x, qw, scales, biases].each do |a|
        raise MLX::TypeError, "quantized_matmul expects MLX::Array" unless a.is_a?(MLX::Array)
      end

      out = MLX::FFI.mlx_array_new
      rc = MLX::FFI.mlx_quantized_matmul(
        out.pointer,
        x.struct, qw.struct, scales.struct, biases.struct,
        transpose,
        MLX::FFI.opt_int(group_size),
        MLX::FFI.opt_int(bits),
        AFFINE,
        MLX.stream_struct
      )
      MLX.check!(rc, "mlx_quantized_matmul")
      MLX::Array.from_struct(out)
    end
  end

  module_function

  # Convenience top-level aliases — match the Python mlx API surface.
  def quantize(weights, bits: 4, group_size: 64)
    Quantized.quantize(weights, bits: bits, group_size: group_size)
  end

  def dequantize(qw, scales, biases, bits: 4, group_size: 64)
    Quantized.dequantize(qw, scales, biases, bits: bits, group_size: group_size)
  end

  def quantized_matmul(x, qw, scales, biases, bits: 4, group_size: 64, transpose: true)
    Quantized.quantized_matmul(x, qw, scales, biases,
                               bits: bits, group_size: group_size, transpose: transpose)
  end
end
