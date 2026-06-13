# frozen_string_literal: true

module MLX
  module NN
    # Linear layer with 4/8-bit quantized weights. Memory and matmul go
    # through mlx-c's fused quantized kernel; the public surface still looks
    # like {Linear} from the caller's perspective.
    #
    # Weight layout (matches mlx Python and the safetensors files HF ships
    # for pre-quantized models):
    #   * `@weight`  — packed uint32, shape `[out_features, in_features * bits / 32]`.
    #   * `@scales`  — fp16/fp32, shape `[out_features, in_features / group_size]`.
    #   * `@biases`  — fp16/fp32, same shape as `@scales`.
    #   * `@bias`    — optional fp32, shape `[out_features]`. Distinct from
    #                  `@biases` (the per-group dequant offset).
    #
    # `freeze` is the steady state — quantization-aware training is out of
    # scope for Phase 4. The layer is therefore frozen at construction and
    # `parameters` returns the bias only (the quantized weights are not
    # differentiable through this code path).
    class QuantizedLinear < Module
      attr_reader :in_features, :out_features, :bits, :group_size, :use_bias

      # Construct with random quantized weights. Mostly useful for tests; in
      # practice you'll either {.from_linear} an existing Linear or load
      # already-quantized weights from disk and call {#update}.
      def initialize(in_features, out_features, bits: 4, group_size: 64, bias: true)
        super()
        validate!(in_features, bits, group_size)
        @in_features  = in_features
        @out_features = out_features
        @bits         = bits
        @group_size   = group_size
        @use_bias     = bias

        scale = Math.sqrt(1.0 / in_features)
        dense = MLX::Array.random_uniform([out_features, in_features], low: -scale, high: scale)
        qw, scales, biases = MLX::Quantized.quantize(dense, bits: bits, group_size: group_size)
        @weight = qw
        @scales = scales
        @biases = biases
        @bias   = MLX::Array.zeros([out_features]) if bias

        freeze
      end

      # Quantize an existing dense Linear and return a new QuantizedLinear
      # with the same in/out shape. Copies the bias verbatim. Pass `bias:`
      # to override the auto-detection from the source layer.
      def self.from_linear(linear, bits: 4, group_size: 64)
        raise MLX::TypeError, "from_linear expects MLX::NN::Linear" unless linear.is_a?(MLX::NN::Linear)

        ql = allocate
        ql.send(:adopt_from_linear!, linear, bits: bits, group_size: group_size)
        ql
      end

      def forward(x)
        raise MLX::TypeError, "QuantizedLinear#forward expects MLX::Array" unless x.is_a?(MLX::Array)

        y = MLX::Quantized.quantized_matmul(
          x, @weight, @scales, @biases,
          bits: @bits, group_size: @group_size, transpose: true
        )
        @use_bias ? y + @bias : y
      end

      # Dense reconstruction. Useful for testing/inspection — not the path
      # forward inference takes.
      def dequantized_weight
        MLX::Quantized.dequantize(@weight, @scales, @biases,
                                  bits: @bits, group_size: @group_size)
      end

      # The packed weight isn't a fp parameter and shouldn't be touched by
      # optimizers; expose only the trainable bias to `named_parameters`.
      # `named_buffers` returns the quantized state for callers that need
      # the full tensor dictionary (e.g. checkpoint save).
      def named_parameters(prefix = "")
        out = {}
        out[join(prefix, "bias")] = @bias if @use_bias
        out
      end

      def named_buffers(prefix = "")
        {
          join(prefix, "weight") => @weight,
          join(prefix, "scales") => @scales,
          join(prefix, "biases") => @biases
        }
      end

      # Allow `update` to set the quantized state too, since the HF loader
      # ships pre-quantized weights named `weight`, `scales`, `biases`.
      def update(named)
        named.each do |path, arr|
          raise MLX::TypeError, "update value must be MLX::Array" unless arr.is_a?(MLX::Array)

          case path
          when "weight" then @weight = arr
          when "scales" then @scales = arr
          when "biases" then @biases = arr
          when "bias"   then @bias   = arr
          else
            raise ArgumentError, "QuantizedLinear has no parameter #{path.inspect}"
          end
        end
        self
      end

      private

      def join(prefix, name)
        prefix.empty? ? name : "#{prefix}.#{name}"
      end

      def validate!(in_features, bits, group_size)
        unless [2, 3, 4, 6, 8].include?(bits)
          raise ArgumentError, "bits must be one of 2,3,4,6,8 (got #{bits})"
        end
        unless (in_features % group_size).zero?
          raise ArgumentError,
                "in_features (#{in_features}) must be divisible by group_size (#{group_size})"
        end
        return if ((group_size * bits) % 32).zero?

        raise ArgumentError,
              "group_size * bits must be a multiple of 32 (got #{group_size * bits})"
      end

      def adopt_from_linear!(linear, bits:, group_size:)
        super_initialize_frame(linear, bits: bits, group_size: group_size)
        qw, scales, biases = MLX::Quantized.quantize(linear.instance_variable_get(:@weight),
                                                     bits: bits, group_size: group_size)
        @weight = qw
        @scales = scales
        @biases = biases
        if linear.instance_variable_get(:@use_bias)
          @bias = linear.instance_variable_get(:@bias)
        end
        freeze
      end

      def super_initialize_frame(linear, bits:, group_size:)
        Module.instance_method(:initialize).bind_call(self)
        validate!(linear.in_features, bits, group_size)
        @in_features  = linear.in_features
        @out_features = linear.out_features
        @bits         = bits
        @group_size   = group_size
        @use_bias     = linear.use_bias
      end
    end
  end
end
