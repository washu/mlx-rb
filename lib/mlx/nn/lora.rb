# frozen_string_literal: true

module MLX
  module NN
    # Low-rank adapter (LoRA): `delta(x) = (x @ A) @ B * (alpha / rank)`.
    #
    # Shapes — input is `(..., in)`, output is `(..., out)`:
    #   * `@a` (in × rank), initialized Kaiming-uniform.
    #   * `@b` (rank × out), initialized zero so the adapter starts as
    #     an identity perturbation (the composite output equals the
    #     frozen base at step 0).
    #
    # `alpha` is the standard LoRA scaling. Effective scale is
    # `alpha / rank`, so doubling rank halves the update step at fixed
    # alpha.
    class LoRALinear < Module
      attr_reader :in_features, :out_features, :rank, :alpha

      def initialize(in_features, out_features, rank:, alpha: nil)
        super()
        raise ArgumentError, "rank must be positive" unless rank.positive?

        @in_features  = in_features
        @out_features = out_features
        @rank         = rank
        @alpha        = (alpha || rank).to_f
        @scale        = @alpha / rank

        scale = Math.sqrt(1.0 / in_features)
        @a = MLX::Array.random_uniform([in_features, rank], low: -scale, high: scale)
        @b = MLX::Array.zeros([rank, out_features])
      end

      def forward(x)
        raise MLX::TypeError, "LoRALinear#forward expects MLX::Array" unless x.is_a?(MLX::Array)

        x.matmul(@a).matmul(@b) * MLX::Array.new(@scale)
      end

      # Effective dense delta = A @ B * (alpha / rank). Useful for
      # folding the adapter into a base layer for export.
      def delta_weight
        @a.matmul(@b) * MLX::Array.new(@scale)
      end
    end

    # Composite: frozen base ({Linear} or {QuantizedLinear}) + trainable
    # {LoRALinear}. The base contributes its full forward; the LoRA adds
    # a low-rank perturbation.
    #
    # `named_parameters` returns *only* the LoRA pair so an optimizer
    # attached to this composite walks just the adapter — the base
    # weights stay where they are.
    class LoRAQuantizedLinear < Module
      attr_reader :base, :lora

      def initialize(base, rank:, alpha: nil)
        super()
        unless base.is_a?(Linear) || base.is_a?(QuantizedLinear)
          raise MLX::TypeError, "LoRAQuantizedLinear expects Linear or QuantizedLinear, got #{base.class}"
        end

        @base = base
        @lora = LoRALinear.new(base.in_features, base.out_features,
                               rank: rank, alpha: alpha)
        @base.freeze
      end

      def forward(x)
        @base.forward(x) + @lora.forward(x)
      end

      # Only the LoRA pair is trainable. The base's parameters and any
      # quantized buffers are hidden from `named_parameters` so an
      # optimizer walking this composite touches only the adapter.
      def named_parameters(prefix = "")
        out = {}
        @lora.named_parameters.each do |k, v|
          out[prefix.empty? ? "lora.#{k}" : "#{prefix}.lora.#{k}"] = v
        end
        out
      end

      # Convenience accessor for the underlying base weights and
      # buffers — used by adapter-aware checkpoint code and by
      # `MLX::IO.save_adapter` to skip the base.
      def base_state(prefix = "")
        case @base
        when QuantizedLinear
          { params: {}, buffers: @base.named_buffers(prefix.empty? ? "base" : "#{prefix}.base") }
        when Linear
          { params: @base.named_parameters(prefix.empty? ? "base" : "#{prefix}.base"), buffers: {} }
        end
      end
    end
  end
end
