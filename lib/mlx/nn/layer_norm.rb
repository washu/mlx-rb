# frozen_string_literal: true

module MLX
  module NN
    # LayerNorm over the last dimension. Wraps mlx-c's fused fast_layer_norm
    # when available; falls back to elementwise ops otherwise (e.g. when
    # weight/bias aren't allocated). Eps matches mlx Python's default.
    class LayerNorm < Module
      attr_reader :dim, :eps

      def initialize(dim, eps: 1e-5, affine: true)
        super()
        @dim = dim
        @eps = eps.to_f
        return unless affine

        @weight = MLX::Array.ones([dim])
        @bias   = MLX::Array.zeros([dim])
      end

      def forward(x)
        raise MLX::TypeError, "LayerNorm expects MLX::Array" unless x.is_a?(MLX::Array)

        w = defined?(@weight) ? @weight.struct : MLX::FFI.null_array
        b = defined?(@bias)   ? @bias.struct   : MLX::FFI.null_array
        out = MLX::FFI.mlx_array_new
        MLX.check!(
          MLX::FFI.mlx_fast_layer_norm(out.pointer, x.struct, w, b, @eps, MLX.stream_struct),
          "mlx_fast_layer_norm"
        )
        MLX::Array.from_struct(out)
      end
    end
  end
end
