# frozen_string_literal: true

module MLX
  module NN
    # RMSNorm via mlx-c's fused `mlx_fast_rms_norm`. Weight is optional in
    # mlx-c but we always allocate one (initialized to ones) so the
    # parameter list stays stable across configurations.
    class RMSNorm < Module
      attr_reader :dim, :eps

      def initialize(dim, eps: 1e-5)
        super()
        @dim = dim
        @eps = eps.to_f
        @weight = MLX::Array.ones([dim])
      end

      def forward(x)
        raise MLX::TypeError, "RMSNorm expects MLX::Array" unless x.is_a?(MLX::Array)

        out = MLX::FFI.mlx_array_new
        MLX.check!(
          MLX::FFI.mlx_fast_rms_norm(out.pointer, x.struct, @weight.struct, @eps, MLX.stream_struct),
          "mlx_fast_rms_norm"
        )
        MLX::Array.from_struct(out)
      end
    end
  end
end
