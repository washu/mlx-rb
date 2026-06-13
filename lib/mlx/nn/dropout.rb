# frozen_string_literal: true

module MLX
  module NN
    # Inverted dropout. At inference (`training: false`) this is a no-op.
    # The mask is freshly sampled per call.
    class Dropout < Module
      attr_reader :p

      def initialize(p)
        super()
        @p = p.to_f
        raise ArgumentError, "p must be in [0, 1)" unless (0.0..1.0).cover?(@p) && @p < 1.0
      end

      def forward(x, training: true)
        return x unless training
        return x if @p.zero?

        keep = 1.0 - @p
        # Bernoulli(keep) mask. mlx_random_bernoulli wants a probability array.
        prob = MLX::Array.new(keep)
        out = MLX::FFI.mlx_array_new
        shape_ptr = MLX::Array.shape_pointer(x.shape)
        MLX.check!(
          MLX::FFI.mlx_random_bernoulli(out.pointer, prob.struct, shape_ptr, x.shape.size,
                                        MLX::FFI.null_array, MLX.stream_struct),
          "mlx_random_bernoulli"
        )
        mask = MLX::Array.from_struct(out).astype(:float32)
        x * mask * MLX::Array.new(1.0 / keep)
      end
    end
  end
end
