# frozen_string_literal: true

module MLX
  module NN
    # Scaled dot-product multi-head attention. Inputs are (batch, seq, dim);
    # output is the same shape. Uses mlx-c's fused
    # `mlx_fast_scaled_dot_product_attention` when available.
    class MultiHeadAttention < Module
      attr_reader :dim, :num_heads, :head_dim

      def initialize(dim, num_heads, bias: false)
        super()
        raise ArgumentError, "dim must be divisible by num_heads" unless (dim % num_heads).zero?

        @dim = dim
        @num_heads = num_heads
        @head_dim = dim / num_heads

        @q_proj = Linear.new(dim, dim, bias: bias)
        @k_proj = Linear.new(dim, dim, bias: bias)
        @v_proj = Linear.new(dim, dim, bias: bias)
        @out_proj = Linear.new(dim, dim, bias: bias)
      end

      # x: (B, T, D). Optional mask follows mlx's "mask_mode": pass :causal
      # for a triangular causal mask, or nil for no mask.
      def forward(x, mask: nil)
        raise MLX::TypeError, "expects MLX::Array" unless x.is_a?(MLX::Array)

        b, t, = x.shape
        q = @q_proj.call(x)
        k = @k_proj.call(x)
        v = @v_proj.call(x)

        # Reshape to (B, T, H, Dh) → transpose to (B, H, T, Dh)
        q = q.reshape([b, t, @num_heads, @head_dim]).transpose([0, 2, 1, 3])
        k = k.reshape([b, t, @num_heads, @head_dim]).transpose([0, 2, 1, 3])
        v = v.reshape([b, t, @num_heads, @head_dim]).transpose([0, 2, 1, 3])

        scale = 1.0 / Math.sqrt(@head_dim)
        mask_mode =
          case mask
          when nil      then ""
          when :causal  then "causal"
          else raise ArgumentError, "unsupported mask #{mask.inspect}"
          end

        out = MLX::FFI.mlx_array_new
        MLX.check!(
          MLX::FFI.mlx_fast_scaled_dot_product_attention(
            out.pointer, q.struct, k.struct, v.struct, scale,
            mask_mode, MLX::FFI.null_array, MLX::FFI.null_array,
            MLX.stream_struct
          ),
          "mlx_fast_scaled_dot_product_attention"
        )
        attn = MLX::Array.from_struct(out)

        # Back to (B, T, H, Dh) → (B, T, D)
        attn = attn.transpose([0, 2, 1, 3]).reshape([b, t, @dim])
        @out_proj.call(attn)
      end
    end
  end
end
