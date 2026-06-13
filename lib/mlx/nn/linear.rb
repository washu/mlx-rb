# frozen_string_literal: true

module MLX
  module NN
    # y = x @ W^T + b. W is (out_features, in_features), matching mlx Python.
    class Linear < Module
      attr_reader :in_features, :out_features, :use_bias

      def initialize(in_features, out_features, bias: true)
        super()
        @in_features = in_features
        @out_features = out_features
        @use_bias = bias

        # Kaiming-uniform init: scale = sqrt(1 / in_features).
        scale = Math.sqrt(1.0 / in_features)
        @weight = MLX::Array.random_uniform([out_features, in_features], low: -scale, high: scale)
        @bias   = MLX::Array.zeros([out_features]) if bias
      end

      def forward(x)
        raise MLX::TypeError, "Linear#forward expects MLX::Array, got #{x.class}" unless x.is_a?(MLX::Array)

        y = x.matmul(@weight.transpose)
        @use_bias ? y + @bias : y
      end
    end
  end
end
