# frozen_string_literal: true

module MLX
  module NN
    # Embedding table — index integer tokens into a learned matrix.
    class Embedding < Module
      attr_reader :num_embeddings, :embedding_dim

      def initialize(num_embeddings, embedding_dim)
        super()
        @num_embeddings = num_embeddings
        @embedding_dim  = embedding_dim
        # Standard init: N(0, 1) scaled to mlx Python's default (mean=0, std=1).
        @weight = MLX::Array.random_normal([num_embeddings, embedding_dim])
      end

      def forward(indices)
        raise MLX::TypeError, "Embedding expects MLX::Array" unless indices.is_a?(MLX::Array)

        @weight.take(indices, axis: 0)
      end
    end
  end
end
