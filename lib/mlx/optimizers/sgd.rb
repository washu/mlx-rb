# frozen_string_literal: true

module MLX
  module Optimizers
    # Stochastic gradient descent with optional momentum and weight decay.
    #
    # Update rule (matches PyTorch / mlx.optimizers.SGD):
    #   if weight_decay > 0: g <- g + wd * p
    #   if momentum > 0:    v <- momentum * v + g; update = v
    #   else:               update = g
    #   p <- p - lr * update
    class SGD < Optimizer
      attr_accessor :momentum, :weight_decay

      def initialize(model, lr:, momentum: 0.0, weight_decay: 0.0)
        super(model, lr: lr)
        @momentum = momentum.to_f
        @weight_decay = weight_decay.to_f
      end

      protected

      def apply_update(path, param, grad)
        g = if @weight_decay.positive?
              grad + (param * @weight_decay)
            else
              grad
            end

        update =
          if @momentum.positive?
            slot = state_slot(path) { { v: MLX::Array.zeros(param.shape, dtype: param.dtype) } }
            slot[:v] = (slot[:v] * @momentum) + g
            slot[:v]
          else
            g
          end

        param - (update * @lr)
      end
    end
  end
end
