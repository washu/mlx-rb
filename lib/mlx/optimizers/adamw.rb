# frozen_string_literal: true

module MLX
  module Optimizers
    # AdamW optimizer. Decoupled weight decay (Loshchilov & Hutter, 2019),
    # matching the PyTorch/HF and mlx.optimizers.AdamW formulations.
    #
    # Update rule per step t:
    #   m <- beta1 * m + (1 - beta1) * g
    #   v <- beta2 * v + (1 - beta2) * g * g
    #   m_hat <- m / (1 - beta1^t)
    #   v_hat <- v / (1 - beta2^t)
    #   p <- p - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)
    class AdamW < Optimizer
      attr_accessor :betas, :eps, :weight_decay

      def initialize(model, lr:, betas: [0.9, 0.999], eps: 1e-8, weight_decay: 0.01)
        super(model, lr: lr)
        @betas = betas.map(&:to_f)
        @eps = eps.to_f
        @weight_decay = weight_decay.to_f
      end

      protected

      def apply_update(path, param, grad)
        beta1, beta2 = @betas
        slot = state_slot(path) do
          {
            m: MLX::Array.zeros(param.shape, dtype: param.dtype),
            v: MLX::Array.zeros(param.shape, dtype: param.dtype)
          }
        end

        slot[:m] = (slot[:m] * beta1) + (grad * (1.0 - beta1))
        slot[:v] = (slot[:v] * beta2) + (grad * grad * (1.0 - beta2))

        bc1 = 1.0 - (beta1**@step_count)
        bc2 = 1.0 - (beta2**@step_count)
        m_hat = slot[:m] / bc1
        v_hat = slot[:v] / bc2

        update = m_hat / (v_hat.sqrt + @eps)
        update += (param * @weight_decay) if @weight_decay.positive?
        param - (update * @lr)
      end
    end
  end
end
