# frozen_string_literal: true

module MLX
  module Optimizers
    # Base learning-rate scheduler. Wraps an optimizer and mutates its
    # `lr` attribute on each `#step` call. Subclasses implement
    # `#compute_lr(step)`.
    class LRScheduler
      attr_reader :optimizer, :base_lr, :step_count

      def initialize(optimizer)
        @optimizer  = optimizer
        @base_lr    = optimizer.lr
        @step_count = 0
        @optimizer.lr = compute_lr(0)
      end

      def step
        @step_count += 1
        @optimizer.lr = compute_lr(@step_count)
        @optimizer.lr
      end

      def lr
        @optimizer.lr
      end

      protected

      def compute_lr(_step)
        raise NotImplementedError
      end
    end

    # Cosine decay with linear warmup. After `warmup_steps`, the LR follows
    # half a cosine from base_lr down to 0 over the remaining steps.
    class CosineSchedule < LRScheduler
      def initialize(optimizer, total_steps:, warmup_steps: 0)
        @total_steps  = total_steps.to_i
        @warmup_steps = warmup_steps.to_i
        super(optimizer)
      end

      protected

      def compute_lr(step)
        if step < @warmup_steps && @warmup_steps.positive?
          return @base_lr * (step.to_f / @warmup_steps)
        end

        decay_steps = [@total_steps - @warmup_steps, 1].max
        progress = ((step - @warmup_steps).to_f / decay_steps).clamp(0.0, 1.0)
        0.5 * @base_lr * (1.0 + Math.cos(Math::PI * progress))
      end
    end

    # Linear warmup from 0 to base_lr over `warmup_steps`, constant thereafter.
    class LinearWarmup < LRScheduler
      def initialize(optimizer, warmup_steps:)
        @warmup_steps = warmup_steps.to_i
        super(optimizer)
      end

      protected

      def compute_lr(step)
        return @base_lr if step >= @warmup_steps || @warmup_steps.zero?

        @base_lr * (step.to_f / @warmup_steps)
      end
    end
  end
end
