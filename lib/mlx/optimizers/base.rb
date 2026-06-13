# frozen_string_literal: true

module MLX
  module Optimizers
    # Base optimizer. Subclasses implement `#apply_update(name, param, grad)`
    # which returns the new parameter value (an MLX::Array). The base class
    # owns the bookkeeping: stepping every named parameter, applying the
    # updates back to the module, and handling zero_grad.
    #
    # State per parameter (momentum buffers, Adam moments, etc.) lives in a
    # Ruby Hash keyed by parameter path. Gradient buffers are MLX::Arrays
    # supplied each step via `#step(grads_hash)`.
    class Optimizer
      attr_accessor :lr
      attr_reader   :module, :state, :step_count

      def initialize(model, lr:)
        @module     = model
        @lr         = lr.to_f
        @state      = {} # path => Hash of optimizer-specific buffers
        @step_count = 0
        @grads      = {} # path => MLX::Array (accumulated grads since zero_grad)
      end

      # Apply gradients (a hash matching `module.named_parameters`) to the
      # underlying parameters. Returns self.
      def step(grads)
        @step_count += 1
        params = @module.named_parameters
        updates = {}
        grads.each do |path, grad|
          param = params[path] or
            raise ArgumentError, "no parameter at #{path}"

          updates[path] = apply_update(path, param, grad)
        end
        @module.update(updates)
        @grads.clear
        self
      end

      # Clear any accumulated gradient buffers held by the optimizer. With
      # functional-style autograd you don't actually accumulate gradients —
      # `MLX.value_and_grad` returns fresh ones every call — so this is a
      # no-op in the common path. Kept for API symmetry with PyTorch.
      def zero_grad
        @grads.clear
        self
      end

      protected

      # Subclass hook.
      def apply_update(_path, _param, _grad)
        raise NotImplementedError
      end

      # Memoize a state slot for a parameter path. The block runs once.
      def state_slot(path)
        @state[path] ||= yield
      end
    end
  end
end
