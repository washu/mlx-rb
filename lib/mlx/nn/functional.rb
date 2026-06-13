# frozen_string_literal: true

module MLX
  module NN
    # Stateless layer ops. These are pure functions: take MLX::Array,
    # return MLX::Array. Trainable counterparts live in concrete Module
    # subclasses that wrap these.
    module F
      module_function

      def relu(x)
        zero = MLX::Array.new(0.0)
        x.maximum(zero)
      end

      def silu(x)
        # x * sigmoid(x) — Swish/SiLU
        x * x.sigmoid
      end

      # GELU approximation matching mlx Python (`approx="tanh"`):
      #   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))
      def gelu(x)
        c = MLX::Array.new(0.7978845608028654) # sqrt(2/pi)
        k = MLX::Array.new(0.044715)
        half = MLX::Array.new(0.5)
        one = MLX::Array.new(1.0)
        inner = c * (x + k * x * x * x)
        half * x * (one + inner.tanh)
      end

      def softmax(x, axis: -1)
        x.softmax(axis: axis)
      end

      # log_softmax(x) = x - logsumexp(x). mlx-c does not expose
      # log_softmax directly; rederive from logsumexp.
      def log_softmax(x, axis: -1)
        x - x.logsumexp(axes: axis, keepdims: true)
      end

      # Cross entropy from raw logits. `targets` is the integer label
      # array (shape = logits.shape minus the class axis). Reduces with
      # `reduction:` :mean | :sum | :none.
      def cross_entropy(logits, targets, axis: -1, reduction: :mean)
        log_p = log_softmax(logits, axis: axis)
        # Gather log-probabilities of the targets along `axis`.
        targets = targets.astype(:int32) if targets.dtype != :int32
        # expand_dims so we can take_along_axis
        idx = targets.expand_dims(axis)
        gathered = take_along_axis(log_p, idx, axis: axis)
        nll = -gathered.sum(axes: axis)
        case reduction
        when :mean then nll.mean
        when :sum  then nll.sum
        when :none then nll
        else raise ArgumentError, "unknown reduction #{reduction.inspect}"
        end
      end

      def mse_loss(pred, target, reduction: :mean)
        diff = pred - target
        sq = diff * diff
        case reduction
        when :mean then sq.mean
        when :sum  then sq.sum
        when :none then sq
        else raise ArgumentError, "unknown reduction #{reduction.inspect}"
        end
      end

      # take_along_axis isn't exposed by Phase-2 FFI; do it via take + axis
      # arithmetic. For 2-D logits (N, C) and 1-D targets (N,), this reduces
      # to a row-wise gather which we implement with mlx_take_axis.
      def take_along_axis(x, indices, axis:)
        # mlx_take_axis takes a 1-D index array per axis — works when
        # indices has the same rank as x. For the common 2-D classification
        # case we implement it by reshaping.
        ax = axis.negative? ? axis + x.ndim : axis
        unless x.ndim == 2 && indices.ndim == 2 && ax == x.ndim - 1
          raise NotImplementedError, "take_along_axis only handles 2-D row gather in Phase 2"
        end

          # Row gather: build flat indices = row * C + targets, take flat.
        c = x.shape[1]
        rows = x.shape[0]
        row_offsets = MLX::Array.arange(0, rows, 1, dtype: :int32) *
                      MLX::Array.new(c, dtype: :int32)
        flat_idx = row_offsets.expand_dims(1) + indices
        flat = x.reshape([rows * c])
        flat.take(flat_idx.reshape([rows]))
            .reshape([rows, 1])
      end
    end
  end
end
