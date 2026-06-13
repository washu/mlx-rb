# frozen_string_literal: true

module MLX
  # Autodiff transforms backed by mlx-c's `mlx_value_and_grad` / closures.
  #
  # The interesting plumbing is the bridge from a Ruby block to an
  # mlx_closure: we register a Ruby callback whose signature matches
  # `int (*fun)(mlx_vector_array*, const mlx_vector_array)`, copy the input
  # arrays out of the incoming vector_array, call the user's block on them,
  # and re-pack the result into the output vector_array. mlx-c keeps the
  # callback alive only for the duration of the transform; we pin the
  # ::FFI::Function from Ruby to ensure GC doesn't collect it mid-call.
  module Transforms
    module_function

    # MLX.grad(fn) — given a callable returning a scalar loss, return a
    # callable that returns gradients wrt the listed argnums (default 0).
    # Each input must be an MLX::Array; gradients are returned as an
    # MLX::Array (single arg) or an Array of MLX::Array (multiple args).
    def grad(fn = nil, argnums: nil, &block)
      fn ||= block
      raise ArgumentError, "grad requires a callable or block" unless fn

      vag = value_and_grad(fn, argnums: argnums)
      lambda do |*inputs|
        _, g = vag.call(*inputs)
        g
      end
    end

    # MLX.value_and_grad(fn) — returns a callable that yields [value, grads].
    # Mirrors Python mlx.value_and_grad: value is the scalar loss (an
    # MLX::Array) and grads is one or many MLX::Array depending on argnums.
    def value_and_grad(fn = nil, argnums: nil, &block)
      fn ||= block
      raise ArgumentError, "value_and_grad requires a callable or block" unless fn

      lambda do |*inputs|
        inputs.each do |x|
          unless x.is_a?(MLX::Array)
            raise MLX::TypeError, "value_and_grad inputs must be MLX::Array, got #{x.class}"
          end
        end

        wants = Array(argnums || [0])
        wants = wants.map { |i| i.negative? ? i + inputs.size : i }

        # The closure trampoline copies inputs out of mlx_vector_array,
        # invokes the user's block, and writes the result back.
        callback_err = nil
        cb = ::FFI::Function.new(:int, [:pointer, MLX::FFI::MlxVectorArray.by_value]) do |out_ptr, in_vec|
          args = vector_to_arrays(in_vec)
          ret = fn.call(*args)
          outputs = ret.is_a?(::Array) ? ret : [ret]
          unless outputs.all? { |o| o.is_a?(MLX::Array) }
            raise MLX::TypeError, "closure must return MLX::Array (or array of them); got #{ret.class}"
          end

          # The C trampoline hands us a pre-allocated mlx_vector_array,
          # but its ctx is NULL (see private/vector.h::mlx_vector_array_new_).
          # We construct a populated vec via the public API and write
          # its ctx back through out_ptr so the trampoline can read it.
          fresh = MLX::FFI.mlx_vector_array_new
          outputs.each do |o|
            MLX.check!(
              MLX::FFI.mlx_vector_array_append_value(fresh, o.struct),
              "mlx_vector_array_append_value"
            )
          end
          out_ptr.write_pointer(fresh[:ctx])
          0
        rescue StandardError => e
          callback_err = e
          1
        end

        argnums_ptr = ::FFI::MemoryPointer.new(:int, wants.size)
        argnums_ptr.write_array_of_int(wants)

        # Build the mlx_closure. We hold cb in scope until the closure is
        # freed so the C trampoline doesn't reach into a freed FFI::Function.
        closure_struct = MLX::FFI.mlx_closure_new_func(cb)
        vag_struct = MLX::FFI.mlx_closure_value_and_grad_new

        begin
          MLX.check!(
            MLX::FFI.mlx_value_and_grad(vag_struct.pointer, closure_struct,
                                        argnums_ptr, wants.size),
            "mlx_value_and_grad"
          )

          # Apply the value_and_grad closure against the inputs.
          input_vec = build_input_vector(inputs)
          value_vec = MLX::FFI.mlx_vector_array_new
          grads_vec = MLX::FFI.mlx_vector_array_new
          MLX.check!(
            MLX::FFI.mlx_closure_value_and_grad_apply(
              value_vec.pointer, grads_vec.pointer, vag_struct, input_vec
            ),
            "mlx_closure_value_and_grad_apply"
          )

          raise callback_err if callback_err

          values = vector_to_arrays(value_vec)
          grads = vector_to_arrays(grads_vec)
          MLX::FFI.mlx_vector_array_free(input_vec)
          MLX::FFI.mlx_vector_array_free(value_vec)
          MLX::FFI.mlx_vector_array_free(grads_vec)

          value = values.size == 1 ? values.first : values
          grad_out = wants.size == 1 ? grads.first : grads
          [value, grad_out]
        ensure
          MLX::FFI.mlx_closure_value_and_grad_free(vag_struct)
          MLX::FFI.mlx_closure_free(closure_struct)
          # Keep cb alive across the call (no-op reference).
        end
      end
    end

    # ---- helpers ----

    def vector_to_arrays(vec)
      n = MLX::FFI.mlx_vector_array_size(vec).to_i
      (0...n).map do |i|
        slot = MLX::FFI.mlx_array_new
        MLX.check!(
          MLX::FFI.mlx_vector_array_get(slot.pointer, vec, i),
          "mlx_vector_array_get"
        )
        MLX::Array.from_struct(slot)
      end
    end

    def build_input_vector(arrays)
      vec = MLX::FFI.mlx_vector_array_new
      arrays.each do |arr|
        MLX.check!(
          MLX::FFI.mlx_vector_array_append_value(vec, arr.struct),
          "mlx_vector_array_append_value"
        )
      end
      vec
    end
  end

  module_function

  def grad(fn = nil, argnums: nil, &block)
    Transforms.grad(fn, argnums: argnums, &block)
  end

  def value_and_grad(fn = nil, argnums: nil, &block)
    Transforms.value_and_grad(fn, argnums: argnums, &block)
  end
end
