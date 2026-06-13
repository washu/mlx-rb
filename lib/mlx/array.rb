# frozen_string_literal: true

require "ffi"

module MLX
  # MLX::Array — an N-dimensional tensor backed by an mlx-c array.
  #
  # Construction:
  #   MLX::Array.new([[1, 2], [3, 4]])              # 2x2 float32
  #   MLX::Array.zeros([3, 4])                      # zeros
  #   MLX::Array.ones([3, 4], dtype: :float32)      # ones
  #   MLX::Array.arange(0, 10, 1)                   # 1-D range
  #
  # Operations return new MLX::Array instances. Per ADR 0001 evaluation is
  # eager outside of MLX.lazy { ... } — every op forces an `mlx_array_eval`
  # before returning. Inside a lazy block the eval is deferred to block exit.
  class Array
    attr_reader :struct

    # Public initializer. Accepts a nested Ruby array (or a scalar Numeric) and
    # the desired dtype. For Phase 1 only :float32 is fully wired through both
    # construction and #to_a; other dtypes are accepted on construction but
    # extraction will raise.
    def initialize(input, dtype: :float32)
      case input
      when MLX::FFI::MlxArray
        # Internal construction path (used by ops). Wrap an existing struct.
        adopt!(input)
      when ::Array, Numeric
        shape, flat = self.class.infer_shape_and_flatten(input)
        adopt!(self.class.build_from_buffer(flat, shape, dtype))
      else
        raise MLX::TypeError, "cannot build MLX::Array from #{input.class}"
      end
      MLX.auto_eval(self)
    end

    # ---- Constructors ----

    def self.zeros(shape, dtype: :float32)
      shape = Array.normalize_shape(shape)
      shape_ptr = Array.shape_pointer(shape)
      out = MLX::FFI.mlx_array_new
      rc = MLX::FFI.mlx_zeros(out.pointer, shape_ptr, shape.size, MLX::DType.to_c(dtype), MLX.stream_struct)
      MLX.check!(rc, "mlx_zeros")
      from_struct(out)
    end

    def self.ones(shape, dtype: :float32)
      shape = Array.normalize_shape(shape)
      shape_ptr = Array.shape_pointer(shape)
      out = MLX::FFI.mlx_array_new
      rc = MLX::FFI.mlx_ones(out.pointer, shape_ptr, shape.size, MLX::DType.to_c(dtype), MLX.stream_struct)
      MLX.check!(rc, "mlx_ones")
      from_struct(out)
    end

    def self.arange(start, stop = nil, step = 1, dtype: :float32)
      # MLX's arange takes (start, stop, step). Mimic Range semantics where
      # only one positional argument means [0, start).
      if stop.nil?
        stop = start
        start = 0
      end
      out = MLX::FFI.mlx_array_new
      rc = MLX::FFI.mlx_arange(out.pointer, start.to_f, stop.to_f, step.to_f,
                               MLX::DType.to_c(dtype), MLX.stream_struct)
      MLX.check!(rc, "mlx_arange")
      from_struct(out)
    end

    # Random normal samples, dtype :float32 by default. key (an MLX::Array
     # produced by MLX.random_key) is optional; when nil mlx-c uses the
     # global RNG seeded by mlx_random_seed.
    def self.random_normal(shape, loc: 0.0, scale: 1.0, dtype: :float32, key: nil)
      shape = Array.normalize_shape(shape)
      shape_ptr = Array.shape_pointer(shape)
      key_struct = key.is_a?(MLX::Array) ? key.struct : MLX::FFI.null_array
      out = MLX::FFI.mlx_array_new
      rc = MLX::FFI.mlx_random_normal(out.pointer, shape_ptr, shape.size,
                                      MLX::DType.to_c(dtype), loc.to_f, scale.to_f,
                                      key_struct, MLX.stream_struct)
      MLX.check!(rc, "mlx_random_normal")
      from_struct(out)
    end

    def self.random_uniform(shape, low: 0.0, high: 1.0, dtype: :float32, key: nil)
      shape = Array.normalize_shape(shape)
      shape_ptr = Array.shape_pointer(shape)
      low_arr = MLX::Array.new(low.to_f)
      high_arr = MLX::Array.new(high.to_f)
      key_struct = key.is_a?(MLX::Array) ? key.struct : MLX::FFI.null_array
      out = MLX::FFI.mlx_array_new
      rc = MLX::FFI.mlx_random_uniform(out.pointer, low_arr.struct, high_arr.struct,
                                       shape_ptr, shape.size, MLX::DType.to_c(dtype),
                                       key_struct, MLX.stream_struct)
      MLX.check!(rc, "mlx_random_uniform")
      from_struct(out)
    end

    def self.full(shape, value, dtype: :float32)
      shape = Array.normalize_shape(shape)
      shape_ptr = Array.shape_pointer(shape)
      scalar = scalar_array(value, dtype)
      out = MLX::FFI.mlx_array_new
      rc = MLX::FFI.mlx_full(out.pointer, shape_ptr, shape.size, scalar.struct,
                             MLX::DType.to_c(dtype), MLX.stream_struct)
      MLX.check!(rc, "mlx_full")
      from_struct(out)
    end

    # Build an MLX::Array around an MlxArray struct returned from mlx-c.
    # Skips the public initializer's Ruby->mlx conversion.
    def self.from_struct(struct)
      inst = allocate
      inst.send(:adopt!, struct)
      MLX.auto_eval(inst)
      inst
    end

    # Build an MLX::Array from a raw byte buffer (used by IO loaders).
    # `buffer` is an ::FFI::Pointer or ::FFI::MemoryPointer into the source
    # bytes, `shape` is an Array of Integer dims, `dtype` a symbol. mlx-c
    # copies the buffer, so callers don't have to keep it alive.
    def self.from_buffer(buffer, shape, dtype)
      shape = normalize_shape(shape.dup)
      shape_ptr = shape_pointer(shape)
      struct = MLX::FFI.mlx_array_new_data(buffer, shape_ptr, shape.size, MLX::DType.to_c(dtype))
      from_struct(struct)
    end

    # ---- Introspection ----

    def ndim
      MLX::FFI.mlx_array_ndim(@struct).to_i
    end

    def size
      MLX::FFI.mlx_array_size(@struct).to_i
    end

    def shape
      n = ndim
      return [] if n.zero?

      ptr = MLX::FFI.mlx_array_shape(@struct)
      ptr.read_array_of_int(n)
    end

    def dtype
      MLX::DType.from_c(MLX::FFI.mlx_array_dtype(@struct))
    end

    # ---- Arithmetic ----

    def +(other)
      binary_op(:mlx_add, other)
    end

    def -(other)
      binary_op(:mlx_subtract, other)
    end

    def *(other)
      binary_op(:mlx_multiply, other)
    end

    def /(other)
      binary_op(:mlx_divide, other)
    end

    def matmul(other)
      binary_op(:mlx_matmul, other)
    end

    def -@
      unary_op(:mlx_negative)
    end

    def **(other)
      binary_op(:mlx_power, other)
    end

    def maximum(other)
      binary_op(:mlx_maximum, other)
    end

    def equal(other)
      binary_op(:mlx_equal, other)
    end

    # ---- Elementwise unary math ----

    def exp
      unary_op(:mlx_exp)
    end

    def log
      unary_op(:mlx_log)
    end

    def sqrt
      unary_op(:mlx_sqrt)
    end

    def rsqrt
      unary_op(:mlx_rsqrt)
    end

    def square
      unary_op(:mlx_square)
    end

    def abs
      unary_op(:mlx_abs)
    end

    def sigmoid
      unary_op(:mlx_sigmoid)
    end

    def tanh
      unary_op(:mlx_tanh)
    end

    def erf
      unary_op(:mlx_erf)
    end

    def sin
      unary_op(:mlx_sin)
    end

    def cos
      unary_op(:mlx_cos)
    end

    def stop_gradient
      unary_op(:mlx_stop_gradient)
    end

    # ---- Reductions ----

    def sum(axes: nil, keepdims: false)
      reduction_op(:mlx_sum, :mlx_sum_axes, axes, keepdims)
    end

    def mean(axes: nil, keepdims: false)
      reduction_op(:mlx_mean, :mlx_mean_axes, axes, keepdims)
    end

    def logsumexp(axes: nil, keepdims: false)
      axes = axes_array(axes)
      out = MLX::FFI.mlx_array_new
      if axes.nil?
        # logsumexp_axes with axes = all dims, then keepdims
        all_axes = (0...ndim).to_a
        ptr = Array.shape_pointer(all_axes)
        MLX.check!(
          MLX::FFI.mlx_logsumexp_axes(out.pointer, @struct, ptr, all_axes.size, keepdims, MLX.stream_struct),
          "mlx_logsumexp_axes"
        )
      else
        ptr = Array.shape_pointer(axes)
        MLX.check!(
          MLX::FFI.mlx_logsumexp_axes(out.pointer, @struct, ptr, axes.size, keepdims, MLX.stream_struct),
          "mlx_logsumexp_axes"
        )
      end
      self.class.from_struct(out)
    end

    def softmax(axis: -1, precise: false)
      axes = [axis_to_positive(axis)]
      ptr = Array.shape_pointer(axes)
      out = MLX::FFI.mlx_array_new
      MLX.check!(
        MLX::FFI.mlx_softmax_axes(out.pointer, @struct, ptr, axes.size, precise, MLX.stream_struct),
        "mlx_softmax_axes"
      )
      self.class.from_struct(out)
    end

    # ---- Shape / indexing ----

    def broadcast_to(shape)
      shape = Array.normalize_shape(shape)
      ptr = Array.shape_pointer(shape)
      out = MLX::FFI.mlx_array_new
      MLX.check!(
        MLX::FFI.mlx_broadcast_to(out.pointer, @struct, ptr, shape.size, MLX.stream_struct),
        "mlx_broadcast_to"
      )
      self.class.from_struct(out)
    end

    def expand_dims(axes)
      axes = axes.is_a?(::Array) ? axes.map(&:to_i) : [axes.to_i]
      ptr = Array.shape_pointer(axes)
      out = MLX::FFI.mlx_array_new
      MLX.check!(
        MLX::FFI.mlx_expand_dims_axes(out.pointer, @struct, ptr, axes.size, MLX.stream_struct),
        "mlx_expand_dims_axes"
      )
      self.class.from_struct(out)
    end

    def astype(dtype)
      out = MLX::FFI.mlx_array_new
      MLX.check!(
        MLX::FFI.mlx_astype(out.pointer, @struct, MLX::DType.to_c(dtype), MLX.stream_struct),
        "mlx_astype"
      )
      self.class.from_struct(out)
    end

    def take(indices, axis: nil)
      indices = indices.is_a?(MLX::Array) ? indices : MLX::Array.new(indices, dtype: :int32)
      out = MLX::FFI.mlx_array_new
      if axis.nil?
        MLX.check!(
          MLX::FFI.mlx_take_axis(out.pointer, @struct, indices.struct, 0, MLX.stream_struct),
          "mlx_take_axis"
        )
      else
        MLX.check!(
          MLX::FFI.mlx_take_axis(out.pointer, @struct, indices.struct, axis.to_i, MLX.stream_struct),
          "mlx_take_axis"
        )
      end
      self.class.from_struct(out)
    end

    def argmax(axis: nil, keepdims: false)
      out = MLX::FFI.mlx_array_new
      ax = axis.nil? ? 0 : axis_to_positive(axis)
      MLX.check!(
        MLX::FFI.mlx_argmax_axis(out.pointer, @struct, ax, keepdims, MLX.stream_struct),
        "mlx_argmax_axis"
      )
      self.class.from_struct(out)
    end

    def slice(start, stop, strides = nil)
      start = start.map(&:to_i)
      stop  = stop.map(&:to_i)
      strides ||= ::Array.new(start.size, 1)
      strides = strides.map(&:to_i)
      start_ptr   = Array.shape_pointer(start)
      stop_ptr    = Array.shape_pointer(stop)
      strides_ptr = Array.shape_pointer(strides)
      out = MLX::FFI.mlx_array_new
      MLX.check!(
        MLX::FFI.mlx_slice(out.pointer, @struct,
                           start_ptr, start.size,
                           stop_ptr, stop.size,
                           strides_ptr, strides.size,
                           MLX.stream_struct),
        "mlx_slice"
      )
      self.class.from_struct(out)
    end

    def repeat(repeats, axis:)
      out = MLX::FFI.mlx_array_new
      MLX.check!(
        MLX::FFI.mlx_repeat_axis(out.pointer, @struct, repeats.to_i, axis_to_positive(axis), MLX.stream_struct),
        "mlx_repeat_axis"
      )
      self.class.from_struct(out)
    end

    def squeeze
      out = MLX::FFI.mlx_array_new
      MLX.check!(
        MLX::FFI.mlx_squeeze(out.pointer, @struct, MLX.stream_struct),
        "mlx_squeeze"
      )
      self.class.from_struct(out)
    end

    # Class-level concat — combines a list of MLX::Arrays along `axis`.
    def self.concatenate(arrays, axis: 0)
      raise MLX::TypeError, "concatenate needs at least one array" if arrays.empty?

      vec = MLX::FFI.mlx_vector_array_new
      arrays.each do |a|
        raise MLX::TypeError, "concatenate operands must be MLX::Array" unless a.is_a?(MLX::Array)

        MLX.check!(MLX::FFI.mlx_vector_array_append_value(vec, a.struct), "mlx_vector_array_append_value")
      end
      out = MLX::FFI.mlx_array_new
      MLX.check!(
        MLX::FFI.mlx_concatenate_axis(out.pointer, vec, axis.to_i, MLX.stream_struct),
        "mlx_concatenate_axis"
      )
      MLX::FFI.mlx_vector_array_free(vec)
      from_struct(out)
    end

    # ---- Shape ops ----

    def reshape(new_shape)
      new_shape = Array.normalize_shape(new_shape)
      shape_ptr = Array.shape_pointer(new_shape)
      out = MLX::FFI.mlx_array_new
      rc = MLX::FFI.mlx_reshape(out.pointer, @struct, shape_ptr, new_shape.size, MLX.stream_struct)
      MLX.check!(rc, "mlx_reshape")
      self.class.from_struct(out)
    end

    def transpose(axes = nil)
      out = MLX::FFI.mlx_array_new
      rc =
        if axes.nil?
          MLX::FFI.mlx_transpose(out.pointer, @struct, MLX.stream_struct)
        else
          axes = Array.normalize_shape(axes)
          axes_ptr = Array.shape_pointer(axes)
          MLX::FFI.mlx_transpose_axes(out.pointer, @struct, axes_ptr, axes.size, MLX.stream_struct)
        end
      MLX.check!(rc, "mlx_transpose")
      self.class.from_struct(out)
    end

    # ---- Extraction ----

    def eval!
      MLX.check!(MLX::FFI.mlx_array_eval(@struct), "mlx_array_eval")
      self
    end

    def to_a
      eval!
      size
      return scalar_value if ndim.zero?

      flat = read_flat
      Array.nest(flat, shape)
    end

    def to_flat_a
      eval!
      read_flat
    end

    def inspect
      preview = preview_string
      "#<MLX::Array shape=#{shape.inspect} dtype=#{dtype} #{preview}>"
    end

    # ---- Internal helpers ----

    def self.normalize_shape(shape)
      shape = [shape] if shape.is_a?(Integer)
      shape.map! { |d| Integer(d) }
      shape.each do |d|
        raise MLX::ShapeError, "shape dims must be non-negative, got #{shape.inspect}" if d.negative?
      end
      shape
    end

    def self.shape_pointer(shape)
      ptr = ::FFI::MemoryPointer.new(:int, [shape.size, 1].max)
      ptr.write_array_of_int(shape) unless shape.empty?
      ptr
    end

    def self.infer_shape_and_flatten(obj)
      if obj.is_a?(::Array)
        if obj.empty?
          [[0], []]
        else
          subs = obj.map { |x| infer_shape_and_flatten(x) }
          sub_shapes = subs.map(&:first).uniq
          if sub_shapes.size > 1
            raise MLX::ShapeError, "inconsistent sub-array shapes: #{sub_shapes.inspect}"
          end

          inner_shape = subs.first.first
          flat = subs.flat_map(&:last)
          [[obj.size, *inner_shape], flat]
        end
      else
        [[], [obj]]
      end
    end

    def self.build_from_buffer(flat, shape, dtype)
      shape_ptr = shape_pointer(shape)
      data_ptr =
        case dtype
        when :float32
          ptr = ::FFI::MemoryPointer.new(:float, [flat.size, 1].max)
          ptr.write_array_of_float(flat.map(&:to_f))
          ptr
        when :int32
          ptr = ::FFI::MemoryPointer.new(:int32, [flat.size, 1].max)
          ptr.write_array_of_int32(flat.map(&:to_i))
          ptr
        when :int64
          ptr = ::FFI::MemoryPointer.new(:int64, [flat.size, 1].max)
          ptr.write_array_of_int64(flat.map(&:to_i))
          ptr
        when :bool
          ptr = ::FFI::MemoryPointer.new(:uint8, [flat.size, 1].max)
          ptr.write_array_of_uint8(flat.map { |v| v ? 1 : 0 })
          ptr
        else
          raise MLX::DTypeError, "construction from Ruby array not wired for dtype #{dtype}"
        end
      MLX::FFI.mlx_array_new_data(data_ptr, shape_ptr, shape.size, MLX::DType.to_c(dtype))
    end

    def self.nest(flat, shape)
      return flat[0] if shape.empty?

      idx = 0
      build = lambda do |dims|
        if dims.size == 1
          slice = flat[idx, dims[0]]
          idx += dims[0]
          slice
        else
          ::Array.new(dims[0]) { build.call(dims[1..]) }
        end
      end
      build.call(shape)
    end

    def self.scalar_array(value, dtype)
      shape_ptr = shape_pointer([])
      data_ptr =
        case dtype
        when :float32
          ptr = ::FFI::MemoryPointer.new(:float, 1)
          ptr.write_float(value.to_f)
          ptr
        when :int32
          ptr = ::FFI::MemoryPointer.new(:int32, 1)
          ptr.write_int32(value.to_i)
          ptr
        when :int64
          ptr = ::FFI::MemoryPointer.new(:int64, 1)
          ptr.write_int64(value.to_i)
          ptr
        when :bool
          ptr = ::FFI::MemoryPointer.new(:uint8, 1)
          ptr.write_uint8(value ? 1 : 0)
          ptr
        else
          raise MLX::DTypeError, "scalar construction not wired for dtype #{dtype}"
        end
      struct = MLX::FFI.mlx_array_new_data(data_ptr, shape_ptr, 0, MLX::DType.to_c(dtype))
      inst = allocate
      inst.send(:adopt!, struct)
      inst
    end

    private

    def adopt!(struct)
      @struct = struct
      @autopointer = ::FFI::AutoPointer.new(struct[:ctx], MLX::FFI::ARRAY_RELEASER)
      self
    end

    public

    def coerce(other)
      case other
      when Numeric
        [self.class.scalar_array(other, dtype_for_scalar(other)), self]
      else
        raise ::TypeError, "cannot coerce #{other.class} into MLX::Array"
      end
    end

    private

    def coerce_operand(other)
      case other
      when MLX::Array  then other
      when Numeric, TrueClass, FalseClass
        self.class.scalar_array(other, dtype_for_scalar(other))
      else
        raise MLX::TypeError, "cannot operate on #{other.class}"
      end
    end

    def dtype_for_scalar(value)
      case value
      when TrueClass, FalseClass then :bool
      when Float                 then :float32
      else dtype == :bool ? :int32 : dtype
      end
    end

    def binary_op(c_func, other)
      other = coerce_operand(other)
      out = MLX::FFI.mlx_array_new
      rc = MLX::FFI.send(c_func, out.pointer, @struct, other.struct, MLX.stream_struct)
      MLX.check!(rc, c_func.to_s)
      self.class.from_struct(out)
    end

    def unary_op(c_func)
      out = MLX::FFI.mlx_array_new
      rc = MLX::FFI.send(c_func, out.pointer, @struct, MLX.stream_struct)
      MLX.check!(rc, c_func.to_s)
      self.class.from_struct(out)
    end

    def reduction_op(scalar_func, axes_func, axes, keepdims)
      out = MLX::FFI.mlx_array_new
      if axes.nil?
        MLX.check!(
          MLX::FFI.send(scalar_func, out.pointer, @struct, keepdims, MLX.stream_struct),
          scalar_func.to_s
        )
      else
        axes = axes_array(axes)
        ptr = Array.shape_pointer(axes)
        MLX.check!(
          MLX::FFI.send(axes_func, out.pointer, @struct, ptr, axes.size, keepdims, MLX.stream_struct),
          axes_func.to_s
        )
      end
      self.class.from_struct(out)
    end

    def axes_array(axes)
      return nil if axes.nil?

      axes = axes.is_a?(::Array) ? axes : [axes]
      axes.map { |a| axis_to_positive(a) }
    end

    def axis_to_positive(axis)
      axis = axis.to_i
      axis.negative? ? axis + ndim : axis
    end

    def read_flat
      n = size
      return [] if n.zero?

      # mlx_array_data_* returns a pointer to the array's storage which may
      # carry non-default strides (e.g. transpose returns a view). Force a
      # contiguous, row-major materialization first so the pointer matches
      # what callers expect from #to_a.
      contig = ensure_contiguous

      case dtype
      when :float32
        ptr = MLX::FFI.mlx_array_data_float32(contig.struct)
        raise MLX::Error, "array data unavailable (eval failed?)" if ptr.nil? || ptr.null?

        ptr.read_array_of_float(n)
      when :int32
        ptr = MLX::FFI.mlx_array_data_int32(contig.struct)
        raise MLX::Error, "array data unavailable (eval failed?)" if ptr.nil? || ptr.null?

        ptr.read_array_of_int32(n)
      when :int64
        ptr = MLX::FFI.mlx_array_data_int64(contig.struct)
        raise MLX::Error, "array data unavailable (eval failed?)" if ptr.nil? || ptr.null?

        ptr.read_array_of_int64(n)
      when :bool
        ptr = MLX::FFI.mlx_array_data_bool(contig.struct)
        raise MLX::Error, "array data unavailable (eval failed?)" if ptr.nil? || ptr.null?

        ptr.read_array_of_uint8(n).map { |v| v != 0 }
      else
        raise NotImplementedError, "#to_a not wired for dtype #{dtype}"
      end
    end

    def ensure_contiguous
      out = MLX::FFI.mlx_array_new
      rc = MLX::FFI.mlx_contiguous(out.pointer, @struct, false, MLX.stream_struct)
      MLX.check!(rc, "mlx_contiguous")
      copy = self.class.allocate
      copy.send(:adopt!, out)
      copy.eval!
      copy
    end

    def scalar_value
      case dtype
      when :float32
        out = ::FFI::MemoryPointer.new(:float, 1)
        MLX.check!(MLX::FFI.mlx_array_item_float32(out, @struct), "mlx_array_item_float32")
        out.read_float
      when :int32
        out = ::FFI::MemoryPointer.new(:int32, 1)
        MLX.check!(MLX::FFI.mlx_array_item_int32(out, @struct), "mlx_array_item_int32")
        out.read_int32
      when :int64
        out = ::FFI::MemoryPointer.new(:int64, 1)
        MLX.check!(MLX::FFI.mlx_array_item_int64(out, @struct), "mlx_array_item_int64")
        out.read_int64
      when :bool
        out = ::FFI::MemoryPointer.new(:uint8, 1)
        MLX.check!(MLX::FFI.mlx_array_item_bool(out, @struct), "mlx_array_item_bool")
        out.read_uint8 != 0
      else
        raise NotImplementedError, "scalar extraction not wired for dtype #{dtype}"
      end
    end

    def preview_string
      eval!
      flat = read_flat
      if flat.size <= 8
        flat.map { |v| v.is_a?(Float) ? format("%.4g", v) : v.to_s }.join(", ").then { |s| "[#{s}]" }
      else
        head = flat[0, 4].map { |v| format("%.4g", v) }
        tail = flat[-4..].map { |v| format("%.4g", v) }
        "[#{head.join(", ")}, ..., #{tail.join(", ")}]"
      end
    rescue NotImplementedError
      "(non-float32 preview not implemented)"
    end
  end
end
