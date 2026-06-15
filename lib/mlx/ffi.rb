# frozen_string_literal: true

require "ffi"

module MLX
  # Raw FFI declarations against mlx-c.
  #
  # Only the subset of mlx-c needed by Phase 1 is bound here. Idiomatic Ruby
  # wrapping lives in MLX::Array; this module stays close to the C API.
  #
  # mlx-c represents arrays/streams/devices as structs containing a single
  # void* context pointer. The AArch64 calling convention passes those structs
  # in a register identical to a bare pointer, and Ruby FFI's struct-by-value
  # support handles the marshaling. mlx_array* out-parameters are passed as
  # the storage pointer of an existing FFI::Struct.
  module FFI
    extend ::FFI::Library

    # mlx_dtype enum values, ordered to match upstream/mlx/c/array.h.
    MLX_BOOL      = 0
    MLX_UINT8     = 1
    MLX_UINT16    = 2
    MLX_UINT32    = 3
    MLX_UINT64    = 4
    MLX_INT8      = 5
    MLX_INT16     = 6
    MLX_INT32     = 7
    MLX_INT64     = 8
    MLX_FLOAT16   = 9
    MLX_FLOAT32   = 10
    MLX_FLOAT64   = 11
    MLX_BFLOAT16  = 12
    MLX_COMPLEX64 = 13

    # mlx_device_type enum values, ordered to match upstream/mlx/c/device.h.
    MLX_CPU = 0
    MLX_GPU = 1

    # mlx-c struct types. Each is a single-pointer "handle". Passing them
    # by value compiles to a register move on AArch64, matching the C ABI.
    class MlxArray < ::FFI::Struct
      layout :ctx, :pointer
    end

    class MlxStream < ::FFI::Struct
      layout :ctx, :pointer
    end

    class MlxDevice < ::FFI::Struct
      layout :ctx, :pointer
    end

    class MlxString < ::FFI::Struct
      layout :ctx, :pointer
    end

    # mlx_vector_array, mlx_closure*, mlx_map_string_to_array — all
    # single-pointer "ctx" handles, same as MlxArray.
    class MlxVectorArray < ::FFI::Struct
      layout :ctx, :pointer
    end

    class MlxClosure < ::FFI::Struct
      layout :ctx, :pointer
    end

    class MlxClosureValueAndGrad < ::FFI::Struct
      layout :ctx, :pointer
    end

    # mlx_optional_int / mlx_optional_dtype — two-field POD structs from
    # upstream/mlx/c/optional.h. Passed by value. `has_value=false` plus
    # `value=0` is the std::nullopt sentinel mlx-c expects.
    class MlxOptionalInt < ::FFI::Struct
      layout :value, :int, :has_value, :bool
    end

    class MlxOptionalDtype < ::FFI::Struct
      layout :value, :int, :has_value, :bool
    end

    # Build a populated mlx_optional_int.
    def self.opt_int(value)
      s = MlxOptionalInt.new
      if value.nil?
        s[:value] = 0
        s[:has_value] = false
      else
        s[:value] = Integer(value)
        s[:has_value] = true
      end
      s
    end

    def self.opt_dtype(value)
      s = MlxOptionalDtype.new
      if value.nil?
        s[:value] = 0
        s[:has_value] = false
      else
        s[:value] = Integer(value)
        s[:has_value] = true
      end
      s
    end

    # Locate libmlx_bridge.dylib. Search order:
    #   1. MLX_BRIDGE_LIB env var (developer override).
    #   2. The precompiled binary shipped in the gem
    #      (`ext/mlx_bridge/lib/libmlx_bridge.dylib` for arm64-darwin gems).
    #   3. A fresh cargo-release build under
    #      `ext/mlx_bridge/target/release/` (dev checkouts).
    #   4. `mlx_bridge` so the dynamic loader can find it via
    #      DYLD_LIBRARY_PATH.
    #
    # MLX_C_LIB is also honored as a legacy alias from the v0.3.x days
    # so existing dev environments keep working during the transition.
    def self.candidate_lib_paths
      paths = []
      [ENV["MLX_BRIDGE_LIB"], ENV["MLX_C_LIB"]].each do |env|
        paths << env if env && !env.empty?
      end

      gem_root = File.expand_path("../..", __dir__)
      paths.concat([
        # Precompiled platform gem: dylib lives directly inside ext/mlx_bridge/lib/.
        File.join(gem_root, "ext/mlx_bridge/lib/libmlx_bridge.dylib"),
        # Source-gem install: extconf.rb runs cargo at install time and
        # copies the dylib into the same lib/ path. This is the same
        # search location as above; the duplicate keeps the search
        # robust to future layout changes.
        # Dev checkout: cargo build --release leaves the dylib here.
        File.join(gem_root, "ext/mlx_bridge/target/release/libmlx_bridge.dylib")
      ])
      paths << "mlx_bridge"
      paths
    end

    # ffi_lib treats multiple arguments as "must load all", so we walk the
    # candidate list and stop on the first one that opens. The last error is
    # surfaced if nothing works.
    last_error = nil
    loaded = false
    candidate_lib_paths.each do |path|
      ffi_lib(path)
      loaded = true
      break
    rescue LoadError => e
      last_error = e
    end

    @load_error = (last_error unless loaded)

    class << self
      attr_reader :load_error
    end

    def self.loaded?
      @load_error.nil?
    end

    # The attach_function calls below only run if the library loaded. If
    # loading failed we want require "mlx" to still succeed enough to report
    # a useful error from MLX.platform_supported?.
    if loaded?
      # ---- array.h ----
      attach_function :mlx_array_new,        [],                                   MlxArray.by_value
      attach_function :mlx_array_free,       [MlxArray.by_value],                  :int
      attach_function :mlx_array_new_data,   [:pointer, :pointer, :int, :int],     MlxArray.by_value
      attach_function :mlx_array_ndim,       [MlxArray.by_value],                  :size_t
      attach_function :mlx_array_size,       [MlxArray.by_value],                  :size_t
      attach_function :mlx_array_shape,      [MlxArray.by_value],                  :pointer
      attach_function :mlx_array_dtype,      [MlxArray.by_value],                  :int
      attach_function :mlx_array_eval,       [MlxArray.by_value],                  :int

      attach_function :mlx_array_item_float32, [:pointer, MlxArray.by_value],      :int
      attach_function :mlx_array_data_float32, [MlxArray.by_value],                :pointer

      # ---- ops.h ----
      attach_function :mlx_add,        [:pointer, MlxArray.by_value, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_subtract,   [:pointer, MlxArray.by_value, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_multiply,   [:pointer, MlxArray.by_value, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_divide,     [:pointer, MlxArray.by_value, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_matmul,     [:pointer, MlxArray.by_value, MlxArray.by_value, MlxStream.by_value], :int

      attach_function :mlx_reshape,        [:pointer, MlxArray.by_value, :pointer, :size_t, MlxStream.by_value], :int
      attach_function :mlx_transpose,      [:pointer, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_transpose_axes, [:pointer, MlxArray.by_value, :pointer, :size_t, MlxStream.by_value], :int
      attach_function :mlx_contiguous,     [:pointer, MlxArray.by_value, :bool, MlxStream.by_value], :int

      attach_function :mlx_zeros,  [:pointer, :pointer, :size_t, :int, MlxStream.by_value], :int
      attach_function :mlx_ones,   [:pointer, :pointer, :size_t, :int, MlxStream.by_value], :int
      attach_function :mlx_arange, [:pointer, :double, :double, :double, :int, MlxStream.by_value], :int
      attach_function :mlx_full,   [:pointer, :pointer, :size_t, MlxArray.by_value, :int, MlxStream.by_value], :int

      # ---- device.h / stream.h ----
      attach_function :mlx_device_new_type,       [:int, :int],                      MlxDevice.by_value
      attach_function :mlx_device_free,           [MlxDevice.by_value],              :int
      attach_function :mlx_get_default_device,    [:pointer],                        :int
      attach_function :mlx_set_default_device,    [MlxDevice.by_value],              :int
      attach_function :mlx_device_get_type,       [:pointer, MlxDevice.by_value],    :int
      attach_function :mlx_metal_is_available,    [:pointer],                        :int

      attach_function :mlx_default_cpu_stream_new, [], MlxStream.by_value
      attach_function :mlx_default_gpu_stream_new, [], MlxStream.by_value
      attach_function :mlx_stream_free,            [MlxStream.by_value], :int
      attach_function :mlx_synchronize,            [MlxStream.by_value], :int

      # ---- additional array.h extractors (Phase 2 dtype coverage) ----
      attach_function :mlx_array_item_int32,   [:pointer, MlxArray.by_value], :int
      attach_function :mlx_array_item_int64,   [:pointer, MlxArray.by_value], :int
      attach_function :mlx_array_item_bool,    [:pointer, MlxArray.by_value], :int
      attach_function :mlx_array_data_int32,   [MlxArray.by_value],           :pointer
      attach_function :mlx_array_data_int64,   [MlxArray.by_value],           :pointer
      attach_function :mlx_array_data_bool,    [MlxArray.by_value],           :pointer
      attach_function :mlx_array_data_uint16,  [MlxArray.by_value],           :pointer
      attach_function :mlx_array_data_uint32,  [MlxArray.by_value],           :pointer
      attach_function :mlx_array_data_float16, [MlxArray.by_value],           :pointer
      attach_function :mlx_array_data_bfloat16, [MlxArray.by_value],          :pointer

      # ---- ops.h (Phase 2 additions) ----
      attach_function :mlx_negative,     [:pointer, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_exp,          [:pointer, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_log,          [:pointer, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_sqrt,         [:pointer, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_rsqrt,        [:pointer, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_square,       [:pointer, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_abs,          [:pointer, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_sigmoid,      [:pointer, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_tanh,         [:pointer, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_stop_gradient,[:pointer, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_erf,          [:pointer, MlxArray.by_value, MlxStream.by_value], :int

      attach_function :mlx_power,        [:pointer, MlxArray.by_value, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_maximum,      [:pointer, MlxArray.by_value, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_equal,        [:pointer, MlxArray.by_value, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_where,        [:pointer, MlxArray.by_value, MlxArray.by_value, MlxArray.by_value, MlxStream.by_value], :int

      attach_function :mlx_sum,            [:pointer, MlxArray.by_value, :bool, MlxStream.by_value], :int
      attach_function :mlx_sum_axes,       [:pointer, MlxArray.by_value, :pointer, :size_t, :bool, MlxStream.by_value], :int
      attach_function :mlx_mean,           [:pointer, MlxArray.by_value, :bool, MlxStream.by_value], :int
      attach_function :mlx_mean_axes,      [:pointer, MlxArray.by_value, :pointer, :size_t, :bool, MlxStream.by_value], :int
      attach_function :mlx_var_axes,       [:pointer, MlxArray.by_value, :pointer, :size_t, :bool, :int, MlxStream.by_value], :int
      attach_function :mlx_logsumexp_axes, [:pointer, MlxArray.by_value, :pointer, :size_t, :bool, MlxStream.by_value], :int
      attach_function :mlx_softmax_axes,   [:pointer, MlxArray.by_value, :pointer, :size_t, :bool, MlxStream.by_value], :int

      attach_function :mlx_broadcast_to,   [:pointer, MlxArray.by_value, :pointer, :size_t, MlxStream.by_value], :int
      attach_function :mlx_expand_dims_axes, [:pointer, MlxArray.by_value, :pointer, :size_t, MlxStream.by_value], :int
      attach_function :mlx_astype,         [:pointer, MlxArray.by_value, :int, MlxStream.by_value], :int
      attach_function :mlx_take_axis,      [:pointer, MlxArray.by_value, MlxArray.by_value, :int, MlxStream.by_value], :int

      # ---- ops.h (Phase 3 additions) ----
      attach_function :mlx_sin,             [:pointer, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_cos,             [:pointer, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_concatenate_axis, [:pointer, MlxVectorArray.by_value, :int, MlxStream.by_value], :int
      attach_function :mlx_slice,
                      [:pointer, MlxArray.by_value, :pointer, :size_t, :pointer, :size_t, :pointer, :size_t, MlxStream.by_value], :int
      attach_function :mlx_argmax_axis,     [:pointer, MlxArray.by_value, :int, :bool, MlxStream.by_value], :int
      attach_function :mlx_repeat_axis,     [:pointer, MlxArray.by_value, :int, :int, MlxStream.by_value], :int
      attach_function :mlx_squeeze,         [:pointer, MlxArray.by_value, MlxStream.by_value], :int

      # ---- random.h ----
      attach_function :mlx_random_seed,   [:uint64], :int
      attach_function :mlx_random_key,    [:pointer, :uint64], :int
      attach_function :mlx_random_normal, [:pointer, :pointer, :size_t, :int, :float, :float, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_random_uniform,
                      [:pointer, MlxArray.by_value, MlxArray.by_value, :pointer, :size_t, :int, MlxArray.by_value, MlxStream.by_value], :int
      attach_function :mlx_random_bernoulli, [:pointer, MlxArray.by_value, :pointer, :size_t, MlxArray.by_value, MlxStream.by_value], :int

      # ---- fast.h ----
      attach_function :mlx_fast_layer_norm,
                      [:pointer, MlxArray.by_value, MlxArray.by_value, MlxArray.by_value, :float, MlxStream.by_value], :int
      attach_function :mlx_fast_rms_norm,   [:pointer, MlxArray.by_value, MlxArray.by_value, :float, MlxStream.by_value], :int
      attach_function :mlx_fast_scaled_dot_product_attention,
                      [:pointer, MlxArray.by_value, MlxArray.by_value, MlxArray.by_value,
                       :float, :string, MlxVectorArray.by_value, MlxStream.by_value], :int

      # ---- vector.h (vector_array) ----
      attach_function :mlx_vector_array_new,           [],                                     MlxVectorArray.by_value
      attach_function :mlx_vector_array_free,          [MlxVectorArray.by_value],              :int
      attach_function :mlx_vector_array_size,          [MlxVectorArray.by_value],              :size_t
      attach_function :mlx_vector_array_get,           [:pointer, MlxVectorArray.by_value, :size_t], :int
      attach_function :mlx_vector_array_append_value,  [MlxVectorArray.by_value, MlxArray.by_value], :int
      attach_function :mlx_vector_array_set_value,     [:pointer, MlxArray.by_value],          :int

      # ---- closure.h ----
      # Callback type: int (*)(mlx_vector_array*, mlx_vector_array)
      # The first arg is the output pointer (out-param), the second is the
      # input vector_array passed by value (single ctx pointer in the ABI).
      callback :mlx_closure_func, [:pointer, MlxVectorArray.by_value], :int

      attach_function :mlx_closure_new,            [],                                MlxClosure.by_value
      attach_function :mlx_closure_free,           [MlxClosure.by_value],             :int
      attach_function :mlx_closure_new_func,       [:mlx_closure_func],               MlxClosure.by_value
      attach_function :mlx_closure_apply,          [:pointer, MlxClosure.by_value, MlxVectorArray.by_value], :int

      attach_function :mlx_closure_value_and_grad_new,   [], MlxClosureValueAndGrad.by_value
      attach_function :mlx_closure_value_and_grad_free,  [MlxClosureValueAndGrad.by_value], :int
      attach_function :mlx_closure_value_and_grad_apply, [:pointer, :pointer, MlxClosureValueAndGrad.by_value, MlxVectorArray.by_value],
                      :int

      # ---- transforms.h ----
      attach_function :mlx_value_and_grad, [:pointer, MlxClosure.by_value, :pointer, :size_t], :int

      # ---- ops.h (Phase 4: quantization) ----
      # The mlx-c API used to take mlx_optional_int for group_size/bits
      # and an `mlx_vector_array` out-param. The newer mlx-c flattens
      # all of those to plain ints and three out-pointers.
      attach_function :mlx_quantize,
                      [:pointer, :pointer, :pointer, MlxArray.by_value,
                       :int, :int, MlxStream.by_value], :int

      attach_function :mlx_dequantize,
                      [:pointer, MlxArray.by_value, MlxArray.by_value, MlxArray.by_value,
                       :int, :int, MlxStream.by_value], :int

      attach_function :mlx_quantized_matmul,
                      [:pointer, MlxArray.by_value, MlxArray.by_value,
                       MlxArray.by_value, MlxArray.by_value, :bool,
                       :int, :int, MlxStream.by_value], :int
    end

    # Releaser used by AutoPointer. The wrapped pointer is the mlx_array's
    # ctx field; we reconstitute the single-pointer struct and call free.
    ARRAY_RELEASER = lambda do |ctx_ptr|
      next if ctx_ptr.nil? || ctx_ptr.null?

      struct = MlxArray.new
      struct[:ctx] = ctx_ptr
      mlx_array_free(struct)
    end

    # Wrap a returned MlxArray struct in an AutoPointer-backed handle. We
    # return a fresh MlxArray struct that shares the same ctx; the AutoPointer
    # carries the GC lifetime, and the struct is what we pass back to mlx-c.
    def self.wrap_array(struct)
      ctx = struct[:ctx]
      auto = ::FFI::AutoPointer.new(ctx, ARRAY_RELEASER)
      [struct, auto]
    end

    # Helper: allocate a fresh empty MlxArray to be used as an mlx_array* out
    # parameter, and return [struct, struct.pointer].
    def self.new_out_array
      struct = mlx_array_new
      [struct, struct.pointer]
    end

    # A "null" mlx_array (ctx=NULL). Passed by-value to functions whose
    # signature documents the parameter as "may be null"; mlx-c checks
    # `arr.ctx` and substitutes std::nullopt.
    def self.null_array
      s = MlxArray.new
      s[:ctx] = ::FFI::Pointer::NULL
      s
    end

    # Releaser for mlx_vector_array. Same shape as ARRAY_RELEASER: rebuild
    # the single-pointer handle from the AutoPointer'd ctx and free.
    VECTOR_ARRAY_RELEASER = lambda do |ctx_ptr|
      next if ctx_ptr.nil? || ctx_ptr.null?

      struct = MlxVectorArray.new
      struct[:ctx] = ctx_ptr
      mlx_vector_array_free(struct)
    end

    CLOSURE_VAG_RELEASER = lambda do |ctx_ptr|
      next if ctx_ptr.nil? || ctx_ptr.null?

      struct = MlxClosureValueAndGrad.new
      struct[:ctx] = ctx_ptr
      mlx_closure_value_and_grad_free(struct)
    end

    CLOSURE_RELEASER = lambda do |ctx_ptr|
      next if ctx_ptr.nil? || ctx_ptr.null?

      struct = MlxClosure.new
      struct[:ctx] = ctx_ptr
      mlx_closure_free(struct)
    end
  end
end
