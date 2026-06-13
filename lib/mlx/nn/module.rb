# frozen_string_literal: true

module MLX
  module NN
    # MLX::NN::Module — base class for neural network modules.
    #
    # Per ADR 0002 we do NOT mirror Python mlx's tree-of-dataclasses pattern.
    # Modules are mutable; parameters and submodules are detected by
    # walking instance variables on demand. Anything that is an MLX::Array
    # is a parameter; anything that is another Module is a child. Arrays
    # of those are flattened (so `@layers = [Linear.new, Linear.new]` works).
    #
    # Subclasses implement `#forward(*inputs)`. `#call(*inputs)` is an
    # alias to match PyTorch convention.
    #
    # `freeze` / `unfreeze` toggle a `frozen?` flag; downstream optimizers
    # (Phase 3) will consult this when deciding which params to update.
    #
    # The ivars `@_frozen` is excluded from the param walk (anything
    # starting with `@_` is reserved for the framework).
    class Module
      def initialize
        @_frozen = false
      end

      # Public API ---------------------------------------------------------

      def forward(*_inputs)
        raise NotImplementedError, "#{self.class} must implement #forward"
      end

      def call(*inputs, **kwargs)
        kwargs.empty? ? forward(*inputs) : forward(*inputs, **kwargs)
      end

      # Depth-first flat array of trainable arrays. Ordering matches
      # `#named_parameters`.
      def parameters
        named_parameters.values
      end

      # Hash of "path" => MLX::Array. Path uses dotted segments with the
      # `@` sigil stripped from ivar names. Arrays of submodules expose
      # numeric indices (e.g. "layers.0.weight").
      def named_parameters(prefix = "")
        out = {}
        each_registered_attr do |name, value|
          path = join(prefix, name)
          case value
          when MLX::Array
            out[path] = value
          when Module
            value.named_parameters(path).each { |k, v| out[k] = v }
          when ::Array
            value.each_with_index do |item, i|
              sub = join(path, i.to_s)
              case item
              when MLX::Array then out[sub] = item
              when Module     then item.named_parameters(sub).each { |k, v| out[k] = v }
              end
            end
          end
        end
        out
      end

      # Direct child modules. Used by freeze/unfreeze recursion.
      def children
        out = []
        each_registered_attr do |_, value|
          case value
          when Module
            out << value
          when ::Array
            value.each { |item| out << item if item.is_a?(Module) }
          end
        end
        out
      end

      def freeze
        @_frozen = true
        children.each(&:freeze)
        self
      end

      def unfreeze
        @_frozen = false
        children.each(&:unfreeze)
        self
      end

      def frozen?
        @_frozen == true
      end

      # Bulk parameter replacement. Used by Phase 3 optimizers: hand in a
      # hash matching `named_parameters` and the corresponding ivars are
      # swapped in-place. Unknown paths raise so silent typos surface.
      def update(named_params)
        named_params.each do |path, new_arr|
          unless new_arr.is_a?(MLX::Array)
            raise MLX::TypeError, "update value must be MLX::Array, got #{new_arr.class}"
          end

          set_by_path(path.split("."), new_arr)
        end
        self
      end

      def inspect
        "#<#{self.class} params=#{named_parameters.size}>"
      end

      private

      # Iterate over user-set instance variables, skipping framework-private
      # names (anything starting with `@_`).
      def each_registered_attr
        instance_variables.each do |ivar|
          name = ivar.to_s
          next if name.start_with?("@_")

          value = instance_variable_get(ivar)
          next unless value.is_a?(MLX::Array) || value.is_a?(Module) || value.is_a?(::Array)

          yield name.delete_prefix("@"), value
        end
      end

      def join(prefix, name)
        prefix.empty? ? name : "#{prefix}.#{name}"
      end

      def set_by_path(segments, value)
        head = segments.shift
        ivar = "@#{head}"
        current = instance_variable_get(ivar)
        if segments.empty?
          unless current.is_a?(MLX::Array)
            raise ArgumentError, "no parameter at #{head} (got #{current.class})"
          end

          instance_variable_set(ivar, value)
          return
        end

        case current
        when Module
          current.send(:set_by_path, segments, value)
        when ::Array
          idx = Integer(segments.shift)
          sub = current[idx]
          if segments.empty? && sub.is_a?(MLX::Array)
            current[idx] = value
          elsif sub.is_a?(Module)
            sub.send(:set_by_path, segments, value)
          else
            raise ArgumentError, "cannot descend into #{head}.#{idx}"
          end
        else
          raise ArgumentError, "cannot descend into #{head} (#{current.class})"
        end
      end
    end
  end
end
