# frozen_string_literal: true

module MLX
  # Walks a Module tree and replaces every {MLX::NN::Linear} with a
  # {MLX::NN::QuantizedLinear} of the same shape, with weights quantized
  # in place.
  #
  # An optional predicate gates the replacement. It receives `(path, layer)`
  # and should return truthy to quantize. By default, every Linear is
  # quantized. The common pattern is to skip the language-model head (lm_head)
  # and/or the embedding to avoid measurable perplexity loss:
  #
  #   MLX.quantize_model(model) { |path, _| path != "lm_head" }
  #
  # Returns the same model object (mutated in place) for convenience.
  module ModelQuantization
    module_function

    def quantize_model(mod, bits: 4, group_size: 64, predicate: nil)
      raise MLX::TypeError, "quantize_model expects MLX::NN::Module" unless mod.is_a?(MLX::NN::Module)

      walk(mod, "", bits, group_size, predicate)
      mod
    end

    def walk(mod, prefix, bits, group_size, predicate)
      mod.instance_variables.each do |ivar|
        name = ivar.to_s
        next if name.start_with?("@_")

        value = mod.instance_variable_get(ivar)
        path_base = prefix.empty? ? name.delete_prefix("@") : "#{prefix}.#{name.delete_prefix("@")}"

        case value
        when MLX::NN::Linear
          if should_quantize?(predicate, path_base, value)
            ql = MLX::NN::QuantizedLinear.from_linear(value, bits: bits, group_size: group_size)
            mod.instance_variable_set(ivar, ql)
          end
        when MLX::NN::Module
          walk(value, path_base, bits, group_size, predicate)
        when ::Array
          value.each_with_index do |item, i|
            sub_path = "#{path_base}.#{i}"
            case item
            when MLX::NN::Linear
              if should_quantize?(predicate, sub_path, item)
                value[i] = MLX::NN::QuantizedLinear.from_linear(item, bits: bits, group_size: group_size)
              end
            when MLX::NN::Module
              walk(item, sub_path, bits, group_size, predicate)
            end
          end
        end
      end
    end

    def should_quantize?(predicate, path, layer)
      return true if predicate.nil?

      predicate.call(path, layer)
    end
  end

  module_function

  def quantize_model(mod, bits: 4, group_size: 64, predicate: nil, &block)
    pred = predicate || block
    ModelQuantization.quantize_model(mod, bits: bits, group_size: group_size, predicate: pred)
  end
end
