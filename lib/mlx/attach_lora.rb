# frozen_string_literal: true

module MLX
  # Walk a Module tree and wrap every matching `Linear` (or
  # `QuantizedLinear`) in a `LoRAQuantizedLinear` composite. Mirrors the
  # shape of {MLX.quantize_model}.
  #
  # Common idiom — attach to attention Q/V projections only, following
  # the original LoRA paper:
  #
  #   MLX.attach_lora(model, rank: 8) { |path, _| path.end_with?("q_proj", "v_proj") }
  module AttachLoRA
    module_function

    def attach_lora(mod, rank:, alpha: nil, predicate: nil)
      raise MLX::TypeError, "attach_lora expects MLX::NN::Module" unless mod.is_a?(MLX::NN::Module)

      walk(mod, "", rank, alpha, predicate)
      mod
    end

    def walk(mod, prefix, rank, alpha, predicate)
      mod.instance_variables.each do |ivar|
        name = ivar.to_s
        next if name.start_with?("@_")

        value = mod.instance_variable_get(ivar)
        path  = prefix.empty? ? name.delete_prefix("@") : "#{prefix}.#{name.delete_prefix("@")}"

        case value
        when MLX::NN::LoRAQuantizedLinear
          # Already wrapped — skip rather than nesting adapters.
          next
        when MLX::NN::Linear, MLX::NN::QuantizedLinear
          if matches?(predicate, path, value)
            wrapped = MLX::NN::LoRAQuantizedLinear.new(value, rank: rank, alpha: alpha)
            mod.instance_variable_set(ivar, wrapped)
          end
        when MLX::NN::Module
          walk(value, path, rank, alpha, predicate)
        when ::Array
          value.each_with_index do |item, i|
            sub_path = "#{path}.#{i}"
            case item
            when MLX::NN::LoRAQuantizedLinear
              next
            when MLX::NN::Linear, MLX::NN::QuantizedLinear
              if matches?(predicate, sub_path, item)
                value[i] = MLX::NN::LoRAQuantizedLinear.new(item, rank: rank, alpha: alpha)
              end
            when MLX::NN::Module
              walk(item, sub_path, rank, alpha, predicate)
            end
          end
        end
      end
    end

    def matches?(predicate, path, layer)
      return true if predicate.nil?

      predicate.call(path, layer)
    end
  end

  module_function

  def attach_lora(mod, rank:, alpha: nil, predicate: nil, &block)
    pred = predicate || block
    AttachLoRA.attach_lora(mod, rank: rank, alpha: alpha, predicate: pred)
  end
end
