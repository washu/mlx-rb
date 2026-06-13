# frozen_string_literal: true

module MLX
  module IO
    # Adapter checkpoint I/O — saves only the trainable LoRA pairs from
    # a module tree, not the frozen base weights. The result is a
    # standard safetensors file, ~10–50 MB on a 7B base instead of the
    # 13 GB the full checkpoint takes.
    #
    # The on-disk schema mirrors the in-memory parameter paths produced
    # by walking the model:
    #
    #   "{path}.lora.a"  -> (in × rank) tensor
    #   "{path}.lora.b"  -> (rank × out) tensor
    #
    # plus a JSON metadata blob carrying `rank` / `alpha` so loading
    # back doesn't need to re-derive them.
    module Adapter
      module_function

      def save(model, path)
        pairs = collect_lora(model)
        raise ArgumentError, "no LoRA layers found — did you forget MLX.attach_lora?" if pairs.empty?

        tensors  = {}
        metadata = { "format" => "mlx-rb.adapter.v1", "layers" => {} }

        pairs.each do |path_prefix, composite|
          a_path = "#{path_prefix}.lora.a"
          b_path = "#{path_prefix}.lora.b"
          tensors[a_path] = composite.lora.instance_variable_get(:@a)
          tensors[b_path] = composite.lora.instance_variable_get(:@b)
          metadata["layers"][path_prefix] = {
            "rank"  => composite.lora.rank,
            "alpha" => composite.lora.alpha,
            "in"    => composite.lora.in_features,
            "out"   => composite.lora.out_features
          }
        end

        MLX::IO::Safetensors.save(tensors, path, metadata: { "adapter" => JSON.dump(metadata) })
        path
      end

      def load(model, path)
        tensors = MLX::IO::Safetensors.load(path)
        meta_str = MLX::IO::Safetensors.load_metadata(path)["adapter"]
        raise ArgumentError, "no adapter metadata in #{path}" unless meta_str

        meta = JSON.parse(meta_str)
        meta.fetch("layers").each do |path_prefix, spec|
          target = resolve(model, path_prefix.split("."))
          unless target.is_a?(MLX::NN::LoRAQuantizedLinear)
            raise ArgumentError, "adapter expects LoRA at #{path_prefix.inspect} but found #{target.class}"
          end

          a = tensors.fetch("#{path_prefix}.lora.a")
          b = tensors.fetch("#{path_prefix}.lora.b")
          unless [a.shape[0], a.shape[1]] == [spec["in"], spec["rank"]]
            raise ArgumentError, "adapter shape mismatch at #{path_prefix}: " \
                                 "expected (#{spec["in"]}, #{spec["rank"]}), got #{a.shape.inspect}"
          end

          target.lora.instance_variable_set(:@a, a)
          target.lora.instance_variable_set(:@b, b)
        end
        model
      end

      # ---- helpers ----

      def collect_lora(model, prefix = "", out = {})
        model.instance_variables.each do |ivar|
          name = ivar.to_s
          next if name.start_with?("@_")

          value = model.instance_variable_get(ivar)
          path = prefix.empty? ? name.delete_prefix("@") : "#{prefix}.#{name.delete_prefix("@")}"
          case value
          when MLX::NN::LoRAQuantizedLinear
            out[path] = value
          when MLX::NN::Module
            collect_lora(value, path, out)
          when ::Array
            value.each_with_index do |item, i|
              sub = "#{path}.#{i}"
              case item
              when MLX::NN::LoRAQuantizedLinear then out[sub] = item
              when MLX::NN::Module then collect_lora(item, sub, out)
              end
            end
          end
        end
        out
      end

      def resolve(node, segments)
        segments.each do |seg|
          node = node.is_a?(::Array) ? node[Integer(seg)] : node.instance_variable_get("@#{seg}")
          raise ArgumentError, "unresolved path segment #{seg.inspect}" if node.nil?
        end
        node
      end
    end

    module_function

    def save_adapter(model, path)
      Adapter.save(model, path)
    end

    def load_adapter(model, path)
      Adapter.load(model, path)
    end
  end
end
