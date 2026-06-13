# frozen_string_literal: true

require "json"

module MLX
  module IO
    # Minimal Hugging Face loader. Today it handles only a local directory
    # layout (the most common case for tests and developer workflows). If
    # `repo_or_path` is a directory we read directly from it; if it looks
    # like a `org/name` slug we look under `$HF_HOME/hub/models--org--name/
    # snapshots/<rev>/` (the standard `huggingface_hub` cache layout).
    #
    # We do not download from the Hub here — that's a job for an external
    # CLI (`huggingface-cli download`) or a user script. Phase 3 stays
    # network-free per the scope guard.
    module HuggingFace
      module_function

      # Returns a ready, weight-loaded MLX::NN::Module.
      #
      # If `repo_or_path` is an `org/name` slug and isn't in the local
      # HF cache, fetches it via {MLX::IO::Hub.download} first. Pass
      # `download: false` to require an already-cached/local path.
      def load(repo_or_path, download: true, revision: "main",
               allow_patterns: ["*.json", "*.safetensors", "tokenizer*"])
        dir = resolve_local_dir(repo_or_path)
        if dir.nil?
          raise ArgumentError, "no local model directory for #{repo_or_path.inspect}" unless download

          dir = MLX::IO::Hub.download(repo_or_path,
                                      revision: revision,
                                      allow_patterns: allow_patterns)
        end
        config_path = File.join(dir, "config.json")
        raise ArgumentError, "no config.json in #{dir}" unless File.exist?(config_path)

        config_hash = JSON.parse(File.read(config_path))
        arch = pick_arch(config_hash)
        klass = MLX::Models.lookup(arch)

        # Each Models::* class is expected to expose a Config and the model
        # itself. By convention Llama -> LlamaConfig. We use a module-level
        # lookup to avoid hard-coding the pair.
        config_klass = config_class_for(klass)
        config = config_klass.new(config_hash)
        model = klass.new(config)

        # If the checkpoint is pre-quantized, swap matching Linear layers
        # to QuantizedLinear *before* loading so the safetensors names
        # (weight/scales/biases) land in the right ivars.
        qcfg = quantization_config(config_hash)
        if qcfg
          MLX.quantize_model(model,
                             bits: qcfg[:bits],
                             group_size: qcfg[:group_size],
                             predicate: qcfg[:predicate])
        end

        weights = load_weights(dir)
        # Allow the model class to remap weight names if it wants to —
        # this keeps the architecture file in charge of compatibility with
        # the public HF checkpoint naming.
        weights = klass.remap_weights(weights) if klass.respond_to?(:remap_weights)

        slots = collect_slots(model)
        applied = {}
        missing = []
        slots.each_key do |path|
          tensor = weights[path]
          if tensor.nil?
            missing << path
          else
            applied[path] = tensor
          end
        end
        unexpected = weights.keys - applied.keys

        assign_slots(model, applied)
        model.instance_variable_set(:@_load_report, {
                                      applied: applied.keys,
                                      missing: missing,
                                      unexpected: unexpected
                                    })
        model
      end

      # Resolve `repo_or_path` to a local snapshot directory or return
      # nil if nothing is cached. Auto-download is left to {#load}.
      def resolve_local_dir(repo_or_path)
        return repo_or_path if File.directory?(repo_or_path)
        return nil unless repo_or_path.include?("/")

        MLX::IO::Hub.cached_snapshot(repo_or_path)
      end

      def pick_arch(config_hash)
        archs = config_hash["architectures"]
        return archs.first if archs.is_a?(::Array) && !archs.empty?

        raise ArgumentError, "config.json missing `architectures` key"
      end

      # By naming convention: Llama -> LlamaConfig. Reach into the same
      # namespace as the model class.
      def config_class_for(model_class)
        ns = model_class.name.split("::")[0..-2].inject(Object) { |m, n| m.const_get(n) }
        cfg_name = "#{model_class.name.split("::").last}Config"
        ns.const_get(cfg_name)
      end

      def load_weights(dir)
        index_path = File.join(dir, "model.safetensors.index.json")
        if File.exist?(index_path)
          load_sharded(dir, index_path)
        else
          single = File.join(dir, "model.safetensors")
          unless File.exist?(single)
            raise ArgumentError, "no model.safetensors or model.safetensors.index.json in #{dir}"
          end

          MLX::IO::Safetensors.load(single)
        end
      end

      # Pre-quantized HF checkpoints carry a `quantization` block. mlx's
      # own format uses {"bits":..., "group_size":...}; we honor the same
      # keys, plus an optional `skip_modules` list for layers (typically
      # `lm_head`) the producer chose to leave dense.
      def quantization_config(config_hash)
        q = config_hash["quantization"]
        return nil unless q.is_a?(Hash)

        bits       = Integer(q["bits"] || 4)
        group_size = Integer(q["group_size"] || 64)
        skip       = Array(q["skip_modules"] || [])
        predicate  = skip.empty? ? nil : ->(path, _layer) { !skip.include?(path) }
        { bits: bits, group_size: group_size, predicate: predicate }
      end

      # Like Module#named_parameters but also yields the quantized
      # buffers (weight/scales/biases) for any QuantizedLinear children.
      def collect_slots(mod, prefix = "")
        out = mod.named_parameters(prefix).dup
        each_quantized(mod, prefix) do |path, ql|
          ql.named_buffers(path).each { |k, v| out[k] = v }
        end
        out
      end

      def each_quantized(mod, prefix, &block)
        mod.instance_variables.each do |ivar|
          name = ivar.to_s
          next if name.start_with?("@_")

          value = mod.instance_variable_get(ivar)
          path = prefix.empty? ? name.delete_prefix("@") : "#{prefix}.#{name.delete_prefix("@")}"
          case value
          when MLX::NN::QuantizedLinear
            block.call(path, value)
          when MLX::NN::Module
            each_quantized(value, path, &block)
          when ::Array
            value.each_with_index do |item, i|
              sub = "#{path}.#{i}"
              case item
              when MLX::NN::QuantizedLinear then block.call(sub, item)
              when MLX::NN::Module          then each_quantized(item, sub, &block)
              end
            end
          end
        end
      end

      # Slot assignment that handles both the standard Module#update path
      # and the QuantizedLinear update path (weight/scales/biases live on
      # the leaf layer, not as Module-tree parameters).
      def assign_slots(model, applied)
        regular = {}
        per_ql = Hash.new { |h, k| h[k] = {} }

        ql_paths = []
        each_quantized(model, "") { |path, _| ql_paths << path }

        applied.each do |path, arr|
          ql_root = ql_paths.find { |p| path == "#{p}.weight" || path == "#{p}.scales" || path == "#{p}.biases" || path == "#{p}.bias" }
          if ql_root
            leaf = path.delete_prefix("#{ql_root}.")
            per_ql[ql_root][leaf] = arr
          else
            regular[path] = arr
          end
        end

        model.update(regular) unless regular.empty?
        per_ql.each do |ql_path, leaves|
          ql_layer = resolve_layer(model, ql_path.split("."))
          ql_layer.update(leaves)
        end
      end

      def resolve_layer(node, segments)
        segments.each do |seg|
          if node.is_a?(::Array)
            node = node[Integer(seg)]
          else
            node = node.instance_variable_get("@#{seg}")
          end
          raise ArgumentError, "could not resolve quantized layer at #{segments.inspect}" if node.nil?
        end
        node
      end

      def load_sharded(dir, index_path)
        index = JSON.parse(File.read(index_path))
        wmap = index.fetch("weight_map")
        shards = wmap.values.uniq
        merged = {}
        shards.each do |shard|
          shard_tensors = MLX::IO::Safetensors.load(File.join(dir, shard))
          merged.merge!(shard_tensors)
        end
        merged
      end
    end

    module_function

    def load_huggingface(repo_or_path)
      HuggingFace.load(repo_or_path)
    end
  end
end
