# frozen_string_literal: true

module MLX
  # Registry of built-in model architectures. Phase 3 ships Llama as the
  # reference; other architectures (Mistral, Qwen, Gemma, ...) belong in
  # third-party gems that add entries here.
  module Models
    REGISTRY = {} # rubocop:disable Style/MutableConstant — registry mutated by ::register

    module_function

    def register(arch_name, klass)
      REGISTRY[arch_name.to_s] = klass
    end

    def lookup(arch_name)
      REGISTRY[arch_name.to_s] or
        raise ArgumentError, "no model registered for #{arch_name.inspect} (available: #{REGISTRY.keys.inspect})"
    end
  end
end

require_relative "models/llama"
require_relative "models/mistral"
require_relative "models/qwen2"

MLX::Models.register("LlamaForCausalLM", MLX::Models::Llama)
