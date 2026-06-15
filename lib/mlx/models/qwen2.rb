# frozen_string_literal: true

module MLX
  module Models
    # Qwen2 (Qwen2 / Qwen2.5 / Qwen3 base series) — close-cousin of
    # Llama. Same RoPE, same SwiGLU MLP, same RMSNorm, same grouped-
    # query attention layout, same `tie_word_embeddings` handling.
    # The only architectural difference that affects weight shapes is
    # **bias on the Q/K/V projections** — Qwen2 has them, Llama
    # doesn't. The output projection stays bias-free.
    #
    # HuggingFace configs for Qwen2 set `"attention_bias": true`
    # (older releases used `"qkv_bias": true`); both are handled by
    # `LlamaConfig`. We force the default to `true` here so that a
    # config emitted without that key still loads correctly.
    class Qwen2Config < LlamaConfig
      def initialize(hash)
        h = hash.transform_keys(&:to_s)
        # If the upstream config didn't specify, default to true (the
        # Qwen2 architectural default). Honor an explicit `false` so
        # an unusual checkpoint can override.
        h["attention_bias"] = true unless h.key?("attention_bias") || h.key?("qkv_bias")
        super(h)
      end
    end

    # Qwen2 model. Reuses the entire Llama transformer stack — the
    # bias flag is the only behavioural delta and it propagates
    # through `LlamaAttention` automatically.
    class Qwen2 < Llama
    end
  end
end

MLX::Models.register("Qwen2ForCausalLM", MLX::Models::Qwen2)
MLX::Models.register("Qwen2_5ForCausalLM", MLX::Models::Qwen2)
MLX::Models.register("Qwen3ForCausalLM", MLX::Models::Qwen2)
