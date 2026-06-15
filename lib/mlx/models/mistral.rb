# frozen_string_literal: true

module MLX
  module Models
    # Mistral — registry alias for {Llama}.
    #
    # Mistral-7B is architecturally a Llama variant: same RoPE, same
    # SwiGLU MLP, same RMSNorm, same Q/K/V/O linear shapes (bias-free),
    # same grouped-query attention. The original v0.1 release used
    # sliding-window attention with `sliding_window=4096`; v0.2 and
    # later (`Mistral-7B-Instruct-v0.2`, `v0.3`, `Mixtral-{...}-MoE`
    # excluded) dropped SWA in favour of dense attention.
    #
    # Because Mistral ≥ v0.2 is bit-identical to Llama in its
    # attention block, we register the `MistralForCausalLM` arch
    # string to point at our existing {Llama} model. If you load a
    # v0.1 checkpoint the model still runs, but the sliding-window
    # behaviour isn't enforced; for that checkpoint generation will
    # be slightly off the reference. The recommendation is to use
    # ≥ v0.2 (the default for any current download from
    # `mistralai/*`).
    #
    # No new config class — `MistralConfig = LlamaConfig`. The
    # HuggingFace loader naming convention (LlamaConfig pairs with
    # Llama) still resolves correctly because
    # `MLX::IO::HuggingFace.config_class_for` looks up
    # `MistralConfig` in this namespace.
    MistralConfig = LlamaConfig
    Mistral       = Llama
  end
end

MLX::Models.register("MistralForCausalLM", MLX::Models::Mistral)
