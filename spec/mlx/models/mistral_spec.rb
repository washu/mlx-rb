# frozen_string_literal: true

require "spec_helper"

RSpec.describe "MLX::Models::Mistral" do
  it "is a registry alias for Llama" do
    expect(MLX::Models::Mistral).to be(MLX::Models::Llama)
    expect(MLX::Models::MistralConfig).to be(MLX::Models::LlamaConfig)
  end

  it "is reachable through the architecture registry" do
    expect(MLX::Models.lookup("MistralForCausalLM")).to be(MLX::Models::Llama)
  end

  it "builds and generates with a Llama-shape config (Mistral ≥ v0.2 is dense Llama)" do
    MLX.random_seed(0)
    config = MLX::Models::MistralConfig.new(
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "vocab_size" => 64,
      "max_position_embeddings" => 128
    )
    model = MLX::Models::Mistral.new(config)
    expect(model.named_parameters.keys).to include("layers.0.self_attn.q_proj.weight")

    # Q/K/V remain bias-free (Mistral matches Llama there).
    q_proj = model.instance_variable_get(:@layers)[0]
                  .instance_variable_get(:@self_attn)
                  .instance_variable_get(:@q_proj)
    expect(q_proj.use_bias).to be false

    tokens = model.generate([1, 2, 3], max_new_tokens: 2)
    expect(tokens.size).to eq(2)
  end
end
