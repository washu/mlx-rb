# frozen_string_literal: true

require "spec_helper"

RSpec.describe MLX::Models::Qwen2 do
  let(:config) do
    MLX::Models::Qwen2Config.new(
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
  end

  before { MLX.random_seed(0) }

  it "defaults attention_bias to true (Qwen2 architectural default)" do
    expect(config.attention_bias).to be true
  end

  it "honors an explicit attention_bias=false override" do
    cfg = MLX::Models::Qwen2Config.new(
      "hidden_size" => 32, "intermediate_size" => 64, "num_hidden_layers" => 1,
      "num_attention_heads" => 4, "num_key_value_heads" => 2, "vocab_size" => 64,
      "attention_bias" => false
    )
    expect(cfg.attention_bias).to be false
  end

  it "honors the legacy qkv_bias key" do
    cfg = MLX::Models::Qwen2Config.new(
      "hidden_size" => 32, "intermediate_size" => 64, "num_hidden_layers" => 1,
      "num_attention_heads" => 4, "num_key_value_heads" => 2, "vocab_size" => 64,
      "qkv_bias" => true
    )
    expect(cfg.attention_bias).to be true
  end

  it "wires Q/K/V projections with biases" do
    model = described_class.new(config)
    attn  = model.instance_variable_get(:@layers)[0]
                 .instance_variable_get(:@self_attn)
    expect(attn.instance_variable_get(:@q_proj).use_bias).to be true
    expect(attn.instance_variable_get(:@k_proj).use_bias).to be true
    expect(attn.instance_variable_get(:@v_proj).use_bias).to be true
    expect(attn.instance_variable_get(:@o_proj).use_bias).to be false
  end

  it "includes the q/k/v bias tensors in named_parameters" do
    model = described_class.new(config)
    keys  = model.named_parameters.keys
    expect(keys).to include(
      "layers.0.self_attn.q_proj.bias",
      "layers.0.self_attn.k_proj.bias",
      "layers.0.self_attn.v_proj.bias"
    )
    # but no o_proj bias
    expect(keys).not_to include("layers.0.self_attn.o_proj.bias")
  end

  it "generates token ids without error" do
    model = described_class.new(config)
    out = model.generate([1, 2, 3], max_new_tokens: 2)
    expect(out.size).to eq(2)
  end

  describe "architecture registry" do
    it "registers Qwen2 / Qwen2.5 / Qwen3 aliases all pointing at Qwen2" do
      %w[Qwen2ForCausalLM Qwen2_5ForCausalLM Qwen3ForCausalLM].each do |arch|
        expect(MLX::Models.lookup(arch)).to be(MLX::Models::Qwen2)
      end
    end
  end
end
