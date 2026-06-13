# frozen_string_literal: true

require "spec_helper"

RSpec.describe MLX::Models::Llama do
  let(:config) do
    MLX::Models::LlamaConfig.new(
      "hidden_size" => 16,
      "intermediate_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "vocab_size" => 32,
      "tie_word_embeddings" => false,
      "max_position_embeddings" => 64
    )
  end

  before { MLX.random_seed(0) }

  it "exposes the expected parameter set" do
    model = described_class.new(config)
    names = model.named_parameters.keys
    expect(names).to include(
      "embed_tokens.weight",
      "layers.0.self_attn.q_proj.weight",
      "layers.0.self_attn.k_proj.weight",
      "layers.0.self_attn.v_proj.weight",
      "layers.0.self_attn.o_proj.weight",
      "layers.0.mlp.gate_proj.weight",
      "layers.0.mlp.up_proj.weight",
      "layers.0.mlp.down_proj.weight",
      "layers.0.input_layernorm.weight",
      "layers.0.post_attention_layernorm.weight",
      "norm.weight",
      "lm_head.weight"
    )
  end

  it "returns logits of shape (B, T, vocab)" do
    model = described_class.new(config)
    toks = MLX::Array.new([[1, 2, 3, 4]], dtype: :int32)
    logits = model.call(toks)
    expect(logits.shape).to eq([1, 4, 32])
  end

  it "produces the same last-token logits whether prefilling all at once or via the KV cache" do
    model = described_class.new(config)
    full = model.call(MLX::Array.new([[1, 2, 3, 4, 5]], dtype: :int32)).to_a[0][4]

    caches = model.make_caches
    model.call(MLX::Array.new([[1, 2, 3, 4]], dtype: :int32), caches: caches)
    cached = model.call(MLX::Array.new([[5]], dtype: :int32), caches: caches).to_a[0][0]

    expect_close(cached, full, tol: 1e-4)
  end

  it "generates deterministic tokens given a seed" do
    model = described_class.new(config)
    out = model.generate([1, 2, 3, 4], max_new_tokens: 5)
    expect(out).to be_an(::Array)
    expect(out.size).to eq(5)
    expect(out).to all(be_a(Integer))
  end
end
