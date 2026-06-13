# frozen_string_literal: true

require "spec_helper"

RSpec.describe MLX::NN::MultiHeadAttention do
  it "preserves (B, T, D) shape" do
    mha = MLX::NN::MultiHeadAttention.new(8, 2)
    x = MLX::Array.random_normal([2, 4, 8])
    expect(mha.call(x).shape).to eq([2, 4, 8])
  end

  it "supports a causal mask" do
    mha = MLX::NN::MultiHeadAttention.new(8, 2)
    x = MLX::Array.random_normal([1, 4, 8])
    expect(mha.call(x, mask: :causal).shape).to eq([1, 4, 8])
  end

  it "matches the python oracle on a hand-set weight configuration", :oracle do
    dim = 8
    heads = 2
    # Deterministic weights for reproducibility — small enough to type out.
    wq = Array.new(dim) { |i| Array.new(dim) { |j| ((i + j).even? ? 0.1 : -0.1) } }
    wk = Array.new(dim) { |i| Array.new(dim) { |j| ((i * j) % 3 == 0 ? 0.2 : 0.0) } }
    wv = Array.new(dim) { |i| Array.new(dim) { |j| ((i - j).abs / 10.0) } }
    wo = Array.new(dim) { |i| Array.new(dim) { |j| (i == j ? 1.0 : 0.0) } }
    x  = Array.new(2) { Array.new(3) { Array.new(dim) { rand(-1.0..1.0).round(4) } } }

    mha = MLX::NN::MultiHeadAttention.new(dim, heads, bias: false)
    mha.update(
      "q_proj.weight" => MLX::Array.new(wq),
      "k_proj.weight" => MLX::Array.new(wk),
      "v_proj.weight" => MLX::Array.new(wv),
      "out_proj.weight" => MLX::Array.new(wo)
    )
    ruby = mha.call(MLX::Array.new(x)).to_a

    oracle = PythonOracle.run_script(<<~PY, inputs: [wq, wk, wv, wo, x])
      wq, wk, wv, wo, X = (mx.array(t, dtype=mx.float32) for t in INPUTS)
      mha = mxnn.MultiHeadAttention(8, 2, bias=False)
      mha.query_proj.weight = wq
      mha.key_proj.weight = wk
      mha.value_proj.weight = wv
      mha.out_proj.weight = wo
      emit(mha(X, X, X))
    PY

    expect_close(ruby, oracle, tol: 5e-4)
  end
end
