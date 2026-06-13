# frozen_string_literal: true

require "spec_helper"

RSpec.describe MLX::NN::Linear do
  it "produces output of shape [B, out_features]" do
    l = MLX::NN::Linear.new(4, 3)
    y = l.call(MLX::Array.random_normal([2, 4]))
    expect(y.shape).to eq([2, 3])
  end

  it "registers weight and bias" do
    l = MLX::NN::Linear.new(4, 3)
    expect(l.named_parameters.keys).to eq(%w[weight bias])
  end

  it "omits bias when bias: false" do
    l = MLX::NN::Linear.new(4, 3, bias: false)
    expect(l.named_parameters.keys).to eq(%w[weight])
  end

  it "matches the python oracle for forward pass", :oracle do
    w = [[0.1, 0.2, 0.3, 0.4], [-0.4, -0.3, -0.2, -0.1], [0.5, 0.0, -0.5, 1.0]]
    b = [0.1, 0.0, -0.1]
    x = [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]

    l = MLX::NN::Linear.new(4, 3)
    l.update("weight" => MLX::Array.new(w), "bias" => MLX::Array.new(b))
    ruby = l.call(MLX::Array.new(x)).to_a

    oracle = PythonOracle.run_script(<<~PY, inputs: [w, b, x])
      W, B, X = (mx.array(t, dtype=mx.float32) for t in INPUTS)
      l = mxnn.Linear(4, 3)
      l.weight = W
      l.bias = B
      emit(l(X))
    PY

    expect_close(ruby, oracle, tol: 1e-5)
  end
end
