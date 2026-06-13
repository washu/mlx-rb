# frozen_string_literal: true

require "spec_helper"

RSpec.describe MLX::NN::F do
  it "relu zeroes negatives" do
    out = described_class.relu(MLX::Array.new([-1.0, 0.5, -0.2, 2.0])).to_a
    expect(out).to eq([0.0, 0.5, 0.0, 2.0])
  end

  it "softmax rows sum to 1" do
    out = described_class.softmax(MLX::Array.new([[1.0, 2.0, 3.0]])).to_a.first
    expect(out.sum).to be_within(1e-5).of(1.0)
  end

  it "log_softmax matches log(softmax)", :oracle do
    x = [[1.0, 2.0, 3.0], [0.1, 0.2, 0.7]]
    ruby = described_class.log_softmax(MLX::Array.new(x)).to_a
    oracle = PythonOracle.run_script(<<~PY, inputs: [x])
      X = mx.array(INPUTS[0], dtype=mx.float32)
      emit(X - mx.logsumexp(X, axis=-1, keepdims=True))
    PY
    expect_close(ruby, oracle, tol: 1e-5)
  end

  it "cross_entropy matches oracle", :oracle do
    logits = [[1.0, 2.0, 0.5], [0.0, -1.0, 2.0]]
    targets = [1, 2]
    ruby = described_class.cross_entropy(MLX::Array.new(logits), MLX::Array.new(targets, dtype: :int32)).to_a
    oracle = PythonOracle.run_script(<<~PY, inputs: [logits, targets])
      L = mx.array(INPUTS[0], dtype=mx.float32)
      T = mx.array(INPUTS[1], dtype=mx.int32)
      emit(mxnn.losses.cross_entropy(L, T, reduction="mean"))
    PY
    expect(ruby).to be_within(1e-5).of(oracle)
  end

  it "mse_loss matches oracle", :oracle do
    pred = [1.0, 2.0, 3.0]
    target = [1.5, 1.5, 3.5]
    ruby = described_class.mse_loss(MLX::Array.new(pred), MLX::Array.new(target)).to_a
    oracle = PythonOracle.run_script(<<~PY, inputs: [pred, target])
      P = mx.array(INPUTS[0], dtype=mx.float32)
      T = mx.array(INPUTS[1], dtype=mx.float32)
      emit(mxnn.losses.mse_loss(P, T, reduction="mean"))
    PY
    expect(ruby).to be_within(1e-5).of(oracle)
  end

  it "gelu matches oracle (tanh approx)", :oracle do
    x = [-2.0, -0.5, 0.0, 0.5, 2.0]
    ruby = described_class.gelu(MLX::Array.new(x)).to_a
    oracle = PythonOracle.run_script(<<~PY, inputs: [x])
      X = mx.array(INPUTS[0], dtype=mx.float32)
      emit(mxnn.gelu_approx(X))
    PY
    expect_close(ruby, oracle, tol: 1e-5)
  end
end
