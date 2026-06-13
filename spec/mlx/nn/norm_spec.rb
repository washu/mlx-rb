# frozen_string_literal: true

require "spec_helper"

RSpec.describe "LayerNorm and RMSNorm" do
  it "LayerNorm normalises the last axis to zero mean / unit var" do
    ln = MLX::NN::LayerNorm.new(4)
    x = MLX::Array.new([[1.0, 2.0, 3.0, 4.0]])
    y = ln.call(x).to_a.first
    mean = y.sum / y.size
    var = y.sum { |v| (v - mean)**2 } / y.size
    expect(mean).to be_within(1e-5).of(0.0)
    expect(var).to be_within(5e-3).of(1.0)
  end

  it "LayerNorm matches the python oracle", :oracle do
    x = [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]
    ruby = MLX::NN::LayerNorm.new(4).call(MLX::Array.new(x)).to_a
    oracle = PythonOracle.run_script(<<~PY, inputs: [x])
      X = mx.array(INPUTS[0], dtype=mx.float32)
      ln = mxnn.LayerNorm(4)
      ln.weight = mx.ones((4,))
      ln.bias = mx.zeros((4,))
      emit(ln(X))
    PY
    expect_close(ruby, oracle, tol: 1e-5)
  end

  it "RMSNorm matches the python oracle", :oracle do
    x = [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]
    ruby = MLX::NN::RMSNorm.new(4).call(MLX::Array.new(x)).to_a
    oracle = PythonOracle.run_script(<<~PY, inputs: [x])
      X = mx.array(INPUTS[0], dtype=mx.float32)
      rms = mxnn.RMSNorm(4)
      rms.weight = mx.ones((4,))
      emit(rms(X))
    PY
    expect_close(ruby, oracle, tol: 1e-5)
  end
end
