# frozen_string_literal: true

require "spec_helper"

RSpec.describe MLX::NN::Dropout do
  it "is a no-op when training: false" do
    d = MLX::NN::Dropout.new(0.5)
    x = MLX::Array.new([1.0, 2.0, 3.0, 4.0])
    expect(d.call(x, training: false).to_a).to eq([1.0, 2.0, 3.0, 4.0])
  end

  it "preserves shape during training" do
    MLX.random_seed(123)
    d = MLX::NN::Dropout.new(0.5)
    x = MLX::Array.random_normal([4, 8])
    out = d.call(x, training: true)
    expect(out.shape).to eq([4, 8])
  end

  it "rejects p outside [0, 1)" do
    expect { MLX::NN::Dropout.new(1.0) }.to raise_error(ArgumentError)
  end
end
