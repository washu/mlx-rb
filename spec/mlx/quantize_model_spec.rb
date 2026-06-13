# frozen_string_literal: true

require "spec_helper"

RSpec.describe "MLX.quantize_model" do
  # Tiny MLP that mirrors the Llama-style nesting (root with two Linear
  # ivars + an array of submodules).
  class MiniMLP < MLX::NN::Module
    def initialize
      super()
      @fc_in  = MLX::NN::Linear.new(64, 128, bias: true)
      @blocks = [
        MLX::NN::Linear.new(128, 128, bias: false),
        MLX::NN::Linear.new(128, 128, bias: false)
      ]
      @head   = MLX::NN::Linear.new(128, 10, bias: false)
    end

    def forward(x)
      h = @fc_in.call(x)
      @blocks.each { |b| h = b.call(h) }
      @head.call(h)
    end
  end

  it "replaces every Linear with a QuantizedLinear by default" do
    model = MiniMLP.new
    MLX.quantize_model(model, bits: 4, group_size: 64)

    expect(model.instance_variable_get(:@fc_in)).to be_a(MLX::NN::QuantizedLinear)
    expect(model.instance_variable_get(:@head)).to be_a(MLX::NN::QuantizedLinear)
    model.instance_variable_get(:@blocks).each do |b|
      expect(b).to be_a(MLX::NN::QuantizedLinear)
    end
  end

  it "honors a predicate that skips the lm_head-like leaf" do
    model = MiniMLP.new
    MLX.quantize_model(model) { |path, _| path != "head" }

    expect(model.instance_variable_get(:@head)).to be_a(MLX::NN::Linear)
    expect(model.instance_variable_get(:@fc_in)).to be_a(MLX::NN::QuantizedLinear)
  end

  it "leaves forward pass approximately equivalent at 8-bit" do
    MLX.random_seed(0)
    model = MiniMLP.new
    x = MLX::Array.random_normal([2, 64])
    y_dense = model.call(x).to_a

    MLX.quantize_model(model, bits: 8, group_size: 64)
    y_quant = model.call(x).to_a

    flat_a = y_dense.flatten
    flat_b = y_quant.flatten
    diff = flat_a.zip(flat_b).map { |a, b| (a - b).abs }.sum
    expect(diff / flat_a.size).to be < 0.2
  end
end
