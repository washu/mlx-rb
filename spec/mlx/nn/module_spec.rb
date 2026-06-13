# frozen_string_literal: true

require "spec_helper"

RSpec.describe MLX::NN::Module do
  class TinyNet < MLX::NN::Module
    def initialize
      super
      @w = MLX::Array.new([1.0, 2.0, 3.0])
      @b = MLX::Array.new([0.5])
      @noise = "ignored"
    end

    def forward(x); x; end
  end

  class NestedNet < MLX::NN::Module
    def initialize
      super
      @inner = TinyNet.new
      @gain = MLX::Array.new([2.0])
    end

    def forward(x); x; end
  end

  it "auto-detects MLX::Array instance variables as parameters" do
    n = TinyNet.new
    expect(n.named_parameters.keys).to eq(%w[w b])
    expect(n.parameters.map(&:to_a)).to eq([[1.0, 2.0, 3.0], [0.5]])
  end

  it "skips non-array, non-module instance variables" do
    n = TinyNet.new
    expect(n.named_parameters.keys).not_to include("noise")
  end

  it "recurses into child modules with dotted paths" do
    n = NestedNet.new
    expect(n.named_parameters.keys).to eq(%w[inner.w inner.b gain])
  end

  it "raises when #forward isn't overridden" do
    bare = Class.new(MLX::NN::Module).new
    expect { bare.call(MLX::Array.new(1.0)) }.to raise_error(NotImplementedError)
  end

  it "freezes and unfreezes recursively" do
    n = NestedNet.new
    expect(n).not_to be_frozen
    n.freeze
    expect(n).to be_frozen
    expect(n.instance_variable_get(:@inner)).to be_frozen
    n.unfreeze
    expect(n).not_to be_frozen
    expect(n.instance_variable_get(:@inner)).not_to be_frozen
  end

  it "replaces parameters via #update" do
    n = NestedNet.new
    new_w = MLX::Array.new([9.0, 9.0, 9.0])
    n.update("inner.w" => new_w)
    expect(n.named_parameters["inner.w"].to_a).to eq([9.0, 9.0, 9.0])
  end
end
