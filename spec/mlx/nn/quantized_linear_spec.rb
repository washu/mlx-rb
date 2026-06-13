# frozen_string_literal: true

require "spec_helper"

RSpec.describe MLX::NN::QuantizedLinear do
  describe ".from_linear" do
    it "preserves output shape and approximates the dense forward pass" do
      MLX.random_seed(7)
      dense = MLX::NN::Linear.new(128, 32, bias: true)
      ql    = described_class.from_linear(dense, bits: 4, group_size: 64)

      x = MLX::Array.random_normal([4, 128])
      y_dense = dense.forward(x)
      y_q     = ql.forward(x)

      expect(y_q.shape).to eq([4, 32])
      diff = (y_dense - y_q).abs.sum.to_a.to_f
      expect(diff / (4 * 32)).to be < 1.5   # 4-bit weight noise budget on 128-D dot products
    end

    it "copies the bias verbatim" do
      dense = MLX::NN::Linear.new(64, 8, bias: true)
      ql    = described_class.from_linear(dense)

      expect(ql.instance_variable_get(:@bias).to_a).to eq(
        dense.instance_variable_get(:@bias).to_a
      )
    end

    it "is frozen and exposes only the bias in named_parameters" do
      dense = MLX::NN::Linear.new(64, 8, bias: true)
      ql    = described_class.from_linear(dense)

      expect(ql.frozen?).to be true
      expect(ql.named_parameters.keys).to eq(["bias"])
      expect(ql.named_buffers.keys).to match_array(%w[weight scales biases])
    end
  end

  describe "#dequantized_weight" do
    it "returns a tensor whose shape matches the original" do
      dense = MLX::NN::Linear.new(64, 16, bias: false)
      ql = described_class.from_linear(dense)
      expect(ql.dequantized_weight.shape).to eq([16, 64])
    end
  end

  describe "validation" do
    it "rejects unsupported bit widths" do
      expect { described_class.new(64, 8, bits: 5) }.to raise_error(ArgumentError, /bits/)
    end

    it "rejects an in_features not divisible by group_size" do
      expect { described_class.new(33, 8, bits: 4, group_size: 64) }
        .to raise_error(ArgumentError, /divisible/)
    end
  end
end
