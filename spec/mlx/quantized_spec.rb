# frozen_string_literal: true

require "spec_helper"

RSpec.describe MLX::Quantized do
  describe ".quantize" do
    it "returns [qw, scales, biases] with expected shapes for 4-bit" do
      w = MLX::Array.random_normal([16, 64])
      qw, scales, biases = MLX::Quantized.quantize(w, bits: 4, group_size: 64)

      expect(qw.shape).to eq([16, 64 * 4 / 32])      # packed K → 8
      expect(scales.shape).to eq([16, 1])
      expect(biases.shape).to eq([16, 1])
    end

    it "returns [qw, scales, biases] with expected shapes for 8-bit, group=32" do
      w = MLX::Array.random_normal([8, 128])
      qw, scales, biases = MLX::Quantized.quantize(w, bits: 8, group_size: 32)

      expect(qw.shape).to eq([8, 128 * 8 / 32])     # packed → 32
      expect(scales.shape).to eq([8, 4])             # 128/32
      expect(biases.shape).to eq([8, 4])
    end
  end

  describe ".dequantize" do
    it "round-trips a Gaussian weight matrix within tolerance (4-bit)" do
      MLX.random_seed(0)
      w = MLX::Array.random_normal([32, 256])
      qw, scales, biases = MLX::Quantized.quantize(w, bits: 4, group_size: 64)
      w_hat = MLX::Quantized.dequantize(qw, scales, biases, bits: 4, group_size: 64)

      # 4-bit affine on standard normal: per-weight error ≈ scale/2^4 ≈ 0.06,
      # so a generous global RMS bound catches any catastrophic regression
      # while staying robust to seed noise.
      diff_sq = ((w - w_hat) * (w - w_hat)).sum.to_a.to_f
      rms = Math.sqrt(diff_sq / (32 * 256))
      expect(rms).to be < 0.1
    end

    it "round-trips more tightly at 8-bit" do
      MLX.random_seed(1)
      w = MLX::Array.random_normal([32, 128])
      qw, scales, biases = MLX::Quantized.quantize(w, bits: 8, group_size: 64)
      w_hat = MLX::Quantized.dequantize(qw, scales, biases, bits: 8, group_size: 64)

      diff_sq = ((w - w_hat) * (w - w_hat)).sum.to_a.to_f
      rms = Math.sqrt(diff_sq / (32 * 128))
      expect(rms).to be < 0.01
    end
  end

  describe ".quantized_matmul" do
    it "matches dense (dequantize then matmul) within fp16-ish tolerance" do
      MLX.random_seed(2)
      x = MLX::Array.random_normal([4, 256])
      w = MLX::Array.random_normal([64, 256])
      qw, scales, biases = MLX::Quantized.quantize(w, bits: 4, group_size: 64)

      fused  = MLX::Quantized.quantized_matmul(x, qw, scales, biases, bits: 4, group_size: 64)
      manual = x.matmul(MLX::Quantized.dequantize(qw, scales, biases, bits: 4, group_size: 64).transpose)

      diff = (fused - manual).abs.sum.to_a.to_f
      expect(diff / (4 * 64)).to be < 1e-2
    end

    it "is callable through the MLX.* top-level aliases" do
      x = MLX::Array.ones([2, 64])
      w = MLX::Array.ones([8, 64])
      qw, scales, biases = MLX.quantize(w, bits: 4, group_size: 64)
      out = MLX.quantized_matmul(x, qw, scales, biases, bits: 4, group_size: 64)
      expect(out.shape).to eq([2, 8])
    end
  end

  describe "Python mlx oracle", :oracle do
    it "matches mlx.core.quantize → dequantize element-wise" do
      input = MLX::Array.random_normal([16, 64]).to_a
      ruby_qw, ruby_scales, ruby_biases = MLX.quantize(MLX::Array.new(input), bits: 4, group_size: 64)
      ruby_dq = MLX.dequantize(ruby_qw, ruby_scales, ruby_biases, bits: 4, group_size: 64).to_a

      py = PythonOracle.run_script(<<~PY, inputs: [input])
        w = mx.array(INPUTS[0], dtype=mx.float32)
        qw, scales, biases = mx.quantize(w, bits=4, group_size=64)
        emit(mx.dequantize(qw, scales, biases, bits=4, group_size=64))
      PY

      expect_close(ruby_dq, py, tol: 1e-4)
    end

    it "matches mlx.core.quantized_matmul to fp16 tolerance" do
      x_data = MLX::Array.random_normal([2, 128]).to_a
      w_data = MLX::Array.random_normal([16, 128]).to_a
      qw, scales, biases = MLX.quantize(MLX::Array.new(w_data), bits: 4, group_size: 64)
      ruby_out = MLX.quantized_matmul(MLX::Array.new(x_data), qw, scales, biases,
                                      bits: 4, group_size: 64).to_a

      py = PythonOracle.run_script(<<~PY, inputs: [x_data, w_data])
        x = mx.array(INPUTS[0], dtype=mx.float32)
        w = mx.array(INPUTS[1], dtype=mx.float32)
        qw, scales, biases = mx.quantize(w, bits=4, group_size=64)
        emit(mx.quantized_matmul(x, qw, scales, biases, transpose=True, bits=4, group_size=64))
      PY

      expect_close(ruby_out, py, tol: 1e-3)
    end
  end
end
