# frozen_string_literal: true

require "spec_helper"

RSpec.describe "MLX autograd" do
  describe "MLX.grad" do
    it "differentiates x**2 at x=3 to 6" do
      g = MLX.grad ->(x) { (x * x).sum }
      result = g.call(MLX::Array.new(3.0))
      expect(result.to_a).to be_within(1e-6).of(6.0)
    end

    it "differentiates sum(x * w) wrt w" do
      g = MLX.grad(->(w, x) { (x * w).sum }, argnums: 0)
      w = MLX::Array.new([0.5, 0.5, 0.5])
      x = MLX::Array.new([1.0, 2.0, 3.0])
      grad_w = g.call(w, x)
      expect(grad_w.to_a).to eq([1.0, 2.0, 3.0])
    end

    it "raises when an input is not an MLX::Array" do
      g = MLX.grad ->(x) { (x * x).sum }
      expect { g.call(3.0) }.to raise_error(MLX::TypeError)
    end
  end

  describe "MLX.value_and_grad" do
    it "returns (value, grads) for a single arg" do
      vag = MLX.value_and_grad ->(x) { (x * x).sum }
      v, g = vag.call(MLX::Array.new(3.0))
      expect(v.to_a).to be_within(1e-6).of(9.0)
      expect(g.to_a).to be_within(1e-6).of(6.0)
    end

    it "returns gradients per argnum when multi-arg" do
      vag = MLX.value_and_grad(->(x, w) { (x * w).sum }, argnums: [0, 1])
      v, grads = vag.call(MLX::Array.new([1.0, 2.0, 3.0]), MLX::Array.new([0.5, 0.5, 0.5]))
      expect(v.to_a).to be_within(1e-6).of(3.0)
      expect(grads).to be_an(Array)
      expect(grads.map(&:to_a)).to eq([[0.5, 0.5, 0.5], [1.0, 2.0, 3.0]])
    end
  end

  describe "two-layer MLP forward + backward", :oracle do
    it "matches the python oracle on a tiny MLP gradient" do
      MLX.random_seed(0)
      # Hand-crafted weights so the python oracle and Ruby see identical numbers.
      w1 = MLX::Array.new([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6], [-0.7, 0.8, 0.0], [0.9, -0.1, 0.2]])
      b1 = MLX::Array.new([0.0, 0.1, -0.1])
      w2 = MLX::Array.new([[1.0, -1.0], [0.5, 0.5], [-0.3, 0.7]])
      b2 = MLX::Array.new([0.1, -0.1])
      x = MLX::Array.new([[0.5, -0.3, 0.2, 0.7]])

      fn = lambda do |w1_, b1_, w2_, b2_, x_|
        h = x_.matmul(w1_) + b1_
        h = h.maximum(MLX::Array.new(0.0))
        out = h.matmul(w2_) + b2_
        (out * out).sum
      end

      ruby_val, ruby_grads = MLX.value_and_grad(fn, argnums: [0, 1, 2, 3]).call(w1, b1, w2, b2, x)

      oracle = PythonOracle.run_script(<<~PY, inputs: [w1.to_a, b1.to_a, w2.to_a, b2.to_a, x.to_a])
        w1, b1, w2, b2, x = (mx.array(t, dtype=mx.float32) for t in INPUTS)
        def fn(w1, b1, w2, b2):
            h = x @ w1 + b1
            h = mx.maximum(h, mx.array(0.0))
            o = h @ w2 + b2
            return (o * o).sum()
        v, grads = mx.value_and_grad(fn, argnums=(0, 1, 2, 3))(w1, b1, w2, b2)
        mx.eval(v, *grads)
        emit({"value": v.tolist(), "grads": [g.tolist() for g in grads]})
      PY

      expect(ruby_val.to_a).to be_within(1e-4).of(oracle["value"])
      [w1: 0, b1: 1, w2: 2, b2: 3].first.then {}
      ruby_grads.each_with_index do |rg, i|
        expect_close(rg.to_a, oracle["grads"][i], tol: 1e-4)
      end
    end
  end
end
