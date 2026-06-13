# frozen_string_literal: true

RSpec.describe MLX::Array do
  describe "construction" do
    it "round-trips a 2-D Ruby array" do
      a = MLX::Array.new([[1.0, 2.0], [3.0, 4.0]])
      expect(a.shape).to eq([2, 2])
      expect(a.dtype).to eq(:float32)
      expect(a.size).to eq(4)
      expect(a.ndim).to eq(2)
      expect(a.to_a).to eq([[1.0, 2.0], [3.0, 4.0]])
    end

    it "round-trips a 1-D Ruby array" do
      a = MLX::Array.new([1.0, 2.0, 3.0])
      expect(a.shape).to eq([3])
      expect(a.to_a).to eq([1.0, 2.0, 3.0])
    end

    it "rejects ragged input", :unit do
      expect { MLX::Array.new([[1, 2], [3]]) }.to raise_error(MLX::ShapeError)
    end
  end

  describe "zeros" do
    it "round-trips" do
      a = MLX::Array.zeros([2, 3])
      expect(a.to_a).to eq([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    end

    it "matches the python oracle", :oracle do
      ruby = MLX::Array.zeros([2, 3]).to_a
      py   = PythonOracle.run(:zeros, args: [[2, 3]])
      expect_close(ruby, py)
    end
  end

  describe "ones" do
    it "round-trips" do
      a = MLX::Array.ones([2, 2])
      expect(a.to_a).to eq([[1.0, 1.0], [1.0, 1.0]])
    end

    it "matches the python oracle", :oracle do
      ruby = MLX::Array.ones([3, 2]).to_a
      py   = PythonOracle.run(:ones, args: [[3, 2]])
      expect_close(ruby, py)
    end
  end

  describe "arange" do
    it "round-trips" do
      a = MLX::Array.arange(0, 5, 1)
      expect(a.to_a).to eq([0.0, 1.0, 2.0, 3.0, 4.0])
    end

    it "matches the python oracle", :oracle do
      ruby = MLX::Array.arange(0.0, 1.0, 0.25).to_a
      py   = PythonOracle.run(:arange, args: [0.0, 1.0, 0.25])
      expect_close(ruby, py)
    end
  end

  describe "full" do
    it "round-trips" do
      a = MLX::Array.full([2, 2], 7.5)
      expect(a.to_a).to eq([[7.5, 7.5], [7.5, 7.5]])
    end

    it "matches the python oracle", :oracle do
      ruby = MLX::Array.full([3, 2], 2.5).to_a
      py   = PythonOracle.run(:full, args: [[3, 2], 2.5])
      expect_close(ruby, py)
    end
  end

  describe "add (#+)" do
    it "round-trips with element-wise add" do
      a = MLX::Array.new([1.0, 2.0, 3.0])
      b = MLX::Array.new([10.0, 20.0, 30.0])
      expect((a + b).to_a).to eq([11.0, 22.0, 33.0])
    end

    it "broadcasts a scalar" do
      a = MLX::Array.new([1.0, 2.0, 3.0])
      expect((a + 1).to_a).to eq([2.0, 3.0, 4.0])
    end

    it "matches the python oracle", :oracle do
      ax = [[1.0, 2.0], [3.0, 4.0]]
      bx = [[5.0, 6.0], [7.0, 8.0]]
      ruby = (MLX::Array.new(ax) + MLX::Array.new(bx)).to_a
      py   = PythonOracle.run(:add, ax, bx)
      expect_close(ruby, py)
    end
  end

  describe "subtract (#-)" do
    it "round-trips" do
      a = MLX::Array.new([10.0, 20.0])
      b = MLX::Array.new([1.0, 2.0])
      expect((a - b).to_a).to eq([9.0, 18.0])
    end

    it "matches the python oracle", :oracle do
      ax = [[10.0, 20.0], [30.0, 40.0]]
      bx = [[1.0, 2.0], [3.0, 4.0]]
      ruby = (MLX::Array.new(ax) - MLX::Array.new(bx)).to_a
      py   = PythonOracle.run(:subtract, ax, bx)
      expect_close(ruby, py)
    end
  end

  describe "multiply (#*)" do
    it "round-trips" do
      a = MLX::Array.new([1.0, 2.0, 3.0])
      b = MLX::Array.new([2.0, 2.0, 2.0])
      expect((a * b).to_a).to eq([2.0, 4.0, 6.0])
    end

    it "matches the python oracle", :oracle do
      ax = [[1.0, 2.0], [3.0, 4.0]]
      bx = [[2.0, 2.0], [3.0, 3.0]]
      ruby = (MLX::Array.new(ax) * MLX::Array.new(bx)).to_a
      py   = PythonOracle.run(:multiply, ax, bx)
      expect_close(ruby, py)
    end
  end

  describe "divide (#/)" do
    it "round-trips" do
      a = MLX::Array.new([10.0, 20.0, 30.0])
      b = MLX::Array.new([2.0, 4.0, 5.0])
      expect((a / b).to_a).to eq([5.0, 5.0, 6.0])
    end

    it "matches the python oracle", :oracle do
      ax = [[8.0, 4.0], [12.0, 16.0]]
      bx = [[2.0, 2.0], [3.0, 4.0]]
      ruby = (MLX::Array.new(ax) / MLX::Array.new(bx)).to_a
      py   = PythonOracle.run(:divide, ax, bx)
      expect_close(ruby, py)
    end
  end

  describe "matmul" do
    it "round-trips an identity matmul" do
      a = MLX::Array.new([[1.0, 2.0], [3.0, 4.0]])
      eye = MLX::Array.new([[1.0, 0.0], [0.0, 1.0]])
      expect(a.matmul(eye).to_a).to eq([[1.0, 2.0], [3.0, 4.0]])
    end

    it "matches the python oracle", :oracle do
      ax = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
      bx = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
      ruby = MLX::Array.new(ax).matmul(MLX::Array.new(bx)).to_a
      py   = PythonOracle.run(:matmul, ax, bx)
      expect_close(ruby, py)
    end
  end

  describe "reshape" do
    it "round-trips" do
      a = MLX::Array.arange(0, 6, 1).reshape([2, 3])
      expect(a.shape).to eq([2, 3])
      expect(a.to_a).to eq([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    end

    it "matches the python oracle", :oracle do
      input = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
      ruby  = MLX::Array.new(input).reshape([2, 3]).to_a
      py    = PythonOracle.run(:reshape, input, args: [[2, 3]])
      expect_close(ruby, py)
    end
  end

  describe "transpose" do
    it "round-trips a default transpose (axis reverse)" do
      a = MLX::Array.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      expect(a.transpose.to_a).to eq([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    end

    it "round-trips an explicit-axis transpose" do
      a = MLX::Array.new([[1.0, 2.0], [3.0, 4.0]])
      expect(a.transpose([1, 0]).to_a).to eq([[1.0, 3.0], [2.0, 4.0]])
    end

    it "matches the python oracle (default)", :oracle do
      input = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
      ruby  = MLX::Array.new(input).transpose.to_a
      py    = PythonOracle.run(:transpose, input)
      expect_close(ruby, py)
    end

    it "matches the python oracle (explicit axes)", :oracle do
      input = [[1.0, 2.0], [3.0, 4.0]]
      ruby  = MLX::Array.new(input).transpose([1, 0]).to_a
      py    = PythonOracle.run(:transpose, input, args: [[1, 0]])
      expect_close(ruby, py)
    end
  end

  describe "MLX.lazy" do
    it "defers eval until block exit and still returns correct values" do
      result = MLX.lazy do
        a = MLX::Array.new([1.0, 2.0])
        b = MLX::Array.new([3.0, 4.0])
        a + b
      end
      expect(result.to_a).to eq([4.0, 6.0])
    end
  end

  describe "inspect" do
    it "includes shape and dtype" do
      a = MLX::Array.new([1.0, 2.0, 3.0])
      expect(a.inspect).to match(/shape=\[3\]/)
      expect(a.inspect).to include("dtype=float32")
    end
  end
end
