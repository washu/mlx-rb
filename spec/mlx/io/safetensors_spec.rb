# frozen_string_literal: true

require "spec_helper"
require "tmpdir"

RSpec.describe MLX::IO::Safetensors do
  it "round-trips float32 tensors" do
    Dir.mktmpdir do |dir|
      a = MLX::Array.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      b = MLX::Array.new([10.0, 20.0])
      path = File.join(dir, "x.safetensors")

      MLX::IO.save_safetensors({ "a" => a, "b" => b }, path, metadata: { "framework" => "mlx-rb" })

      loaded = MLX::IO.load_safetensors(path)
      expect(loaded.keys).to contain_exactly("a", "b")
      expect(loaded["a"].shape).to eq([2, 3])
      expect(loaded["a"].to_a).to eq([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      expect(loaded["b"].to_a).to eq([10.0, 20.0])

      meta = MLX::IO.load_safetensors_metadata(path)
      expect(meta).to eq({ "framework" => "mlx-rb" })
    end
  end

  it "round-trips int32 and int64 tensors" do
    Dir.mktmpdir do |dir|
      i32 = MLX::Array.new([1, 2, 3], dtype: :int32)
      i64 = MLX::Array.new([[10, 20], [30, 40]], dtype: :int64)
      path = File.join(dir, "ints.safetensors")
      MLX::IO.save_safetensors({ "i32" => i32, "i64" => i64 }, path)
      loaded = MLX::IO.load_safetensors(path)
      expect(loaded["i32"].dtype).to eq(:int32)
      expect(loaded["i32"].to_a).to eq([1, 2, 3])
      expect(loaded["i64"].dtype).to eq(:int64)
      expect(loaded["i64"].to_a).to eq([[10, 20], [30, 40]])
    end
  end

  it "matches the Python safetensors library byte-for-byte", :oracle do
    skip "Python `safetensors` not importable" unless safetensors_available?

    Dir.mktmpdir do |dir|
      ruby_path = File.join(dir, "ruby.safetensors")
      py_path   = File.join(dir, "py.safetensors")

      a = MLX::Array.new([[1.5, -2.25], [3.75, 4.0]])
      MLX::IO.save_safetensors({ "a" => a }, ruby_path)

      # Build the same tensor via numpy + safetensors and compare bytes.
      PythonOracle.run_script(<<~PY, args: [py_path])
        import numpy as np, safetensors.numpy
        a = np.array([[1.5, -2.25], [3.75, 4.0]], dtype=np.float32)
        safetensors.numpy.save_file({"a": a}, ARGS[0])
        emit(True)
      PY

      expect(File.binread(ruby_path)).to eq(File.binread(py_path))

      # Cross-load: open the python file with our reader.
      loaded = MLX::IO.load_safetensors(py_path)
      expect(loaded["a"].to_a).to eq([[1.5, -2.25], [3.75, 4.0]])
    end
  end

  def safetensors_available?
    return @safetensors_available unless @safetensors_available.nil?

    _, status = Open3.capture2e("python3", "-c", "import safetensors.numpy")
    @safetensors_available = status.success?
  end
end
