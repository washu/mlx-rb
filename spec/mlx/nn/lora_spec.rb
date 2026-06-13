# frozen_string_literal: true

require "spec_helper"
require "tmpdir"

RSpec.describe MLX::NN::LoRALinear do
  describe "shape and init" do
    it "produces zero output at step 0 (B=0 init)" do
      l = described_class.new(16, 8, rank: 4)
      x = MLX::Array.random_normal([2, 16])
      out = l.forward(x).to_a
      out.flatten.each { |v| expect(v.abs).to be < 1e-6 }
    end

    it "produces non-zero output once B is perturbed" do
      l = described_class.new(16, 8, rank: 4)
      l.instance_variable_set(:@b, MLX::Array.random_normal([4, 8]))
      out = l.forward(MLX::Array.random_normal([2, 16]))
      expect(out.shape).to eq([2, 8])
    end

    it "rejects rank <= 0" do
      expect { described_class.new(16, 8, rank: 0) }.to raise_error(ArgumentError)
    end
  end
end

RSpec.describe MLX::NN::LoRAQuantizedLinear do
  it "exposes only the LoRA pair as parameters" do
    base = MLX::NN::Linear.new(16, 8, bias: true)
    comp = described_class.new(base, rank: 4)
    expect(comp.named_parameters.keys).to match_array(["lora.a", "lora.b"])
  end

  it "matches the base forward at step 0 (B=0)" do
    MLX.random_seed(0)
    base = MLX::NN::Linear.new(16, 8, bias: false)
    comp = described_class.new(base, rank: 4)

    x = MLX::Array.random_normal([2, 16])
    diff = (base.forward(x) - comp.forward(x)).abs.sum.to_a.to_f
    expect(diff).to be < 1e-4
  end

  it "wraps QuantizedLinear too" do
    base = MLX::NN::QuantizedLinear.from_linear(MLX::NN::Linear.new(64, 16, bias: false))
    comp = described_class.new(base, rank: 4)
    out = comp.forward(MLX::Array.random_normal([2, 64]))
    expect(out.shape).to eq([2, 16])
  end
end

RSpec.describe "MLX.attach_lora" do
  class LoRATestMLP < MLX::NN::Module
    def initialize
      super()
      @fc1 = MLX::NN::Linear.new(16, 32)
      @fc2 = MLX::NN::Linear.new(32, 8)
    end

    def forward(x); @fc2.call(@fc1.call(x)); end
  end

  it "wraps every matching Linear" do
    m = LoRATestMLP.new
    MLX.attach_lora(m, rank: 4)
    expect(m.instance_variable_get(:@fc1)).to be_a(MLX::NN::LoRAQuantizedLinear)
    expect(m.instance_variable_get(:@fc2)).to be_a(MLX::NN::LoRAQuantizedLinear)
  end

  it "honors a predicate that skips fc2" do
    m = LoRATestMLP.new
    MLX.attach_lora(m, rank: 4) { |path, _| path != "fc2" }
    expect(m.instance_variable_get(:@fc1)).to be_a(MLX::NN::LoRAQuantizedLinear)
    expect(m.instance_variable_get(:@fc2)).to be_a(MLX::NN::Linear)
  end

  it "named_parameters exposes only LoRA tensors after wrapping" do
    m = LoRATestMLP.new
    MLX.attach_lora(m, rank: 4)
    expect(m.named_parameters.keys).to match_array(%w[
      fc1.lora.a fc1.lora.b
      fc2.lora.a fc2.lora.b
    ])
  end

  it "is idempotent on already-wrapped layers" do
    m = LoRATestMLP.new
    MLX.attach_lora(m, rank: 4)
    inner = m.instance_variable_get(:@fc1)
    MLX.attach_lora(m, rank: 4) # second pass
    expect(m.instance_variable_get(:@fc1)).to be(inner)
  end
end

RSpec.describe "MLX::IO.save_adapter / load_adapter" do
  class LoRAIOMLP < MLX::NN::Module
    def initialize
      super()
      @fc = MLX::NN::Linear.new(16, 8, bias: false)
    end

    def forward(x); @fc.call(x); end
  end

  it "round-trips the LoRA A/B tensors bit-exactly" do
    m = LoRAIOMLP.new
    MLX.attach_lora(m, rank: 4)
    m.instance_variable_get(:@fc).lora.instance_variable_set(:@b, MLX::Array.random_normal([4, 8]))

    Dir.mktmpdir do |dir|
      path = File.join(dir, "adapter.safetensors")
      MLX::IO.save_adapter(m, path)

      m2 = LoRAIOMLP.new
      MLX.attach_lora(m2, rank: 4)
      MLX::IO.load_adapter(m2, path)

      a1 = m.instance_variable_get(:@fc).lora.instance_variable_get(:@a).to_a
      a2 = m2.instance_variable_get(:@fc).lora.instance_variable_get(:@a).to_a
      b1 = m.instance_variable_get(:@fc).lora.instance_variable_get(:@b).to_a
      b2 = m2.instance_variable_get(:@fc).lora.instance_variable_get(:@b).to_a
      expect(a1).to eq(a2)
      expect(b1).to eq(b2)
    end
  end

  it "produces a small file (only the adapter pair, not the base)" do
    m = LoRAIOMLP.new
    MLX.attach_lora(m, rank: 4)
    Dir.mktmpdir do |dir|
      path = File.join(dir, "adapter.safetensors")
      MLX::IO.save_adapter(m, path)
      # 16*4 + 4*8 = 96 fp32 = 384 bytes of tensor data + headers.
      expect(File.size(path)).to be < 4096
    end
  end

  it "raises when the target model has no LoRA layers attached" do
    m = LoRAIOMLP.new
    Dir.mktmpdir do |dir|
      path = File.join(dir, "adapter.safetensors")
      expect { MLX::IO.save_adapter(m, path) }.to raise_error(ArgumentError, /no LoRA/)
    end
  end
end
