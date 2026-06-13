# frozen_string_literal: true

require "spec_helper"

# Re-use the same TinyMLP defined in sgd_spec when both specs load in the
# same suite, but redefine it defensively if running alone.
class TinyMLP < MLX::NN::Module
  def initialize(in_dim, hidden, out_dim)
    super()
    @fc1 = MLX::NN::Linear.new(in_dim, hidden)
    @fc2 = MLX::NN::Linear.new(hidden, out_dim)
  end

  def forward(x)
    @fc2.call(MLX::NN::F.relu(@fc1.call(x)))
  end
end

RSpec.describe MLX::Optimizers::AdamW do
  it "decreases loss below threshold over 100 steps" do
    MLX.random_seed(7)
    model = TinyMLP.new(4, 8, 1)
    x = MLX::Array.random_normal([32, 4])
    y = MLX::Array.random_normal([32, 1])
    opt = MLX::Optimizers::AdamW.new(model, lr: 0.05, weight_decay: 0.0)

    param_names = model.named_parameters.keys

    loss_fn = lambda do |*tensors|
      model.update(param_names.zip(tensors).to_h)
      pred = model.call(x)
      MLX::NN::F.mse_loss(pred, y)
    end

    initial_loss = nil
    final_loss = nil
    100.times do |i|
      params = model.parameters
      value, grads = MLX.value_and_grad(loss_fn, argnums: (0...params.size).to_a).call(*params)
      initial_loss = value.to_a if i.zero?
      final_loss = value.to_a if i == 99
      opt.step(param_names.zip(grads).to_h)
    end

    expect(final_loss).to be < initial_loss * 0.25
  end
end

RSpec.describe MLX::Optimizers::CosineSchedule do
  it "warms up linearly then cosines to zero" do
    model = MLX::NN::Linear.new(2, 2)
    opt = MLX::Optimizers::SGD.new(model, lr: 1.0)
    sched = described_class.new(opt, total_steps: 10, warmup_steps: 2)

    # step 0: lr starts at 0 (warmup)
    expect(opt.lr).to be_within(1e-6).of(0.0)
    sched.step # step 1 -> 0.5 (mid-warmup)
    expect(opt.lr).to be_within(1e-6).of(0.5)
    sched.step # step 2 -> 1.0 (end of warmup, full base lr)
    expect(opt.lr).to be_within(1e-6).of(1.0)
    8.times { sched.step } # steps 3..10
    expect(opt.lr).to be_within(1e-6).of(0.0)
  end
end

RSpec.describe MLX::Optimizers::LinearWarmup do
  it "linearly warms up then holds" do
    model = MLX::NN::Linear.new(2, 2)
    opt = MLX::Optimizers::SGD.new(model, lr: 0.1)
    sched = described_class.new(opt, warmup_steps: 4)
    expect(opt.lr).to be_within(1e-6).of(0.0)
    sched.step
    expect(opt.lr).to be_within(1e-6).of(0.025)
    3.times { sched.step }
    expect(opt.lr).to be_within(1e-6).of(0.1)
    5.times { sched.step }
    expect(opt.lr).to be_within(1e-6).of(0.1)
  end
end
