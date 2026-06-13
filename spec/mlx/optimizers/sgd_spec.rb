# frozen_string_literal: true

require "spec_helper"

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

RSpec.describe MLX::Optimizers::SGD do
  it "decreases loss over 100 steps on a synthetic regression task" do
    MLX.random_seed(123)
    model = TinyMLP.new(4, 8, 1)
    x = MLX::Array.random_normal([32, 4])
    y = MLX::Array.random_normal([32, 1])
    opt = MLX::Optimizers::SGD.new(model, lr: 0.05, momentum: 0.9)

    param_names = model.named_parameters.keys

    loss_fn = lambda do |*tensors|
      model.update(param_names.zip(tensors).to_h)
      pred = model.call(x)
      MLX::NN::F.mse_loss(pred, y)
    end

    losses = []
    100.times do
      params = model.parameters
      value, grads = MLX.value_and_grad(loss_fn, argnums: (0...params.size).to_a).call(*params)
      losses << value.to_a
      opt.step(param_names.zip(grads).to_h)
    end

    expect(losses.first).to be > losses.last
    expect(losses.last).to be < losses.first * 0.5
  end
end
