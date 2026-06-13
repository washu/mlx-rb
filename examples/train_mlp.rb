# frozen_string_literal: true

# A 4-layer MLP trained for 100 steps with AdamW + cosine LR. Demonstrates
# the Phase 3 optimizer + scheduler loop and confirms loss decreases
# reproducibly across runs.
#
# Run: bundle exec ruby examples/train_mlp.rb

require "mlx"

class MLP4 < MLX::NN::Module
  def initialize(in_dim, hidden, out_dim)
    super()
    @fc1 = MLX::NN::Linear.new(in_dim, hidden)
    @fc2 = MLX::NN::Linear.new(hidden, hidden)
    @fc3 = MLX::NN::Linear.new(hidden, hidden)
    @fc4 = MLX::NN::Linear.new(hidden, out_dim)
  end

  def forward(x)
    h = MLX::NN::F.relu(@fc1.call(x))
    h = MLX::NN::F.relu(@fc2.call(h))
    h = MLX::NN::F.relu(@fc3.call(h))
    @fc4.call(h)
  end
end

MLX.random_seed(0)
model = MLP4.new(8, 32, 1)
x = MLX::Array.random_normal([64, 8])
y = MLX::Array.random_normal([64, 1])

opt   = MLX::Optimizers::AdamW.new(model, lr: 0.05, weight_decay: 0.0)
sched = MLX::Optimizers::CosineSchedule.new(opt, total_steps: 100, warmup_steps: 10)

param_names = model.named_parameters.keys

loss_fn = lambda do |*tensors|
  model.update(param_names.zip(tensors).to_h)
  MLX::NN::F.mse_loss(model.call(x), y)
end

100.times do |step|
  params = model.parameters
  value, grads = MLX.value_and_grad(loss_fn, argnums: (0...params.size).to_a).call(*params)
  opt.step(param_names.zip(grads).to_h)
  sched.step
  if (step % 10).zero? || step == 99
    puts format("step=%3d lr=%.4f loss=%.6f", step, opt.lr, value.to_a)
  end
end
