# frozen_string_literal: true

# LoRA "fine-tune" demo on a synthetic regression task.
#
# Setup:
#   * Base = frozen 2-layer MLP with random weights.
#   * Target = the same architecture with *different* random weights.
#   * Goal = train a rank-r LoRA adapter on top of the frozen base so
#     the composite output matches the target on a small dataset.
#
# Only the LoRA A and B tensors update; the base weights stay where
# they are. We then persist the adapter as a small safetensors file and
# confirm reloading it reproduces the trained behavior.

require "mlx"
require "tmpdir"

MLX.random_seed(0)

class TwoLayer < MLX::NN::Module
  def initialize
    super()
    @fc1 = MLX::NN::Linear.new(32, 64, bias: false)
    @fc2 = MLX::NN::Linear.new(64, 8, bias: false)
  end

  def forward(x); @fc2.call(MLX::Array.new(0.0).maximum(@fc1.call(x))); end
end

frozen = TwoLayer.new
target = TwoLayer.new

# Snapshot the base weight so we can confirm it doesn't move.
base_fc1_before = frozen.instance_variable_get(:@fc1).instance_variable_get(:@weight).to_a

MLX.attach_lora(frozen, rank: 8, alpha: 16)
optim = MLX::Optimizers::AdamW.new(frozen, lr: 1e-2)

xs = MLX::Array.random_normal([16, 32])
ys = target.call(xs)

# value_and_grad expects positional MLX::Array inputs, one per
# parameter we want gradients for. We zip the path → array hash so
# we can rehydrate `grads_hash` for the optimizer.
param_paths = frozen.named_parameters.keys

loss_fn = lambda do |*params|
  frozen.update(param_paths.zip(params).to_h)
  pred = frozen.call(xs)
  ((pred - ys) * (pred - ys)).mean
end

initial_loss = nil
50.times do |step|
  params = param_paths.map { |p| frozen.named_parameters[p] }
  value, grads = MLX.value_and_grad(loss_fn, argnums: (0...params.size).to_a).call(*params)
  initial_loss ||= value.to_a.to_f
  optim.step(param_paths.zip(grads).to_h)
  puts format("step=%2d loss=%.4f", step, value.to_a.to_f) if (step % 10).zero?
end
final_loss = loss_fn.call(*param_paths.map { |p| frozen.named_parameters[p] }).to_a.to_f
puts format("\nLoss: %.4f -> %.4f  (%.1fx reduction)", initial_loss, final_loss, initial_loss / final_loss)

base_fc1_after = frozen.instance_variable_get(:@fc1).base.instance_variable_get(:@weight).to_a
puts "Base fc1.weight unchanged: #{base_fc1_before == base_fc1_after}"

# Persist + reload + verify the trained composite output.
Dir.mktmpdir("lora-") do |dir|
  path = File.join(dir, "adapter.safetensors")
  MLX::IO.save_adapter(frozen, path)
  puts "Adapter size on disk: #{File.size(path)} bytes"

  fresh = TwoLayer.new
  fresh.update(
    "fc1.weight" => frozen.instance_variable_get(:@fc1).base.instance_variable_get(:@weight),
    "fc2.weight" => frozen.instance_variable_get(:@fc2).base.instance_variable_get(:@weight)
  )
  MLX.attach_lora(fresh, rank: 8, alpha: 16)
  MLX::IO.load_adapter(fresh, path)

  trained_out = frozen.call(xs).to_a
  reloaded_out = fresh.call(xs).to_a
  diff = trained_out.flatten.zip(reloaded_out.flatten).map { |a, b| (a - b).abs }.max
  puts format("Max |trained - reloaded|: %.6e", diff)
end
