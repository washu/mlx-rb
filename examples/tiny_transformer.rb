# frozen_string_literal: true

# Tiny transformer end-to-end demo: a 6-layer decoder-style block stack,
# defined purely in Ruby with MLX::NN, forwarded on a random batch, with
# gradients taken wrt every parameter.
#
# Run:    bundle exec ruby examples/tiny_transformer.rb
#
# This is the Phase 2 acceptance demo per the brief — it touches Linear,
# LayerNorm, MultiHeadAttention, the functional ops (gelu), and autograd
# (MLX.value_and_grad) all in one pass.

require "mlx"

class FeedForward < MLX::NN::Module
  def initialize(dim, mlp_ratio: 4)
    super()
    @up = MLX::NN::Linear.new(dim, dim * mlp_ratio)
    @down = MLX::NN::Linear.new(dim * mlp_ratio, dim)
  end

  def forward(x)
    @down.call(MLX::NN::F.gelu(@up.call(x)))
  end
end

class TransformerBlock < MLX::NN::Module
  def initialize(dim:, num_heads:)
    super()
    @ln1 = MLX::NN::LayerNorm.new(dim)
    @attn = MLX::NN::MultiHeadAttention.new(dim, num_heads)
    @ln2 = MLX::NN::LayerNorm.new(dim)
    @ff = FeedForward.new(dim)
  end

  def forward(x)
    x = x + @attn.call(@ln1.call(x), mask: :causal)
    x + @ff.call(@ln2.call(x))
  end
end

class TinyTransformer < MLX::NN::Module
  def initialize(dim:, num_heads:, depth:)
    super()
    @blocks = Array.new(depth) { TransformerBlock.new(dim: dim, num_heads: num_heads) }
    @ln_f = MLX::NN::LayerNorm.new(dim)
  end

  def forward(x)
    @blocks.each { |b| x = b.call(x) }
    @ln_f.call(x)
  end
end

MLX.random_seed(0)
dim       = 16
num_heads = 4
seq_len   = 8
batch     = 2
depth     = 6

model = TinyTransformer.new(dim: dim, num_heads: num_heads, depth: depth)
x = MLX::Array.random_normal([batch, seq_len, dim])

puts "Model: depth=#{depth} dim=#{dim} heads=#{num_heads}"
puts "Parameters: #{model.parameters.size}"
puts "Input shape: #{x.shape.inspect}"

y = model.call(x)
puts "Forward output shape: #{y.shape.inspect}"

# Loss = mean(y^2) — silly but produces a scalar with grads wrt every param.
# value_and_grad needs the parameter list as its differentiable inputs, so we
# pass them as positional arguments and reconstruct the model inside the fn
# via #update.
params = model.parameters
names  = model.named_parameters.keys

loss_fn = lambda do |*tensors|
  model.update(names.zip(tensors).to_h)
  out = model.call(x)
  (out * out).mean
end

argnums = (0...params.size).to_a
value, grads = MLX.value_and_grad(loss_fn, argnums: argnums).call(*params)
MLX.eval(value, *grads)

puts "Loss: #{value.to_a}"
puts "Grad count: #{grads.size}"
puts "First grad shape: #{grads.first.shape.inspect}"
puts "Grad finite? #{grads.all? { |g| g.to_flat_a.all? { |v| v.finite? } }}"
