# ADR 0001: Eager evaluation by default, lazy via block

## Status

Accepted, 2026-06-10.

## Context

MLX evaluates lazily by default. Operations build a compute graph; `mx.eval`
materializes results. This lets MLX batch and fuse operations across
sequential ops, which is a real performance win on Apple Silicon's unified
memory architecture.

Most Ruby ML developers come from Torch.rb, which inherits PyTorch's eager
semantics: every op executes immediately and returns a concrete result.
Eager is what `puts arr + 1` is expected to do.

We have to pick which model `mlx-rb` exposes by default.

## Decision

**Eager by default. Lazy available via an explicit `MLX.lazy { ... }` block.**

```ruby
# Default behavior — eager
a = MLX::Array.new([1, 2, 3])
b = a + 1                  # evaluates here
puts b.to_a                # [2, 3, 4]

# Opt into laziness for a tight loop
MLX.lazy do
  loss = compute_forward(model, batch)
  grads = MLX.grad(model) { compute_forward(model, batch) }
end                        # MLX.eval(loss, grads) called automatically at block end
