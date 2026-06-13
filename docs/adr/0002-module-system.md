
---

### `docs/adr/0002-module-system.md`

```markdown
# ADR 0002: PyTorch-style module system, not MLX-style trees

## Status

Accepted, 2026-06-10.

## Context

MLX Python represents models as nested trees of `Module` instances, where
`Module` is dataclass-like and parameters are updated functionally via
`tree_map`. This works in Python because `dataclass` + tree utilities give
clean functional semantics.

Torch.rb (and PyTorch) use a different model: subclass `nn.Module`, define
`forward`, parameters auto-register through `__setattr__` / `instance_variable_set`
hooks. Stateful, mutable, but familiar.

Ruby is not a tree-of-immutables language. Trying to mirror MLX's
functional-tree pattern in Ruby produces code that fights the language.

## Decision

**PyTorch-style modules. Subclass `MLX::NN::Module`, define `#forward`,
parameters register automatically from instance variables.**

```ruby
class TransformerBlock < MLX::NN::Module
  def initialize(dim:, num_heads:)
    super()
    @attn = MLX::NN::MultiHeadAttention.new(dim, num_heads)
    @ln1 = MLX::NN::LayerNorm.new(dim)
    @ff = MLX::NN::Linear.new(dim, dim * 4)
    @ln2 = MLX::NN::LayerNorm.new(dim)
  end

  def forward(x)
    x = x + @attn.call(@ln1.call(x))
    x + @ff.call(@ln2.call(x))
  end
end
