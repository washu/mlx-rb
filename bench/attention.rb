# frozen_string_literal: true

# Single attention block forward pass timing.
#
# Configuration is hard-coded to a typical mid-range shape (B=1, T=512,
# H=16, D=64 → hidden=1024). Override via BENCH_T / BENCH_H / BENCH_D.

require "mlx"

B = 1
T = Integer(ENV["BENCH_T"] || 512)
H = Integer(ENV["BENCH_H"] || 16)
D = Integer(ENV["BENCH_D"] || 64)
HIDDEN = H * D
ITERS  = Integer(ENV["BENCH_ITERS"] || 20)

config = MLX::Models::LlamaConfig.new(
  "architectures" => ["LlamaForCausalLM"],
  "hidden_size" => HIDDEN,
  "intermediate_size" => HIDDEN * 4,
  "num_hidden_layers" => 1,
  "num_attention_heads" => H,
  "num_key_value_heads" => H,
  "vocab_size" => 32_000,
  "max_position_embeddings" => 2048
)

attn = MLX::Models::LlamaAttention.new(config)
x = MLX::Array.random_normal([B, T, HIDDEN])

3.times { attn.call(x).eval! }

times = []
ITERS.times do
  t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  y = attn.call(x)
  y.eval!
  times << (Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0)
end
best = times.min * 1000.0
printf "attention fwd B=%d T=%d H=%d D=%d  mlx-rb=%6.2f ms/op\n",
       B, T, H, D, best
