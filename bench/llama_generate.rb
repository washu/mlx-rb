# frozen_string_literal: true

# End-to-end Llama generation throughput.
#
# Default config is a ~1B-scale Llama (hidden=2048, layers=22, heads=32).
# Real Llama-1B has slightly different geometry; the goal here is a
# wall-clock comparison at a representative shape, not bit-equivalence.

require "mlx"

PROMPT_LEN = Integer(ENV["BENCH_PROMPT"] || 16)
NEW_TOKENS = Integer(ENV["BENCH_NEW"] || 64)

config = MLX::Models::LlamaConfig.new(
  "architectures" => ["LlamaForCausalLM"],
  "hidden_size" => 2048,
  "intermediate_size" => 5632,
  "num_hidden_layers" => 22,
  "num_attention_heads" => 32,
  "num_key_value_heads" => 4,
  "rms_norm_eps" => 1e-5,
  "rope_theta" => 10_000.0,
  "vocab_size" => 32_000,
  "tie_word_embeddings" => false,
  "max_position_embeddings" => 2048
)

model = MLX::Models::Llama.new(config)
prompt = (0...PROMPT_LEN).to_a

# Warm-up: one short generation.
model.generate(prompt, max_new_tokens: 4)

t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
model.generate(prompt, max_new_tokens: NEW_TOKENS)
elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0

tok_per_s = NEW_TOKENS / elapsed
printf "Llama-1B-shape  prompt=%d new=%d  elapsed=%.2fs  throughput=%.1f tok/s\n",
       PROMPT_LEN, NEW_TOKENS, elapsed, tok_per_s
