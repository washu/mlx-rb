# frozen_string_literal: true

# Llama inference end-to-end demo.
#
# Usage:
#   bundle exec ruby examples/llama_inference.rb [PATH_OR_REPO] [PROMPT...]
#
# When PATH_OR_REPO is omitted (or set to "synthetic") the example builds a
# tiny randomly-initialized Llama, persists it as an HF-style checkpoint to
# a tmpdir, then loads it back through MLX::IO.load_huggingface and runs
# greedy generation — exercising the full Phase 3 load+infer path without
# requiring a downloaded model.
#
# When PATH_OR_REPO points at a real Llama-3 directory (downloaded via
#   huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir ./llama3
# ), this loads the real weights and generates from them. Tokenization isn't
# included here — pass token ids directly via the script body or paste the
# tokenized prompt as integers on the command line.

require "mlx"
require "json"
require "tmpdir"

path = ARGV.shift || "synthetic"

def synthetic_checkpoint(dir)
  config_hash = {
    "architectures" => ["LlamaForCausalLM"],
    "hidden_size" => 32,
    "intermediate_size" => 64,
    "num_hidden_layers" => 2,
    "num_attention_heads" => 4,
    "num_key_value_heads" => 2,
    "rms_norm_eps" => 1e-5,
    "rope_theta" => 10_000.0,
    "vocab_size" => 128,
    "tie_word_embeddings" => false,
    "max_position_embeddings" => 128
  }
  MLX.random_seed(42)
  config = MLX::Models::LlamaConfig.new(config_hash)
  model = MLX::Models::Llama.new(config)

  File.write(File.join(dir, "config.json"), JSON.dump(config_hash))
  weights = model.named_parameters.transform_keys do |k|
    k == "lm_head.weight" ? k : "model.#{k}"
  end
  MLX::IO.save_safetensors(weights, File.join(dir, "model.safetensors"))
  dir
end

if path == "synthetic"
  dir = Dir.mktmpdir("mlx-rb-llama-")
  synthetic_checkpoint(dir)
  puts "Built synthetic Llama checkpoint in #{dir}"
  path = dir
end

puts "Loading #{path}..."
model = MLX::IO.load_huggingface(path)
puts "Loaded #{model.class.name}: #{model.named_parameters.size} parameter tensors"

prompt = ARGV.empty? ? [1, 5, 9, 13] : ARGV.map(&:to_i)
puts "Prompt token ids: #{prompt.inspect}"
generated = model.generate(prompt, max_new_tokens: 16)
puts "Generated token ids: #{generated.inspect}"
