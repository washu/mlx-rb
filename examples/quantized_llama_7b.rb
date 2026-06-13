# frozen_string_literal: true

# Real Llama-7B quantized inference. *Not* run by CI — assumes you have an
# Apple Silicon machine with a downloaded Llama checkpoint and enough
# Metal-addressable unified memory to hold the quantized model.
#
# 1. Download the checkpoint:
#    huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./llama2-7b
#
# 2. (Optional) Pre-quantize and re-save so subsequent runs skip the
#    quantize step. Today this script always quantizes in-process.
#
# 3. Run:
#    bundle exec ruby examples/quantized_llama_7b.rb ./llama2-7b 1 2 3 4
#
# Memory: fp16 7B is ~13 GB. 4-bit / group_size 64 ≈ 3.7 GB of weights
# plus scales/biases. Phase 4 target: <6 GB resident on M1 Ultra.

require "mlx"

path   = ARGV.shift or abort "usage: quantized_llama_7b.rb PATH_OR_REPO [TOKEN_IDS...]"
prompt = ARGV.empty? ? [1, 2, 3, 4] : ARGV.map(&:to_i)

puts "Loading dense weights from #{path}..."
model = MLX::IO.load_huggingface(path)
puts "Loaded #{model.class.name}: #{model.named_parameters.size} parameter tensors"

puts "Quantizing every Linear in the transformer stack (skipping lm_head)..."
MLX.quantize_model(model, bits: 4, group_size: 64) { |p, _| p != "lm_head" }

# Force evaluation so the dense buffers go out of scope before we measure.
MLX.eval(*model.named_parameters.values)
GC.start

puts "Generating #{prompt.size + 32} tokens from prompt #{prompt.inspect}..."
out = model.generate(prompt, max_new_tokens: 32)
puts "Token ids: #{out.inspect}"
puts
puts "Note: this script does no tokenization. Decode the ids with your"
puts "tokenizer of choice (sentencepiece, tokenizers gem, etc.)."
