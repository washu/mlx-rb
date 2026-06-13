# frozen_string_literal: true

# Synthetic-tiny quantized Llama demo.
#
# Builds a small Llama, runs forward pass at fp32, quantizes every Linear
# in the transformer stack to 4-bit, and runs a second forward pass to
# confirm the model still generates coherent token ids.
#
# Usage:
#   bundle exec ruby examples/quantized_llama.rb

require "mlx"

MLX.random_seed(0)

config = MLX::Models::LlamaConfig.new(
  "architectures" => ["LlamaForCausalLM"],
  "hidden_size" => 64,
  "intermediate_size" => 128,
  "num_hidden_layers" => 2,
  "num_attention_heads" => 4,
  "num_key_value_heads" => 2,
  "rms_norm_eps" => 1e-5,
  "rope_theta" => 10_000.0,
  "vocab_size" => 256,
  "tie_word_embeddings" => false,
  "max_position_embeddings" => 256
)

model = MLX::Models::Llama.new(config)

dense_params = model.named_parameters.size
prompt = [1, 5, 9, 13]
dense_out = model.generate(prompt, max_new_tokens: 8)
puts "Dense Llama: #{dense_params} param tensors"
puts "Dense generation: #{dense_out.inspect}"

# Quantize everything except lm_head — a common convention to preserve
# next-token logit fidelity. The walker mutates the model in place.
MLX.quantize_model(model, bits: 4, group_size: 64) { |path, _| path != "lm_head" }

qparams = model.named_parameters.size
qlin = 0
model.instance_variable_get(:@layers).each do |layer|
  qlin += [layer.instance_variable_get(:@self_attn).instance_variable_get(:@q_proj),
           layer.instance_variable_get(:@self_attn).instance_variable_get(:@k_proj),
           layer.instance_variable_get(:@self_attn).instance_variable_get(:@v_proj),
           layer.instance_variable_get(:@self_attn).instance_variable_get(:@o_proj),
           layer.instance_variable_get(:@mlp).instance_variable_get(:@gate_proj),
           layer.instance_variable_get(:@mlp).instance_variable_get(:@up_proj),
           layer.instance_variable_get(:@mlp).instance_variable_get(:@down_proj)]
          .count { |l| l.is_a?(MLX::NN::QuantizedLinear) }
end
puts "After 4-bit quantization: #{qparams} fp parameter tensors, #{qlin} QuantizedLinear layers"

qout = model.generate(prompt, max_new_tokens: 8)
puts "Quantized generation: #{qout.inspect}"
puts "Token-id overlap with dense: #{dense_out.zip(qout).count { |a, b| a == b }}/#{dense_out.size}"
