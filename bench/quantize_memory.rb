# frozen_string_literal: true

# Synthetic Llama quantization memory benchmark.
#
# Builds a Llama with configurable geometry, materializes weights,
# measures resident set size, quantizes to 4-bit, measures again, then
# runs a tiny generate to confirm the quantized model still produces
# stable token ids.
#
# Defaults are Llama-1B-scale (hidden=2048, layers=22). Override:
#   BENCH_HIDDEN=4096 BENCH_LAYERS=32 BENCH_HEADS=32 ruby bench/quantize_memory.rb
#
# A 7B-scale run is hidden=4096 layers=32 heads=32, intermediate=11008.

require "mlx"

MLX.random_seed(Integer(ENV["BENCH_SEED"] || 0))

HIDDEN  = Integer(ENV["BENCH_HIDDEN"]  || 2048)
LAYERS  = Integer(ENV["BENCH_LAYERS"]  || 22)
HEADS   = Integer(ENV["BENCH_HEADS"]   || 32)
KV      = Integer(ENV["BENCH_KV"]      || 4)
INTER   = Integer(ENV["BENCH_INTER"]   || 5632)
VOCAB   = Integer(ENV["BENCH_VOCAB"]   || 32_000)
PROMPT  = (1..Integer(ENV["BENCH_PROMPT"] || 8)).to_a
NEWTOK  = Integer(ENV["BENCH_NEW"]     || 8)

def rss_mb
  out = `ps -o rss= -p #{Process.pid}`.strip
  out.empty? ? Float::NAN : (out.to_f / 1024.0)
end

# Phys-mem footprint as reported by macOS `vmmap --summary`. This catches
# Metal IOSurface / VM_ALLOCATE backed buffers that don't show up in
# RSS but do count against unified-memory pressure.
def footprint_mb
  raw = `vmmap --summary #{Process.pid} 2>/dev/null`
  m = raw[/Physical footprint:\s+([\d.]+)([GM])/]
  return Float::NAN unless m

  val, unit = raw.match(/Physical footprint:\s+([\d.]+)([GM])/).captures
  unit == "G" ? val.to_f * 1024 : val.to_f
end

def fmt_params(n)
  n >= 1e9 ? format("%.2fB", n / 1e9) : format("%.1fM", n / 1e6)
end

def bytes_per_dtype(sym)
  case sym
  when :float32, :uint32, :int32 then 4
  when :float16, :bfloat16, :int16, :uint16 then 2
  when :int64 then 8
  else 4
  end
end

def tensor_bytes(arr)
  arr.size * bytes_per_dtype(arr.dtype)
end

def model_bytes(model)
  total = 0
  model.named_parameters.each_value { |a| total += tensor_bytes(a) }
  if model.respond_to?(:each)
    # nothing — fall through
  end
  # Walk for QuantizedLinear buffers too.
  walker = lambda do |mod|
    mod.instance_variables.each do |ivar|
      v = mod.instance_variable_get(ivar)
      case v
      when MLX::NN::QuantizedLinear
        v.named_buffers.each_value { |a| total += tensor_bytes(a) }
      when MLX::NN::Module
        walker.call(v)
      when ::Array
        v.each { |item| walker.call(item) if item.is_a?(MLX::NN::Module) }
      end
    end
  end
  walker.call(model)
  total
end

config = MLX::Models::LlamaConfig.new(
  "architectures" => ["LlamaForCausalLM"],
  "hidden_size" => HIDDEN,
  "intermediate_size" => INTER,
  "num_hidden_layers" => LAYERS,
  "num_attention_heads" => HEADS,
  "num_key_value_heads" => KV,
  "rms_norm_eps" => 1e-5,
  "rope_theta" => 10_000.0,
  "vocab_size" => VOCAB,
  "tie_word_embeddings" => false,
  "max_position_embeddings" => 2048
)

puts "device=#{MLX.default_device}  Llama hidden=#{HIDDEN} layers=#{LAYERS} heads=#{HEADS} kv=#{KV} inter=#{INTER}"
baseline = rss_mb
baseline_fp = footprint_mb
puts format("baseline: RSS=%.0f MB  vmmap footprint=%.0f MB", baseline, baseline_fp)

t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
model = MLX::Models::Llama.new(config)
# Force every parameter to be evaluated so the buffers are actually allocated.
MLX.eval(*model.named_parameters.values)
build_s = Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0

nparams = model.named_parameters.values.sum(&:size)
dense_mb = rss_mb
dense_fp = footprint_mb
dense_bytes = model_bytes(model)
puts format("after dense build:    %s params  weight bytes=%.0f MB  RSS=%.0f MB  footprint=%.0f MB  (build=%.1fs)",
            fmt_params(nparams), dense_bytes / 1024.0 / 1024, dense_mb, dense_fp, build_s)

# Greedy generation on the dense model.
t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
dense_tokens = model.generate(PROMPT, max_new_tokens: NEWTOK)
dense_s = Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0
puts format("dense generate %d tok: %.2fs (%.1f tok/s)  ids=%s",
            NEWTOK, dense_s, NEWTOK / dense_s, dense_tokens.inspect)

# Quantize. Walks the whole module tree in place; skip lm_head.
t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
MLX.quantize_model(model, bits: 4, group_size: 64) { |path, _| path != "lm_head" }

# Force every quantized buffer to materialize so mlx's lazy graph
# drops its references to the source dense weights, then GC.
ql_buffers = []
walker = lambda do |m|
  m.instance_variables.each do |ivar|
    v = m.instance_variable_get(ivar)
    case v
    when MLX::NN::QuantizedLinear
      v.named_buffers.each_value { |a| ql_buffers << a }
    when MLX::NN::Module
      walker.call(v)
    when ::Array
      v.each { |item| walker.call(item) if item.is_a?(MLX::NN::Module) }
    end
  end
end
walker.call(model)
MLX.eval(*model.named_parameters.values, *ql_buffers)
3.times { GC.start }
quant_s = Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0
quant_mb = rss_mb
quant_fp = footprint_mb
quant_bytes = model_bytes(model)
puts format("after 4-bit quantize: weight bytes=%.0f MB  RSS=%.0f MB  footprint=%.0f MB  (quant=%.1fs)",
            quant_bytes / 1024.0 / 1024, quant_mb, quant_fp, quant_s)
puts format("compression ratio: %.2fx", dense_bytes.to_f / quant_bytes)

t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
quant_tokens = model.generate(PROMPT, max_new_tokens: NEWTOK)
quant_s = Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0
puts format("quant generate %d tok: %.2fs (%.1f tok/s)  ids=%s",
            NEWTOK, quant_s, NEWTOK / quant_s, quant_tokens.inspect)

overlap = dense_tokens.zip(quant_tokens).count { |a, b| a == b }
puts format("token-id overlap with dense: %d/%d", overlap, NEWTOK)

# Logit fidelity is the right measure on random-init synthetics —
# argmax overlap is brittle when the distribution is uniform-near-zero.
puts "(token drift on random-init synthetic is expected; logit MAE is the fidelity signal)"
