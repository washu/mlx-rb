# frozen_string_literal: true

# Real Llama-2-7B load + 4-bit quantization on M1 Ultra.
#
# Loads the safetensors checkpoint via MLX::IO.load_huggingface, measures
# weight memory before and after quantization, and runs a short generate.

require "mlx"

PATH = ARGV[0] || "/tmp/llama2-7b"

def footprint_mb
  raw = `vmmap --summary #{Process.pid} 2>/dev/null`
  m = raw.match(/Physical footprint:\s+([\d.]+)([GM])/)
  return Float::NAN unless m

  val, unit = m.captures
  unit == "G" ? val.to_f * 1024 : val.to_f
end

BYTES_PER = { float32: 4, float16: 2, bfloat16: 2, uint32: 4, int32: 4, int64: 8, uint16: 2, bool: 1 }
BYTES_PER.default = 4

def tensor_bytes(a) = a.size * BYTES_PER[a.dtype]

def model_bytes(model)
  total = 0
  model.named_parameters.each_value { |a| total += tensor_bytes(a) }
  walker = lambda do |mod|
    mod.instance_variables.each do |iv|
      v = mod.instance_variable_get(iv)
      case v
      when MLX::NN::QuantizedLinear
        v.named_buffers.each_value { |a| total += tensor_bytes(a) }
      when MLX::NN::Module then walker.call(v)
      when ::Array then v.each { |item| walker.call(item) if item.is_a?(MLX::NN::Module) }
      end
    end
  end
  walker.call(model)
  total
end

def each_ql(mod, &block)
  mod.instance_variables.each do |iv|
    v = mod.instance_variable_get(iv)
    case v
    when MLX::NN::QuantizedLinear then block.call(v)
    when MLX::NN::Module then each_ql(v, &block)
    when ::Array then v.each { |item| each_ql(item, &block) if item.is_a?(MLX::NN::Module) }
    end
  end
end

puts "device=#{MLX.default_device}"
puts format("baseline footprint=%.0f MB", footprint_mb)

t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
model = MLX::IO.load_huggingface(PATH)
load_s = Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0
MLX.eval(*model.named_parameters.values)
puts format("loaded in %.1fs, footprint=%.0f MB", load_s, footprint_mb)

report = model.instance_variable_get(:@_load_report)
if report
  puts "applied=#{report[:applied].size} missing=#{report[:missing].size} unexpected=#{report[:unexpected].size}"
  puts "missing (first 5): #{report[:missing].first(5).inspect}" if report[:missing].any?
  puts "unexpected (first 5): #{report[:unexpected].first(5).inspect}" if report[:unexpected].any?
end

nparams = model.named_parameters.values.sum(&:size)
dense_bytes = model_bytes(model)
puts format("dense:  %.2fB params  weight bytes=%.0f MB",
            nparams / 1e9, dense_bytes / 1024.0 / 1024)

prompt = [1, 15043, 29892, 590, 1024, 338]
puts "running 4-token dense generation from prompt #{prompt.inspect}..."
t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
dense_ids = model.generate(prompt, max_new_tokens: 4)
puts format("dense generate 4 tok in %.1fs: %s",
            Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0, dense_ids.inspect)

puts "quantizing to 4-bit (skip lm_head)..."
t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
MLX.quantize_model(model, bits: 4, group_size: 64) { |p, _| p != "lm_head" }
buffers = []
each_ql(model) { |ql| ql.named_buffers.each_value { |a| buffers << a } }
MLX.eval(*model.named_parameters.values, *buffers)
3.times { GC.start }
puts format("quantize in %.1fs, footprint=%.0f MB", Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0, footprint_mb)

q_bytes = model_bytes(model)
puts format("quant:  weight bytes=%.0f MB  (compression %.2fx)",
            q_bytes / 1024.0 / 1024, dense_bytes.to_f / q_bytes)

t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
q_ids = model.generate(prompt, max_new_tokens: 4)
puts format("quant generate 4 tok in %.1fs: %s",
            Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0, q_ids.inspect)

overlap = dense_ids.zip(q_ids).count { |a, b| a == b }
puts "token-id overlap with dense: #{overlap}/4"
