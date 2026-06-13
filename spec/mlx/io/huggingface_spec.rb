# frozen_string_literal: true

require "spec_helper"
require "tmpdir"
require "json"

RSpec.describe MLX::IO, ".load_huggingface" do
  # A tiny synthetic Llama: build it, save it in HF layout, reload it, and
  # check that the reloaded model generates identical tokens to the original.
  # No external network access — fixture is generated per-run.
  let(:config_hash) do
    {
      "architectures" => ["LlamaForCausalLM"],
      "hidden_size" => 16,
      "intermediate_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "vocab_size" => 32,
      "tie_word_embeddings" => false,
      "max_position_embeddings" => 64
    }
  end

  let(:prompt) { [3, 7, 1, 14] }

  it "loads a Llama checkpoint from disk and reproduces the original's output" do
    Dir.mktmpdir do |dir|
      MLX.random_seed(0)
      config = MLX::Models::LlamaConfig.new(config_hash)
      original = MLX::Models::Llama.new(config)

      File.write(File.join(dir, "config.json"), JSON.dump(config_hash))
      # Persist weights with the HF `model.` prefix so we exercise the
      # loader's prefix-stripping path.
      hf_weights = original.named_parameters.transform_keys do |k|
        k == "lm_head.weight" ? k : "model.#{k}"
      end
      MLX::IO.save_safetensors(hf_weights, File.join(dir, "model.safetensors"))

      reloaded = MLX::IO.load_huggingface(dir)
      expect(reloaded).to be_a(MLX::Models::Llama)
      expect(reloaded.named_parameters.size).to eq(original.named_parameters.size)

      tokens_original = original.generate(prompt, max_new_tokens: 5)
      tokens_reloaded = reloaded.generate(prompt, max_new_tokens: 5)
      expect(tokens_reloaded).to eq(tokens_original)
    end
  end

  it "loads a sharded checkpoint via the index file" do
    Dir.mktmpdir do |dir|
      MLX.random_seed(1)
      config = MLX::Models::LlamaConfig.new(config_hash)
      original = MLX::Models::Llama.new(config)
      File.write(File.join(dir, "config.json"), JSON.dump(config_hash))

      params = original.named_parameters
      # Split params into two shards.
      keys = params.keys
      shard_a = keys[0, keys.size / 2]
      shard_b = keys[keys.size / 2..]

      to_hf = lambda do |list|
        list.to_h { |k| [k == "lm_head.weight" ? k : "model.#{k}", params[k]] }
      end

      MLX::IO.save_safetensors(to_hf.call(shard_a), File.join(dir, "model-00001-of-00002.safetensors"))
      MLX::IO.save_safetensors(to_hf.call(shard_b), File.join(dir, "model-00002-of-00002.safetensors"))

      index = { "weight_map" => {} }
      shard_a.each { |k| index["weight_map"][k == "lm_head.weight" ? k : "model.#{k}"] = "model-00001-of-00002.safetensors" }
      shard_b.each { |k| index["weight_map"][k == "lm_head.weight" ? k : "model.#{k}"] = "model-00002-of-00002.safetensors" }
      File.write(File.join(dir, "model.safetensors.index.json"), JSON.dump(index))

      reloaded = MLX::IO.load_huggingface(dir)
      expect(reloaded.generate(prompt, max_new_tokens: 3))
        .to eq(original.generate(prompt, max_new_tokens: 3))
    end
  end
end
