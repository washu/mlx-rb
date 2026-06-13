# frozen_string_literal: true

module MLX
  module Models
    # Llama (decoder-only). Compatible with the Llama 3 family config layout:
    # config.json keys `hidden_size`, `intermediate_size`, `num_hidden_layers`,
    # `num_attention_heads`, `num_key_value_heads`, `rms_norm_eps`,
    # `rope_theta`, `vocab_size`, `tie_word_embeddings`. Grouped-query
    # attention is supported when `num_key_value_heads < num_attention_heads`.
    #
    # This is the reference architecture for Phase 3. Other architectures
    # (Mistral, Qwen, Gemma) can be added by third-party gems following the
    # same shape.
    class LlamaConfig
      attr_reader :hidden_size, :intermediate_size, :num_hidden_layers,
                  :num_attention_heads, :num_key_value_heads, :head_dim,
                  :rms_norm_eps, :rope_theta, :vocab_size, :tie_word_embeddings,
                  :max_position_embeddings

      def initialize(hash)
        h = hash.transform_keys(&:to_s)
        @hidden_size         = Integer(h.fetch("hidden_size"))
        @intermediate_size   = Integer(h.fetch("intermediate_size"))
        @num_hidden_layers   = Integer(h.fetch("num_hidden_layers"))
        @num_attention_heads = Integer(h.fetch("num_attention_heads"))
        @num_key_value_heads = Integer(h.fetch("num_key_value_heads", @num_attention_heads))
        @head_dim            = Integer(h.fetch("head_dim", @hidden_size / @num_attention_heads))
        @rms_norm_eps        = Float(h.fetch("rms_norm_eps", 1e-5))
        @rope_theta          = Float(h.fetch("rope_theta", 10_000.0))
        @vocab_size          = Integer(h.fetch("vocab_size"))
        @tie_word_embeddings = h.fetch("tie_word_embeddings", false) ? true : false
        @max_position_embeddings = Integer(h.fetch("max_position_embeddings", 2048))
      end
    end

    # Rotary positional embedding applied to (B, H, T, Dh). Pre-computes the
    # cos/sin tables for positions [0, max_seq_len) once and slices into them
    # for each call. Tables are computed in float32 and broadcast against the
    # query dtype on use.
    class LlamaRoPE
      def initialize(head_dim:, max_seq_len:, theta:)
        @head_dim = head_dim
        @max_seq_len = max_seq_len
        @theta = theta
        @cos, @sin = build_tables
      end

      # x: (B, H, T, Dh). offset: starting position index (for KV-cached
      # generation). Returns the rotated tensor with same shape.
      def call(x, offset: 0)
        b, h, t, dh = x.shape
        cos = @cos.slice([offset, 0], [offset + t, dh])
        sin = @sin.slice([offset, 0], [offset + t, dh])
        cos = cos.reshape([1, 1, t, dh])
        sin = sin.reshape([1, 1, t, dh])

        half = dh / 2
        x1 = x.slice([0, 0, 0, 0],    [b, h, t, half])
        x2 = x.slice([0, 0, 0, half], [b, h, t, dh])
        rot = MLX::Array.concatenate([-x2, x1], axis: -1)
        (x * cos) + (rot * sin)
      end

      private

      def build_tables
        half = @head_dim / 2
        inv_freq = MLX::Array.arange(0, half, 1, dtype: :float32) *
                   MLX::Array.new(2.0 / @head_dim)
        inv_freq = MLX::Array.new(@theta)**(-inv_freq)
        t = MLX::Array.arange(0, @max_seq_len, 1, dtype: :float32)
        freqs = t.reshape([@max_seq_len, 1]).matmul(inv_freq.reshape([1, half]))
        emb = MLX::Array.concatenate([freqs, freqs], axis: -1)
        [emb.cos, emb.sin]
      end
    end

    class LlamaAttention < MLX::NN::Module
      def initialize(config)
        super()
        @num_heads = config.num_attention_heads
        @num_kv_heads = config.num_key_value_heads
        @head_dim = config.head_dim
        @rep = @num_heads / @num_kv_heads

        @q_proj = MLX::NN::Linear.new(config.hidden_size, @num_heads * @head_dim, bias: false)
        @k_proj = MLX::NN::Linear.new(config.hidden_size, @num_kv_heads * @head_dim, bias: false)
        @v_proj = MLX::NN::Linear.new(config.hidden_size, @num_kv_heads * @head_dim, bias: false)
        @o_proj = MLX::NN::Linear.new(@num_heads * @head_dim, config.hidden_size, bias: false)
      end

      # x: (B, T, D). cache: optional {k:, v:} hash for autoregressive decode.
      # rope: a LlamaRoPE instance.
      def forward(x, rope:, cache: nil, mask: nil)
        b, t, = x.shape
        q = @q_proj.call(x).reshape([b, t, @num_heads, @head_dim]).transpose([0, 2, 1, 3])
        k = @k_proj.call(x).reshape([b, t, @num_kv_heads, @head_dim]).transpose([0, 2, 1, 3])
        v = @v_proj.call(x).reshape([b, t, @num_kv_heads, @head_dim]).transpose([0, 2, 1, 3])

        offset = cache ? cache[:offset] : 0
        q = rope.call(q, offset: offset)
        k = rope.call(k, offset: offset)

        if cache && cache[:k]
          k = MLX::Array.concatenate([cache[:k], k], axis: 2)
          v = MLX::Array.concatenate([cache[:v], v], axis: 2)
        end
        if cache
          cache[:k] = k
          cache[:v] = v
          cache[:offset] = offset + t
        end

        # GQA: repeat kv heads to match num_heads.
        if @rep > 1
          k = k.repeat(@rep, axis: 1)
          v = v.repeat(@rep, axis: 1)
        end

        scale = 1.0 / Math.sqrt(@head_dim)
        out = MLX::FFI.mlx_array_new
        MLX.check!(
          MLX::FFI.mlx_fast_scaled_dot_product_attention(
            out.pointer, q.struct, k.struct, v.struct, scale,
            mask || "", MLX::FFI.null_array, MLX::FFI.null_array,
            MLX.stream_struct
          ),
          "mlx_fast_scaled_dot_product_attention"
        )
        attn = MLX::Array.from_struct(out)
        attn = attn.transpose([0, 2, 1, 3]).reshape([b, t, @num_heads * @head_dim])
        @o_proj.call(attn)
      end
    end

    class LlamaMLP < MLX::NN::Module
      def initialize(config)
        super()
        @gate_proj = MLX::NN::Linear.new(config.hidden_size, config.intermediate_size, bias: false)
        @up_proj   = MLX::NN::Linear.new(config.hidden_size, config.intermediate_size, bias: false)
        @down_proj = MLX::NN::Linear.new(config.intermediate_size, config.hidden_size, bias: false)
      end

      def forward(x)
        @down_proj.call(MLX::NN::F.silu(@gate_proj.call(x)) * @up_proj.call(x))
      end
    end

    class LlamaBlock < MLX::NN::Module
      attr_reader :self_attn, :mlp, :input_layernorm, :post_attention_layernorm

      def initialize(config)
        super()
        @self_attn = LlamaAttention.new(config)
        @mlp       = LlamaMLP.new(config)
        @input_layernorm          = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)
        @post_attention_layernorm = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)
      end

      def forward(x, rope:, cache: nil, mask: nil)
        h = x + @self_attn.call(@input_layernorm.call(x), rope: rope, cache: cache, mask: mask)
        h + @mlp.call(@post_attention_layernorm.call(h))
      end
    end

    # The model itself. `forward` accepts integer token ids (shape (B, T)) and
    # returns logits over the vocabulary.
    class Llama < MLX::NN::Module
      attr_reader :config

      # HuggingFace's Llama checkpoint names every weight under a `model.`
      # prefix (e.g. `model.layers.0.self_attn.q_proj.weight`). Our module
      # tree skips that, so strip it on load. `lm_head.weight` stays as-is.
      def self.remap_weights(weights)
        out = {}
        weights.each do |name, arr|
          if name.start_with?("model.")
            out[name.sub(/\Amodel\./, "")] = arr
          else
            out[name] = arr
          end
        end
        out
      end

      def initialize(config)
        super()
        @config = config
        @embed_tokens = MLX::NN::Embedding.new(config.vocab_size, config.hidden_size)
        @layers = ::Array.new(config.num_hidden_layers) { LlamaBlock.new(config) }
        @norm   = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)
        @lm_head = if config.tie_word_embeddings
                     nil
                   else
                     MLX::NN::Linear.new(config.hidden_size, config.vocab_size, bias: false)
                   end
        @_rope = LlamaRoPE.new(
          head_dim: config.head_dim,
          max_seq_len: config.max_position_embeddings,
          theta: config.rope_theta
        )
      end

      # tokens: MLX::Array of int32 with shape (B, T). caches: optional array
      # of per-layer cache hashes, one per decoder block. Returns logits
      # of shape (B, T, vocab_size).
      def forward(tokens, caches: nil)
        h = @embed_tokens.call(tokens)
        t = h.shape[1]
        # `causal` whenever query length > 1 — q attends to its own prefix.
        # When t == 1 no mask is needed (one query can attend to all keys).
        mask = t > 1 ? "causal" : ""
        @layers.each_with_index do |layer, i|
          h = layer.call(h, rope: @_rope, cache: caches&.[](i), mask: mask)
        end
        h = @norm.call(h)
        if @lm_head
          @lm_head.call(h)
        else
          # weight tying: logits = h @ E^T
          h.matmul(@embed_tokens.instance_variable_get(:@weight).transpose)
        end
      end

      # Build an empty cache structure matching the number of layers.
      def make_caches
        ::Array.new(@config.num_hidden_layers) { { k: nil, v: nil, offset: 0 } }
      end

      # Greedy generation. Prompt is an Array<Integer> of token ids. Returns
      # an Array<Integer> of newly generated ids (does not include the prompt).
      def generate(prompt_ids, max_new_tokens:)
        ids = prompt_ids.dup
        caches = make_caches

        # Prefill in one pass with the whole prompt.
        toks = MLX::Array.new([ids], dtype: :int32)
        logits = forward(toks, caches: caches)
        next_id = pick_last(logits)
        out = [next_id]
        ids << next_id

        (max_new_tokens - 1).times do
          toks = MLX::Array.new([[next_id]], dtype: :int32)
          logits = forward(toks, caches: caches)
          next_id = pick_last(logits)
          out << next_id
          ids << next_id
        end
        out
      end

      private

      # logits: (B=1, T, V) — argmax of the last position. Returns a Ruby Integer.
      def pick_last(logits)
        _, t, v = logits.shape
        last = logits.slice([0, t - 1, 0], [1, t, v]).reshape([v])
        last = last.astype(:float32) if last.dtype != :float32
        last.argmax.astype(:int32).to_a
      end
    end
  end
end
