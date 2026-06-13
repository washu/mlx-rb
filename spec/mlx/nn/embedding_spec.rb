# frozen_string_literal: true

require "spec_helper"

RSpec.describe MLX::NN::Embedding do
  it "returns rows of the weight matrix for integer indices" do
    emb = MLX::NN::Embedding.new(5, 3)
    w = MLX::Array.new([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2], [4.0, 4.1, 4.2]])
    emb.update("weight" => w)
    out = emb.call(MLX::Array.new([0, 2, 4], dtype: :int32)).to_a
    expect_close(out, [[0.0, 0.1, 0.2], [2.0, 2.1, 2.2], [4.0, 4.1, 4.2]], tol: 1e-5)
  end

  it "matches the python oracle", :oracle do
    w = (0..14).each_slice(3).map { |t| t.map(&:to_f) }
    ids = [0, 2, 4]
    ruby = (
      emb = MLX::NN::Embedding.new(5, 3)
      emb.update("weight" => MLX::Array.new(w))
      emb.call(MLX::Array.new(ids, dtype: :int32))
    ).to_a

    oracle = PythonOracle.run_script(<<~PY, inputs: [w, ids])
      W = mx.array(INPUTS[0], dtype=mx.float32)
      ids = mx.array(INPUTS[1], dtype=mx.int32)
      emb = mxnn.Embedding(5, 3)
      emb.weight = W
      emit(emb(ids))
    PY
    expect_close(ruby, oracle, tol: 1e-6)
  end
end
