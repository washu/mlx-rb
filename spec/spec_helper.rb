# frozen_string_literal: true

require "mlx"
require_relative "support/python_oracle"

RSpec.configure do |config|
  config.example_status_persistence_file_path = ".rspec_status"
  config.disable_monkey_patching!

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end

  config.filter_run_excluding oracle: true unless PythonOracle.available?
end

module SpecHelpers
  # Flatten and compare with float tolerance.
  def expect_close(actual, expected, tol: 1e-5)
    flat_a = actual.flatten
    flat_e = expected.flatten
    expect(flat_a.size).to eq(flat_e.size),
                            "size mismatch: ruby=#{flat_a.size} oracle=#{flat_e.size}"
    flat_a.zip(flat_e).each_with_index do |(a, b), i|
      diff = (a.to_f - b.to_f).abs
      expect(diff).to be <= tol,
                      "index #{i}: ruby=#{a} oracle=#{b} diff=#{diff}"
    end
  end
end

RSpec.configure do |config|
  config.include SpecHelpers
end
