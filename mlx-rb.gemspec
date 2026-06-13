# frozen_string_literal: true

require_relative "lib/mlx/version"

Gem::Specification.new do |spec|
  spec.name = "mlx-rb"
  spec.version = MLX::VERSION
  spec.authors = ["Sal Scotto"]
  spec.email = ["sal.scotto@gmail.com"]

  spec.summary = "Ruby bindings for Apple's MLX machine learning framework."
  spec.description = <<~DESC
    mlx-rb is a Ruby FFI binding over mlx-c, Apple's official C API for MLX.
    It exposes tensors, autograd, neural-network modules, optimizers,
    safetensors / HuggingFace loading, and 4/8-bit quantization. Apple
    Silicon only — MLX is built on Metal and unified memory.
  DESC
  spec.homepage = "https://github.com/washu/mlx-rb"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.1.0"

  spec.metadata["homepage_uri"]    = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["changelog_uri"]   = "#{spec.homepage}/blob/main/CHANGELOG.md"
  spec.metadata["bug_tracker_uri"] = "#{spec.homepage}/issues"
  spec.metadata["rubygems_mfa_required"] = "true"

  gemspec = File.basename(__FILE__)
  spec.files = IO.popen(%w[git ls-files -z], chdir: __dir__, err: IO::NULL) do |ls|
    ls.readlines("\x0", chomp: true).reject do |f|
      (f == gemspec) ||
        f.start_with?(*%w[test/ spec/ features/ bench/ .git .github .idea appveyor Gemfile.lock phase-]) ||
        f.end_with?("-summary.md") ||
        !File.exist?(File.join(__dir__, f))
    end
  end
  spec.bindir = "exe"
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]
  spec.platform = "arm64-darwin" # explicit Apple Silicon

  spec.add_dependency "ffi", "~> 1.16"
end
