# frozen_string_literal: true

require_relative "lib/mlx/version"

Gem::Specification.new do |spec|
  spec.name = "mlx-rb"
  spec.version = MLX::VERSION
  spec.authors = ["Sal Scotto"]
  spec.email = ["sal.scotto@gmail.com"]

  spec.summary = "Ruby bindings for Apple's MLX machine learning framework."
  spec.description = <<~DESC
    mlx-rb is a Ruby FFI binding over a Rust bridge crate (mlx-rs) that
    statically links MLX C++ and Metal into a single dylib. Apple
    Silicon only. Exposes tensors, autograd, neural-network modules,
    optimizers, safetensors / HuggingFace loading, 4/8-bit quantization,
    and a LoRA adapter API.
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
  tracked = IO.popen(%w[git ls-files -z], chdir: __dir__, err: IO::NULL) do |ls|
    ls.readlines("\x0", chomp: true).reject do |f|
      (f == gemspec) ||
        f.start_with?(*%w[test/ spec/ features/ bench/ .git .github .idea appveyor Gemfile.lock phase-]) ||
        f.end_with?("-summary.md") ||
        !File.exist?(File.join(__dir__, f))
    end
  end

  # Source gems carry the Rust crate sources and build at install time;
  # platform gems carry the precompiled dylib instead and skip cargo.
  is_platform_gem = (ENV["MLX_RB_PLATFORM"] || "ruby").start_with?("arm64-darwin")
  if is_platform_gem
    tracked = tracked.reject { |f| f.start_with?("ext/mlx_bridge/src/", "ext/mlx_bridge/Cargo.toml", "ext/mlx_bridge/build.rs", "ext/mlx_bridge/exports.txt", "ext/mlx_bridge/extconf.rb") }
    prebuilt = "ext/mlx_bridge/lib/libmlx_bridge.dylib"
    tracked << prebuilt if File.exist?(File.join(__dir__, prebuilt))
  end
  spec.files = tracked
  spec.bindir = "exe"
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  # The default gem is a source gem: it ships the Rust crate sources and
  # runs cargo at install time via ext/mlx_bridge/extconf.rb. The
  # `rake native_gem` task in Rakefile produces a precompiled
  # arm64-darwin variant that skips the cargo step.
  spec.platform   = ENV["MLX_RB_PLATFORM"] || "ruby"
  spec.extensions = ["ext/mlx_bridge/extconf.rb"] unless spec.platform.to_s.start_with?("arm64-darwin")

  spec.add_dependency "ffi", "~> 1.16"
end
