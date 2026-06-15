# frozen_string_literal: true

require "bundler/gem_tasks"
require "rspec/core/rake_task"
require "fileutils"

RSpec::Core::RakeTask.new(:spec)

require "rubocop/rake_task"

RuboCop::RakeTask.new

task default: %i[spec rubocop]

# ---- Rust bridge compile + precompiled gem ----

BRIDGE_DIR = File.expand_path("ext/mlx_bridge", __dir__)
BRIDGE_DYLIB_SRC = File.join(BRIDGE_DIR, "target/release/libmlx_bridge.dylib")
BRIDGE_DYLIB_DST = File.join(BRIDGE_DIR, "lib/libmlx_bridge.dylib")

XCODE_DEV = "/Applications/Xcode.app/Contents/Developer"
XCODE_LIBCLANG = "#{XCODE_DEV}/Toolchains/XcodeDefault.xctoolchain/usr/lib".freeze

desc "Build the Rust bridge crate (release) and copy the dylib under ext/mlx_bridge/lib"
task :compile do
  env = {}
  env["DEVELOPER_DIR"] ||= XCODE_DEV if File.directory?(XCODE_DEV)
  env["LIBCLANG_PATH"] ||= XCODE_LIBCLANG if File.directory?(XCODE_LIBCLANG)
  sh env, "cargo build --release --manifest-path #{File.join(BRIDGE_DIR, "Cargo.toml")}"
  FileUtils.mkdir_p(File.dirname(BRIDGE_DYLIB_DST))
  FileUtils.cp(BRIDGE_DYLIB_SRC, BRIDGE_DYLIB_DST)
end

desc "Build the precompiled arm64-darwin platform gem (skips cargo at install)"
task native_gem: :compile do
  ENV["MLX_RB_PLATFORM"] = "arm64-darwin"
  Rake::Task["build"].invoke
ensure
  ENV.delete("MLX_RB_PLATFORM")
end

desc "Remove the bridge's build/lib outputs"
task :clobber_bridge do
  FileUtils.rm_rf(File.join(BRIDGE_DIR, "target"))
  FileUtils.rm_rf(File.join(BRIDGE_DIR, "lib"))
end

namespace :release do
  desc "Build both the source gem and the precompiled arm64-darwin gem into pkg/"
  task gems: :compile do
    require "bundler"
    FileUtils.mkdir_p("pkg")

    # gem build evaluates mlx-rb.gemspec which reads MLX_RB_PLATFORM.
    # We invoke it outside the current `bundle exec` environment —
    # otherwise Bundler tries to re-resolve the `path: .` source for
    # all RUBY_PLATFORMS, which fails when the gemspec emits a single
    # platform (arm64-darwin).
    build_gem = lambda do |output_name, env_overrides = {}|
      Bundler.with_unbundled_env do
        env_overrides.each { |k, v| ENV[k] = v }
        begin
          sh "gem build mlx-rb.gemspec --output #{output_name}"
        ensure
          env_overrides.each_key { |k| ENV.delete(k) }
        end
      end
    end

    build_gem.call("pkg/mlx-rb-#{MLX::VERSION}.gem")
    build_gem.call("pkg/mlx-rb-#{MLX::VERSION}-arm64-darwin.gem", "MLX_RB_PLATFORM" => "arm64-darwin")

    puts "\nBuilt:"
    Dir["pkg/mlx-rb-#{MLX::VERSION}*.gem"].sort.each { |g| puts "  #{g}  (#{File.size(g) / 1024} KB)" }
  end
end

# Load MLX::VERSION for the release:gems task.
require_relative "lib/mlx/version"
