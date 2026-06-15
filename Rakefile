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

desc "Build the Rust bridge crate (release) and copy the dylib under ext/mlx_bridge/lib"
task :compile do
  env = {}
  env["DEVELOPER_DIR"] ||= "/Applications/Xcode.app/Contents/Developer" if File.directory?("/Applications/Xcode.app")
  env["LIBCLANG_PATH"] ||= "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib" if File.directory?("/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib")
  sh env, "cargo build --release --manifest-path #{File.join(BRIDGE_DIR, 'Cargo.toml')}"
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
    FileUtils.mkdir_p("pkg")

    # Source gem — Cargo runs at install time on the user's machine.
    sh "gem build mlx-rb.gemspec --output pkg/mlx-rb-#{MLX::VERSION}.gem"

    # Precompiled platform gem — dylib already inside the gem.
    ENV["MLX_RB_PLATFORM"] = "arm64-darwin"
    begin
      sh "gem build mlx-rb.gemspec --output pkg/mlx-rb-#{MLX::VERSION}-arm64-darwin.gem"
    ensure
      ENV.delete("MLX_RB_PLATFORM")
    end

    puts "\nBuilt:"
    Dir["pkg/mlx-rb-#{MLX::VERSION}*.gem"].sort.each { |g| puts "  #{g}  (#{File.size(g) / 1024} KB)" }
  end
end

# Load MLX::VERSION for the release:gems task.
require_relative "lib/mlx/version"
