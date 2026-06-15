# frozen_string_literal: true

# extconf.rb — invoked by `gem install mlx-rb` on a *source* gem.
#
# Builds the Rust bridge crate with `cargo build --release` and copies
# the resulting `libmlx_bridge.dylib` to `ext/mlx_bridge/lib/` where
# `lib/mlx/ffi.rb` will find it.
#
# For platform gems (the arm64-darwin precompiled variant produced by
# `rake native_gem`), the dylib is already inside the gem and this
# script is skipped — see Rakefile.

require "mkmf"
require "fileutils"

RUST_DIR = __dir__
CARGO    = ENV["CARGO"] || "cargo"

abort "mlx-rb requires macOS arm64" unless RbConfig::CONFIG["host_os"].start_with?("darwin") &&
                                          ["arm64", "aarch64"].include?(RbConfig::CONFIG["host_cpu"])

unless system("#{CARGO} --version >/dev/null 2>&1")
  abort <<~MSG
    cargo not found on PATH. Either:
      * install Rust via https://rustup.rs/, then re-run `gem install mlx-rb`
      * or install the precompiled arm64-darwin gem instead:
          gem install mlx-rb --platform arm64-darwin
  MSG
end

# Use the same env tweaks our build needs at dev-time. Users with full
# Xcode get the right toolchain selected automatically; CommandLineTools-
# only systems still work because mlx-sys vendors the kernel sources.
env = ENV.to_h
env["DEVELOPER_DIR"] ||= "/Applications/Xcode.app/Contents/Developer" if File.directory?("/Applications/Xcode.app")
env["LIBCLANG_PATH"] ||= "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib" if File.directory?("/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib")

cmd = "cd #{RUST_DIR.shellescape} && #{CARGO} build --release"
puts "[mlx-rb] #{cmd}"
unless system(env, cmd)
  abort "[mlx-rb] cargo build failed"
end

src  = File.join(RUST_DIR, "target", "release", "libmlx_bridge.dylib")
dest_dir = File.join(RUST_DIR, "lib")
dest = File.join(dest_dir, "libmlx_bridge.dylib")
abort "[mlx-rb] expected #{src} after cargo build but it's missing" unless File.exist?(src)

FileUtils.mkdir_p(dest_dir)
FileUtils.cp(src, dest)
puts "[mlx-rb] installed #{dest}"

# mkmf wants a Makefile; emit a no-op one so RubyGems doesn't complain.
File.write("Makefile", "all install clean:\n\t@true\n")
