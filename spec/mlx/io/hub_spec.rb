# frozen_string_literal: true

require "spec_helper"
require "socket"
require "json"
require "tmpdir"

# Tiny in-process HTTP server that speaks just enough of the HF Hub API
# for Hub#download. We avoid pulling WebMock as a dev dep; stdlib only.
class FakeHub
  attr_reader :port, :requests

  def initialize(files)
    @files = files          # { "path/in/repo" => "raw bytes" }
    @requests = []
    @server = TCPServer.new("127.0.0.1", 0)
    @port = @server.addr[1]
    @thread = Thread.new { serve_loop }
  end

  def stop
    @stop = true
    Thread.new { TCPSocket.new("127.0.0.1", @port).close rescue nil }
    @thread.join(1)
    @server.close
  end

  private

  def serve_loop
    until @stop
      begin
        client = @server.accept
        handle(client)
      rescue StandardError
        next
      end
    end
  end

  def handle(client)
    req_line = client.gets&.chomp
    headers = {}
    while (line = client.gets) && line != "\r\n"
      k, v = line.chomp.split(": ", 2)
      headers[k.downcase] = v
    end
    @requests << { line: req_line, headers: headers }
    method, path, _ = req_line.split(" ")
    respond(client, method, path, headers)
  rescue StandardError => e
    warn "fake hub error: #{e}"
  ensure
    client&.close
  end

  def respond(client, _method, path, headers)
    if path.start_with?("/api/models/")
      # /api/models/{repo}/revision/{rev}   -> meta with sha
      # /api/models/{repo}/tree/{rev}       -> list of files
      if path.include?("/revision/")
        body = JSON.dump("sha" => "deadbeef")
      elsif path.include?("/tree/")
        body = JSON.dump(@files.map { |p, bytes|
          { "type" => "file", "path" => p, "oid" => "oid-#{p.tr('/', '_')}", "size" => bytes.bytesize }
        })
      else
        body = "[]"
      end
      reply(client, 200, body, "application/json")
    elsif path.include?("/resolve/")
      # /<repo>/resolve/{rev}/<file_path>
      file_path = path.split("/resolve/", 2)[1].sub(%r{^[^/]+/}, "")
      data = @files[file_path]
      return reply(client, 404, "no such file") unless data

      if (range = headers["range"]) && (m = range.match(/bytes=(\d+)-/))
        start_byte = m[1].to_i
        slice = data[start_byte..]
        client.write "HTTP/1.1 206 Partial Content\r\n"
        client.write "Content-Length: #{slice.bytesize}\r\n"
        client.write "Content-Range: bytes #{start_byte}-#{data.bytesize - 1}/#{data.bytesize}\r\n\r\n"
        client.write(slice)
      else
        reply(client, 200, data, "application/octet-stream")
      end
    else
      reply(client, 404, "not found")
    end
  end

  def reply(client, code, body, ctype = "text/plain")
    client.write "HTTP/1.1 #{code} OK\r\n"
    client.write "Content-Type: #{ctype}\r\n"
    client.write "Content-Length: #{body.bytesize}\r\n\r\n"
    client.write(body)
  end
end

RSpec.describe MLX::IO::Hub do
  let(:files) do
    {
      "config.json"        => '{"hello":"world"}',
      "model.safetensors"  => ("\x00\x11\x22\x33" * 4096)
    }
  end

  around do |ex|
    @hub = FakeHub.new(files)
    original_api = MLX::IO::Hub::API_ROOT
    original_cdn = MLX::IO::Hub::CDN_ROOT
    MLX::IO::Hub.send(:remove_const, :API_ROOT)
    MLX::IO::Hub.send(:remove_const, :CDN_ROOT)
    MLX::IO::Hub.const_set(:API_ROOT, "http://127.0.0.1:#{@hub.port}/api/models")
    MLX::IO::Hub.const_set(:CDN_ROOT, "http://127.0.0.1:#{@hub.port}")
    ex.run
  ensure
    @hub.stop
    MLX::IO::Hub.send(:remove_const, :API_ROOT)
    MLX::IO::Hub.send(:remove_const, :CDN_ROOT)
    MLX::IO::Hub.const_set(:API_ROOT, original_api)
    MLX::IO::Hub.const_set(:CDN_ROOT, original_cdn)
  end

  it "downloads files into the huggingface_hub-compatible cache layout" do
    Dir.mktmpdir do |cache|
      snapshot = MLX::IO::Hub.download("org/repo", cache_dir: cache, workers: 2)

      expect(File.read(File.join(snapshot, "config.json"))).to eq('{"hello":"world"}')
      expect(File.read(File.join(snapshot, "model.safetensors")).bytesize)
        .to eq(files["model.safetensors"].bytesize)

      repo_dir = File.dirname(File.dirname(snapshot))
      expect(File.read(File.join(repo_dir, "refs", "main"))).to eq("deadbeef")
      expect(Dir.children(File.join(repo_dir, "blobs")).size).to eq(2)
      expect(File.symlink?(File.join(snapshot, "config.json"))).to be true
    end
  end

  it "resumes a partial download via HTTP Range" do
    Dir.mktmpdir do |cache|
      repo_dir = File.join(cache, "models--org--repo")
      FileUtils.mkdir_p(File.join(repo_dir, "blobs"))
      partial = File.join(repo_dir, "blobs", "oid-model.safetensors.incomplete")
      File.binwrite(partial, files["model.safetensors"][0, 100])

      MLX::IO::Hub.download("org/repo", cache_dir: cache, workers: 2)

      resume_requests = @hub.requests.select { |r| r[:headers].key?("range") }
      expect(resume_requests).not_to be_empty
      expect(resume_requests.first[:headers]["range"]).to eq("bytes=100-")

      final = File.join(repo_dir, "blobs", "oid-model.safetensors")
      expect(File.binread(final).bytesize).to eq(files["model.safetensors"].bytesize)
    end
  end

  it "filters by allow_patterns" do
    Dir.mktmpdir do |cache|
      snapshot = MLX::IO::Hub.download("org/repo",
                                      cache_dir: cache,
                                      allow_patterns: ["*.json"],
                                      workers: 2)
      expect(File.exist?(File.join(snapshot, "config.json"))).to be true
      expect(File.exist?(File.join(snapshot, "model.safetensors"))).to be false
    end
  end

  describe "real HF round-trip", :online do
    # Hits live HF. Auto-skipped unless HF_ONLINE=1 is set.
    before { skip "HF_ONLINE unset" unless ENV["HF_ONLINE"] == "1" }

    it "downloads a tiny public repo" do
      Dir.mktmpdir do |cache|
        snapshot = MLX::IO::Hub.download(
          "hf-internal-testing/tiny-random-LlamaForCausalLM",
          cache_dir: cache,
          allow_patterns: ["*.json"]
        )
        expect(File.exist?(File.join(snapshot, "config.json"))).to be true
      end
    end
  end
end
