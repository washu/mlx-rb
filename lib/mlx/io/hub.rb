# frozen_string_literal: true

require "net/http"
require "uri"
require "json"
require "fileutils"
require "pathname"

module MLX
  module IO
    # Native HuggingFace Hub downloader.
    #
    # Layout matches `huggingface_hub`'s cache convention so a checkpoint
    # downloaded here is interchangeable with one downloaded via the
    # Python CLI:
    #
    #   $HF_HOME/hub/models--<org>--<name>/
    #     refs/<revision>         # text file: the resolved commit sha
    #     blobs/<oid>             # raw file bytes, named by HF's oid
    #     snapshots/<commit>/<path>   # symlink → ../../blobs/<oid>
    #
    # No new gem dependencies; everything goes through stdlib
    # `Net::HTTP`. LFS redirects are followed, `Range` resumes a partial
    # download, and a configurable thread pool fans out across files.
    module Hub
      API_ROOT  = "https://huggingface.co/api/models"
      CDN_ROOT  = "https://huggingface.co"
      DEFAULT_WORKERS = 8

      class Error < StandardError; end
      class HTTPError < Error; end
      class AuthError < Error; end

      module_function

      # Resolve and download the matching files of `repo_id` at the
      # given `revision`, returning the absolute path of the snapshot
      # directory.
      #
      # @param repo_id [String] "org/name"
      # @param revision [String] branch, tag, or commit. Default "main".
      # @param allow_patterns [Array<String>, nil] glob list; only matching
      #   files are downloaded. nil = everything.
      # @param workers [Integer] concurrent file downloads.
      # @param progress [Proc, nil] callback invoked as
      #   `progress.call(path, bytes_so_far, total_bytes)`.
      def download(repo_id, revision: "main", allow_patterns: nil,
                   workers: DEFAULT_WORKERS, progress: nil, cache_dir: nil)
        cache_dir ||= default_cache_dir
        repo_dir  = File.join(cache_dir, "models--#{repo_id.gsub("/", "--")}")

        meta = fetch_meta(repo_id, revision)
        commit = meta.fetch("sha")
        tree   = fetch_tree(repo_id, revision)

        files = tree.select { |entry| entry["type"] == "file" }
        files = filter_files(files, allow_patterns) if allow_patterns

        snapshot_dir = File.join(repo_dir, "snapshots", commit)
        FileUtils.mkdir_p(File.join(repo_dir, "blobs"))
        FileUtils.mkdir_p(snapshot_dir)
        write_ref(repo_dir, revision, commit)

        download_all(repo_id, revision, files, repo_dir, snapshot_dir,
                     workers: workers, progress: progress)
        snapshot_dir
      end

      # The standard $HF_HOME/hub directory. Honors HF_HOME first, then
      # ~/.cache/huggingface, matching `huggingface_hub`.
      def default_cache_dir
        hub = ENV["HF_HOME"] || File.join(Dir.home, ".cache/huggingface")
        File.join(hub, "hub")
      end

      # Look up an already-downloaded snapshot for the repo. Returns the
      # snapshot directory path, or nil if no cache exists.
      def cached_snapshot(repo_id, cache_dir: nil)
        cache_dir ||= default_cache_dir
        repo_dir = File.join(cache_dir, "models--#{repo_id.gsub("/", "--")}")
        snapshots = Dir[File.join(repo_dir, "snapshots", "*")]
        snapshots.find { |d| File.directory?(d) }
      end

      # ---- HTTP -----------------------------------------------------

      def fetch_meta(repo_id, revision)
        json_get("#{API_ROOT}/#{repo_id}/revision/#{revision}")
      end

      def fetch_tree(repo_id, revision)
        out = []
        cursor = nil
        loop do
          url = "#{API_ROOT}/#{repo_id}/tree/#{revision}?recursive=true"
          url += "&cursor=#{cursor}" if cursor
          resp = http_get(URI.parse(url))
          batch = JSON.parse(resp.body)
          out.concat(batch)
          # HF paginates large repos via X-Linked-Next; small repos
          # return everything in one shot.
          link = resp["X-Linked-Next"] || resp["link"]
          break unless link && (m = link.match(/cursor=([^&>]+)/))

          cursor = m[1]
        end
        out
      end

      def json_get(url)
        resp = http_get(URI.parse(url))
        JSON.parse(resp.body)
      end

      def http_get(uri, headers: {})
        request_with_redirects(uri, "GET", headers: headers)
      end

      def request_with_redirects(uri, method, headers: {}, body: nil, limit: 5)
        raise HTTPError, "too many redirects" if limit.zero?

        Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https") do |http|
          req = Net::HTTPGenericRequest.new(method, body ? true : false, true, uri.request_uri)
          headers.each { |k, v| req[k] = v }
          inject_auth(req)
          req["User-Agent"] = "mlx-rb/#{MLX::VERSION}"
          req.body = body if body

          resp = http.request(req)
          case resp
          when Net::HTTPSuccess, Net::HTTPPartialContent
            resp
          when Net::HTTPRedirection
            new_uri = URI.parse(resp["location"])
            new_uri = uri + resp["location"] if new_uri.host.nil?
            request_with_redirects(new_uri, method, headers: headers, body: body, limit: limit - 1)
          when Net::HTTPUnauthorized, Net::HTTPForbidden
            raise AuthError, "#{resp.code} for #{uri} — set HF_TOKEN or check repo access"
          else
            raise HTTPError, "#{resp.code} for #{uri}: #{resp.body[0, 200]}"
          end
        end
      end

      def inject_auth(req)
        token = ENV["HF_TOKEN"] || read_token_file
        req["Authorization"] = "Bearer #{token}" if token
      end

      def read_token_file
        path = File.expand_path("~/.cache/huggingface/token")
        return nil unless File.exist?(path)

        File.read(path).strip.then { |s| s.empty? ? nil : s }
      end

      # ---- download orchestration -----------------------------------

      def filter_files(files, patterns)
        files.select do |entry|
          patterns.any? { |pat| File.fnmatch?(pat, entry["path"], File::FNM_PATHNAME) }
        end
      end

      def write_ref(repo_dir, revision, commit)
        ref_path = File.join(repo_dir, "refs", revision)
        FileUtils.mkdir_p(File.dirname(ref_path))
        File.write(ref_path, commit)
      end

      def download_all(repo_id, revision, files, repo_dir, snapshot_dir,
                       workers:, progress:)
        queue = Queue.new
        files.each { |f| queue << f }
        workers.times { queue << :done }

        threads = workers.times.map do
          Thread.new do
            loop do
              entry = queue.pop
              break if entry == :done

              download_one(repo_id, revision, entry, repo_dir, snapshot_dir, progress)
            end
          end
        end
        threads.each(&:join)
      end

      def download_one(repo_id, revision, entry, repo_dir, snapshot_dir, progress)
        path = entry.fetch("path")
        oid  = entry.dig("lfs", "oid") || entry.fetch("oid")
        size = entry.dig("lfs", "size") || entry["size"] || 0

        blob_path = File.join(repo_dir, "blobs", oid)
        link_path = File.join(snapshot_dir, path)
        FileUtils.mkdir_p(File.dirname(link_path))

        if File.exist?(blob_path) && File.size(blob_path) == size
          ensure_symlink(blob_path, link_path)
          progress&.call(path, size, size)
          return
        end

        partial = "#{blob_path}.incomplete"
        start_byte = File.exist?(partial) ? File.size(partial) : 0
        url = URI.parse("#{CDN_ROOT}/#{repo_id}/resolve/#{revision}/#{path}")

        headers = start_byte.positive? ? { "Range" => "bytes=#{start_byte}-" } : {}
        stream_to_file(url, partial, start_byte, size, headers, path, progress)

        File.rename(partial, blob_path)
        ensure_symlink(blob_path, link_path)
      end

      def ensure_symlink(target, link)
        File.delete(link) if File.symlink?(link) || File.exist?(link)
        rel = Pathname.new(target).relative_path_from(Pathname.new(File.dirname(link)))
        File.symlink(rel, link)
      rescue Errno::EPERM, NotImplementedError
        FileUtils.cp(target, link) # Windows/non-symlink fallback
      end

      def stream_to_file(uri, dest, start_byte, total, headers, display_path, progress)
        bytes = start_byte
        request_with_redirects_stream(uri, headers) do |chunk|
          File.open(dest, bytes.zero? ? "wb" : "ab") do |f|
            f.write(chunk)
          end
          bytes += chunk.bytesize
          progress&.call(display_path, bytes, total)
        end
      end

      def request_with_redirects_stream(uri, headers, limit: 5, &block)
        raise HTTPError, "too many redirects" if limit.zero?

        redirect_target = nil
        Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https") do |http|
          req = Net::HTTP::Get.new(uri.request_uri)
          headers.each { |k, v| req[k] = v }
          inject_auth(req)
          req["User-Agent"] = "mlx-rb/#{MLX::VERSION}"

          http.request(req) do |resp|
            case resp
            when Net::HTTPSuccess, Net::HTTPPartialContent
              resp.read_body(&block)
            when Net::HTTPRedirection
              new_uri = URI.parse(resp["location"])
              new_uri = uri + resp["location"] if new_uri.host.nil?
              redirect_target = new_uri
            when Net::HTTPUnauthorized, Net::HTTPForbidden
              raise AuthError, "#{resp.code} for #{uri}"
            else
              raise HTTPError, "#{resp.code} for #{uri}"
            end
          end
        end
        return unless redirect_target

        request_with_redirects_stream(redirect_target, headers, limit: limit - 1, &block)
      end
    end
  end
end
