# frozen_string_literal: true

require "json"
require "ffi"

module MLX
  module IO
    # Pure-Ruby safetensors reader and writer.
    #
    # File layout (per https://github.com/huggingface/safetensors):
    #   <u64 little-endian header_len>
    #   <header_len bytes of UTF-8 JSON>
    #   <byte region for tensors, packed in `data_offsets` order>
    #
    # JSON header is a hash of `name => {dtype, shape, data_offsets}` plus an
    # optional `"__metadata__"` key for arbitrary string metadata.
    module Safetensors
      module_function

      # Mapping from safetensors dtype tag to MLX dtype symbol. Only the
      # subset that MLX::DType currently handles is loadable; anything else
      # raises (a future phase can widen this if needed).
      DTYPE_MAP = {
        "F32"  => :float32,
        "F16"  => :float16,
        "BF16" => :bfloat16,
        "I32"  => :int32,
        "I64"  => :int64,
        "U16"  => :uint16,
        "U32"  => :uint32,
        "BOOL" => :bool
      }.freeze

      DTYPE_TAG = DTYPE_MAP.invert.freeze

      def load(path)
        File.open(path, "rb") do |io|
          header_len = io.read(8).unpack1("Q<")
          header_json = io.read(header_len)
          header = JSON.parse(header_json)
          header.delete("__metadata__")
          data_start = 8 + header_len

          tensors = {}
          header.each do |name, info|
            dtype = DTYPE_MAP[info["dtype"]] or
              raise MLX::DTypeError, "unsupported safetensors dtype #{info["dtype"]} for #{name.inspect}"

            shape = info["shape"]
            offs  = info["data_offsets"]
            nbytes = offs[1] - offs[0]
            io.seek(data_start + offs[0])
            bytes = io.read(nbytes)
            buf = ::FFI::MemoryPointer.new(:uint8, [nbytes, 1].max)
            buf.write_bytes(bytes) if nbytes.positive?
            tensors[name] = MLX::Array.from_buffer(buf, shape, dtype)
          end
          tensors
        end
      end

      def load_metadata(path)
        File.open(path, "rb") do |io|
          header_len = io.read(8).unpack1("Q<")
          header = JSON.parse(io.read(header_len))
          header["__metadata__"] || {}
        end
      end

      def save(tensors, path, metadata: {})
        # 1. Materialize each tensor to a contiguous float-array's worth of
        #    bytes by reading mlx-c's storage pointer. We only support dtypes
        #    where MLX::Array can deliver a row-major buffer.
        ordered = tensors.to_a
        entries = []
        cursor = 0
        ordered.each do |name, arr|
          unless arr.is_a?(MLX::Array)
            raise MLX::TypeError, "save tensor #{name.inspect} is #{arr.class}, need MLX::Array"
          end

          tag = DTYPE_TAG[arr.dtype] or
            raise MLX::DTypeError, "no safetensors dtype tag for #{arr.dtype}"

          nbytes = arr.size * MLX::DType.bytesize(arr.dtype)
          entries << {
            name: name,
            tag: tag,
            shape: arr.shape,
            offset: cursor,
            nbytes: nbytes,
            array: arr
          }
          cursor += nbytes
        end

        header = {}
        meta = metadata.transform_keys(&:to_s).transform_values(&:to_s)
        header["__metadata__"] = meta unless meta.empty?
        entries.each do |e|
          header[e[:name].to_s] = {
            "dtype" => e[:tag],
            "shape" => e[:shape],
            "data_offsets" => [e[:offset], e[:offset] + e[:nbytes]]
          }
        end
        header_json = JSON.generate(header)
        # safetensors requires 8-byte alignment of the data region after
        # the header; pad with spaces (valid JSON whitespace).
        padding = (8 - (header_json.bytesize % 8)) % 8
        header_json += (" " * padding)

        File.open(path, "wb") do |io|
          io.write([header_json.bytesize].pack("Q<"))
          io.write(header_json)
          entries.each do |e|
            io.write(raw_bytes_for(e[:array]))
          end
        end
        path
      end

      # Read out the byte representation of an MLX::Array. We force a
      # contiguous copy first (transpose returns a view whose storage is the
      # pre-transpose buffer; serializing without contiguous-ing would write
      # the wrong layout).
      def raw_bytes_for(arr)
        contig = arr.send(:ensure_contiguous)
        nbytes = arr.size * MLX::DType.bytesize(arr.dtype)
        ptr = case arr.dtype
              when :float32  then MLX::FFI.mlx_array_data_float32(contig.struct)
              when :float16  then MLX::FFI.mlx_array_data_float16(contig.struct)
              when :bfloat16 then MLX::FFI.mlx_array_data_bfloat16(contig.struct)
              when :int32    then MLX::FFI.mlx_array_data_int32(contig.struct)
              when :int64    then MLX::FFI.mlx_array_data_int64(contig.struct)
              when :uint16   then MLX::FFI.mlx_array_data_uint16(contig.struct)
              when :uint32   then MLX::FFI.mlx_array_data_uint32(contig.struct)
              when :bool     then MLX::FFI.mlx_array_data_bool(contig.struct)
              end
        if ptr.nil?
          raise MLX::DTypeError, "saving dtype #{arr.dtype} not yet supported"
        end

        ptr.read_bytes(nbytes)
      end
    end

    module_function

    def load_safetensors(path)
      Safetensors.load(path)
    end

    def save_safetensors(tensors, path, metadata: {})
      Safetensors.save(tensors, path, metadata: metadata)
    end

    def load_safetensors_metadata(path)
      Safetensors.load_metadata(path)
    end
  end
end
