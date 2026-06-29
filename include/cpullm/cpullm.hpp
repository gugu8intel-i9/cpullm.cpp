#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cpullm {

class GgufFile;

struct CpuFeatures {
  bool sse2 = false;
  bool avx2 = false;
  bool avx512f = false;
  bool neon = false;
  std::string summary() const;
};

CpuFeatures detect_cpu_features();

class Arena {
public:
  explicit Arena(std::size_t bytes);
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;
  Arena(Arena&&) noexcept;
  Arena& operator=(Arena&&) noexcept;
  ~Arena();

  void* allocate(std::size_t bytes, std::size_t alignment = 64);
  void reset() noexcept;
  std::size_t used() const noexcept;
  std::size_t capacity() const noexcept;

private:
  std::byte* data_ = nullptr;
  std::size_t capacity_ = 0;
  std::size_t offset_ = 0;
};

enum class DataType : std::uint8_t { f32, f16, q4_0, q4_1, q8_0, unknown };

std::size_t dtype_size(DataType dtype);

struct TensorView {
  DataType dtype = DataType::f32;
  std::vector<std::size_t> shape;
  std::span<const std::byte> bytes;

  std::size_t elements() const;
};

struct MatmulPlan {
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  bool transpose_b = true;
};

void matmul_f32(const float* a, const float* b, float* c, MatmulPlan plan);

struct Q4Block {
  float scale = 0.0f;
  std::uint8_t packed[16]{};
};

std::vector<Q4Block> quantize_q4_0(std::span<const float> values);
float dot_q4_0_f32(std::span<const Q4Block> quantized, std::span<const float> x);
void matvec_q4_0_f32(std::span<const Q4Block> rows, std::span<const float> x,
                     std::span<float> y, std::size_t cols);

float fp16_to_f32(std::uint16_t bits) noexcept;
float dot_gguf_q4_0_f32(std::span<const std::byte> row, std::span<const float> x);
void matvec_gguf_q4_0_f32(std::span<const std::byte> rows, std::span<const float> x,
                          std::span<float> y, std::size_t cols);

class TensorStore {
public:
  void add(std::string name, DataType dtype, std::vector<std::size_t> shape, std::vector<std::byte> data);
  const TensorView* find(std::string_view name) const noexcept;
  std::size_t size() const noexcept;

private:
  struct Entry {
    std::string name;
    DataType dtype;
    std::vector<std::size_t> shape;
    std::vector<std::byte> data;
  };
  std::vector<Entry> entries_;
};

class Tokenizer {
public:
  explicit Tokenizer(std::vector<std::string> vocab = {});
  static Tokenizer from_gguf(const GgufFile& gguf);
  std::vector<std::uint32_t> encode(std::string_view text) const;
  std::string decode(std::span<const std::uint32_t> ids) const;
  std::size_t vocab_size() const noexcept;

private:
  std::vector<std::string> vocab_;
};

enum class ModelFormat : std::uint8_t { manifest, gguf_probe };

struct GgufProbe {
  bool valid = false;
  std::uint32_t version = 0;
  std::uint64_t tensor_count = 0;
  std::uint64_t metadata_count = 0;
};

struct GgufTensorInfo {
  std::string name;
  DataType dtype = DataType::unknown;
  std::uint32_t ggml_type = 0;
  std::vector<std::uint64_t> shape;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
};

struct GgufMetadataEntry {
  std::string key;
  std::string value;
};

class GgufFile {
public:
  GgufFile();
  GgufFile(GgufFile&&) noexcept;
  GgufFile& operator=(GgufFile&&) noexcept;
  GgufFile(const GgufFile&) = delete;
  GgufFile& operator=(const GgufFile&) = delete;
  ~GgufFile();

  static GgufFile open(std::string_view path);

  bool valid() const noexcept;
  std::uint32_t version() const noexcept;
  std::uint64_t tensor_count() const noexcept;
  std::uint64_t metadata_count() const noexcept;
  std::uint32_t alignment() const noexcept;
  std::span<const GgufTensorInfo> tensors() const noexcept;
  std::span<const GgufMetadataEntry> metadata() const noexcept;
  std::optional<std::string> metadata_value(std::string_view key) const;
  std::span<const std::string> tokenizer_tokens() const noexcept;
  const GgufTensorInfo* find_tensor(std::string_view name) const noexcept;
  std::span<const std::byte> tensor_bytes(const GgufTensorInfo& tensor) const;
  std::uint64_t prefetch_tensors(bool touch_pages = false) const;
  std::string summary() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

GgufProbe probe_gguf(std::string_view path);

enum class SpeculativeMode : std::uint8_t { off, mtp, draft_model };

struct MtpCapability {
  bool present = false;
  std::size_t head_count = 0;
  std::vector<std::string> tensor_names;
};

struct ModelMetadata {
  std::string name = "unknown";
  std::string architecture = "decoder-only";
  std::size_t parameters = 0;
  std::size_t context_length = 0;
  ModelFormat format = ModelFormat::manifest;
  GgufProbe gguf;
  std::uint32_t block_count = 0;
  std::uint32_t embedding_length = 0;
  std::uint32_t feed_forward_length = 0;
  std::uint32_t attention_heads = 0;
  std::uint32_t attention_kv_heads = 0;
  std::uint32_t vocab_size = 0;
  MtpCapability mtp;
};

class Model {
public:
  static Model load_manifest(std::string_view path);
  static Model load(std::string_view path);
  const ModelMetadata& metadata() const noexcept;
  const TensorStore& tensors() const noexcept;
  TensorStore& tensors() noexcept;
  const GgufFile* gguf_file() const noexcept;

private:
  ModelMetadata metadata_;
  TensorStore tensors_;
  std::unique_ptr<GgufFile> gguf_file_;
};

struct SpeculativeConfig {
  SpeculativeMode mode = SpeculativeMode::off;
  std::size_t draft_n_max = 0;
  std::string draft_model_path;
  bool strict = false;
};

struct SpeculativeRuntimeState {
  SpeculativeMode requested = SpeculativeMode::off;
  bool active = false;
  std::string fallback_reason;
};

using MtpConfig = SpeculativeConfig;

struct GenerationConfig {
  std::size_t max_tokens = 32;
  float temperature = 0.8f;
  std::size_t top_k = 40;
  float top_p = 0.95f;
  std::uint64_t seed = 0;
  SpeculativeConfig speculative;
};

struct SpeculativeStats {
  std::size_t target_tokens = 0;
  std::size_t verifier_steps = 0;
  std::size_t drafted_tokens = 0;
  std::size_t accepted_tokens = 0;
  std::size_t rejected_tokens = 0;
};

using MtpStats = SpeculativeStats;

using MtpDraftFn = std::function<std::vector<std::uint32_t>(std::span<const std::uint32_t> context, std::size_t draft_n_max)>;
using MtpVerifyFn = std::function<std::vector<std::uint32_t>(std::span<const std::uint32_t> context, std::span<const std::uint32_t> draft)>;
using MtpAcceptFn = std::function<bool(std::uint32_t token, bool drafted)>;

SpeculativeStats speculative_greedy_decode(std::vector<std::uint32_t>& context, std::size_t target_tokens,
                                            std::size_t draft_n_max, const MtpDraftFn& draft,
                                            const MtpVerifyFn& verify, const MtpAcceptFn& accept);

MtpStats mtp_greedy_decode(std::vector<std::uint32_t>& context, std::size_t target_tokens,
                           std::size_t draft_n_max, const MtpDraftFn& draft,
                           const MtpVerifyFn& verify, const MtpAcceptFn& accept);

struct TokenEvent {
  std::uint32_t id = 0;
  std::string text;
  std::size_t index = 0;
};

using TokenCallback = std::function<bool(const TokenEvent&)>;

class Sampler {
public:
  explicit Sampler(GenerationConfig config);
  std::uint32_t sample(std::span<const float> logits);

private:
  GenerationConfig config_;
  std::uint64_t state_;
};

class KVCache {
public:
  KVCache(std::size_t layers, std::size_t context, std::size_t heads, std::size_t head_dim);
  std::size_t bytes() const noexcept;
  std::size_t capacity_tokens() const noexcept;
  void reset() noexcept;
  bool append_slot() noexcept;

private:
  std::size_t layers_ = 0;
  std::size_t context_ = 0;
  std::size_t heads_ = 0;
  std::size_t head_dim_ = 0;
  std::size_t used_tokens_ = 0;
  std::vector<float> keys_;
  std::vector<float> values_;
};

class InferenceSession {
public:
  InferenceSession(const Model& model, GenerationConfig config = {});
  std::string generate(std::string_view prompt);
  void generate_stream(std::string_view prompt, const TokenCallback& callback);
  const KVCache& kv_cache() const noexcept;
  const SpeculativeRuntimeState& speculative_state() const noexcept;

private:
  const Model& model_;
  GenerationConfig config_;
  Tokenizer tokenizer_;
  KVCache kv_cache_;
  SpeculativeRuntimeState speculative_state_;
};

class Engine {
public:
  explicit Engine(Model model);
  std::string generate(std::string_view prompt, const GenerationConfig& config = {});
  void generate_stream(std::string_view prompt, const GenerationConfig& config, const TokenCallback& callback);
  const Model& model() const noexcept;

private:
  Model model_;
  Tokenizer tokenizer_;
};

class Graph {
public:
  std::uint32_t add_node(std::string op_name);
  std::size_t size() const noexcept;

private:
  std::vector<std::string> nodes_;
};

} // namespace cpullm
