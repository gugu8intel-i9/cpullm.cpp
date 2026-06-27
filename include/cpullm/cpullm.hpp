#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cpullm {

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

enum class DataType : std::uint8_t { f32, q4_0 };

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

class Tokenizer {
public:
  explicit Tokenizer(std::vector<std::string> vocab = {});
  std::vector<std::uint32_t> encode(std::string_view text) const;
  std::string decode(std::span<const std::uint32_t> ids) const;
  std::size_t vocab_size() const noexcept;

private:
  std::vector<std::string> vocab_;
};

struct ModelMetadata {
  std::string name = "unknown";
  std::string architecture = "decoder-only";
  std::size_t parameters = 0;
  std::size_t context_length = 0;
};

class Model {
public:
  static Model load_manifest(std::string_view path);
  const ModelMetadata& metadata() const noexcept;

private:
  ModelMetadata metadata_;
};

struct GenerationConfig {
  std::size_t max_tokens = 32;
  float temperature = 0.8f;
};

class Engine {
public:
  explicit Engine(Model model);
  std::string generate(std::string_view prompt, const GenerationConfig& config = {});
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
