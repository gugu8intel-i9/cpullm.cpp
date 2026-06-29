#include "cpullm/cpullm.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

#if defined(_WIN32)
  #define NOMINMAX
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

namespace cpullm {
namespace {

enum class MetaType : std::uint32_t {
  u8 = 0, i8 = 1, u16 = 2, i16 = 3, u32 = 4, i32 = 5, f32 = 6,
  boolean = 7, string = 8, array = 9, u64 = 10, i64 = 11, f64 = 12,
};

struct Cursor {
  const std::byte* base = nullptr;
  std::size_t size = 0;
  std::size_t pos = 0;

  void require(std::size_t n) const {
    if (n > size || pos > size - n) throw std::runtime_error("truncated GGUF file");
  }

  template <class T>
  T read() {
    static_assert(std::is_trivially_copyable_v<T>);
    require(sizeof(T));
    T out{};
    std::memcpy(&out, base + pos, sizeof(T));
    pos += sizeof(T);
    return out;
  }

  std::string read_string() {
    const auto len = read<std::uint64_t>();
    if (len > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      throw std::runtime_error("GGUF string too large");
    }
    require(static_cast<std::size_t>(len));
    std::string out(reinterpret_cast<const char*>(base + pos), static_cast<std::size_t>(len));
    pos += static_cast<std::size_t>(len);
    return out;
  }
};

std::string scalar_to_string(Cursor& c, MetaType t);

std::vector<std::string> read_string_array(Cursor& c, std::uint64_t count) {
  std::vector<std::string> out;
  out.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(count, 1000000)));
  for (std::uint64_t i = 0; i < count; ++i) out.push_back(c.read_string());
  return out;
}

std::string array_to_string(Cursor& c) {
  const auto elem_type = static_cast<MetaType>(c.read<std::uint32_t>());
  const auto count = c.read<std::uint64_t>();
  std::ostringstream out;
  out << '[';
  const std::uint64_t show = std::min<std::uint64_t>(count, 16);
  for (std::uint64_t i = 0; i < count; ++i) {
    auto v = scalar_to_string(c, elem_type);
    if (i < show) {
      if (i) out << ", ";
      out << v;
    }
  }
  if (count > show) out << ", ... (" << count << " items)";
  out << ']';
  return out.str();
}

std::string scalar_to_string(Cursor& c, MetaType t) {
  std::ostringstream out;
  switch (t) {
    case MetaType::u8: out << static_cast<unsigned>(c.read<std::uint8_t>()); break;
    case MetaType::i8: out << static_cast<int>(c.read<std::int8_t>()); break;
    case MetaType::u16: out << c.read<std::uint16_t>(); break;
    case MetaType::i16: out << c.read<std::int16_t>(); break;
    case MetaType::u32: out << c.read<std::uint32_t>(); break;
    case MetaType::i32: out << c.read<std::int32_t>(); break;
    case MetaType::f32: out << c.read<float>(); break;
    case MetaType::boolean: out << (c.read<std::uint8_t>() ? "true" : "false"); break;
    case MetaType::string: return c.read_string();
    case MetaType::array: return array_to_string(c);
    case MetaType::u64: out << c.read<std::uint64_t>(); break;
    case MetaType::i64: out << c.read<std::int64_t>(); break;
    case MetaType::f64: out << c.read<double>(); break;
    default: throw std::runtime_error("unknown GGUF metadata type");
  }
  return out.str();
}

DataType ggml_to_dtype(std::uint32_t type) {
  switch (type) {
    case 0: return DataType::f32;
    case 1: return DataType::f16;
    case 2: return DataType::q4_0;
    case 3: return DataType::q4_1;
    case 8: return DataType::q8_0;
    default: return DataType::unknown;
  }
}

std::uint64_t align_up(std::uint64_t x, std::uint64_t a) {
  return a == 0 ? x : ((x + a - 1) / a) * a;
}

std::uint64_t element_count(const std::vector<std::uint64_t>& shape) {
  std::uint64_t n = 1;
  for (auto d : shape) n *= d;
  return shape.empty() ? 0 : n;
}

std::uint64_t tensor_nbytes(DataType dtype, const std::vector<std::uint64_t>& shape) {
  const auto n = element_count(shape);
  switch (dtype) {
    case DataType::f32: return n * 4;
    case DataType::f16: return n * 2;
    case DataType::q4_0: return ((n + 31) / 32) * 18;
    case DataType::q4_1: return ((n + 31) / 32) * (sizeof(float) * 2 + 16);
    case DataType::q8_0: return ((n + 31) / 32) * (sizeof(float) + 32);
    case DataType::unknown: return 0;
  }
  return 0;
}

} // namespace

struct GgufFile::Impl {
#if defined(_WIN32)
  HANDLE file = INVALID_HANDLE_VALUE;
  HANDLE mapping = nullptr;
#else
  int fd = -1;
#endif
  const std::byte* data = nullptr;
  std::size_t size = 0;
  bool valid = false;
  std::uint32_t version = 0;
  std::uint64_t tensor_count = 0;
  std::uint64_t metadata_count = 0;
  std::uint32_t alignment = 32;
  std::uint64_t data_start = 0;
  std::vector<GgufTensorInfo> tensors;
  std::vector<GgufMetadataEntry> metadata;
  std::vector<std::string> tokenizer_tokens;

  ~Impl() {
#if defined(_WIN32)
    if (data) UnmapViewOfFile(data);
    if (mapping) CloseHandle(mapping);
    if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
#else
    if (data) munmap(const_cast<std::byte*>(data), size);
    if (fd >= 0) close(fd);
#endif
  }
};

GgufFile::GgufFile() = default;
GgufFile::GgufFile(GgufFile&&) noexcept = default;
GgufFile& GgufFile::operator=(GgufFile&&) noexcept = default;
GgufFile::~GgufFile() = default;

GgufFile GgufFile::open(std::string_view path_sv) {
  const std::string path{path_sv};
  auto impl = std::make_unique<Impl>();
#if defined(_WIN32)
  impl->file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (impl->file == INVALID_HANDLE_VALUE) throw std::runtime_error("failed to open GGUF file");
  LARGE_INTEGER sz{};
  if (!GetFileSizeEx(impl->file, &sz)) throw std::runtime_error("failed to stat GGUF file");
  impl->size = static_cast<std::size_t>(sz.QuadPart);
  impl->mapping = CreateFileMappingA(impl->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!impl->mapping) throw std::runtime_error("failed to map GGUF file");
  impl->data = static_cast<const std::byte*>(MapViewOfFile(impl->mapping, FILE_MAP_READ, 0, 0, 0));
  if (!impl->data) throw std::runtime_error("failed to view GGUF file");
#else
  impl->fd = ::open(path.c_str(), O_RDONLY);
  if (impl->fd < 0) throw std::runtime_error("failed to open GGUF file");
  struct stat st{};
  if (fstat(impl->fd, &st) != 0) throw std::runtime_error("failed to stat GGUF file");
  impl->size = static_cast<std::size_t>(st.st_size);
  void* mapped = mmap(nullptr, impl->size, PROT_READ, MAP_PRIVATE, impl->fd, 0);
  if (mapped == MAP_FAILED) throw std::runtime_error("failed to mmap GGUF file");
  impl->data = static_cast<const std::byte*>(mapped);
#endif

  Cursor c{impl->data, impl->size, 0};
  char magic[4]{};
  for (char& ch : magic) ch = static_cast<char>(c.read<std::uint8_t>());
  if (std::string_view(magic, 4) != "GGUF") throw std::runtime_error("invalid GGUF magic");
  impl->version = c.read<std::uint32_t>();
  impl->tensor_count = c.read<std::uint64_t>();
  impl->metadata_count = c.read<std::uint64_t>();

  impl->metadata.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(impl->metadata_count, 1000000)));
  for (std::uint64_t i = 0; i < impl->metadata_count; ++i) {
    auto key = c.read_string();
    const auto type = static_cast<MetaType>(c.read<std::uint32_t>());
    std::string value;
    if (key == "tokenizer.ggml.tokens" && type == MetaType::array) {
      const auto elem_type = static_cast<MetaType>(c.read<std::uint32_t>());
      const auto count = c.read<std::uint64_t>();
      if (elem_type != MetaType::string) throw std::runtime_error("tokenizer.ggml.tokens is not a string array");
      impl->tokenizer_tokens = read_string_array(c, count);
      std::ostringstream v;
      v << "[" << impl->tokenizer_tokens.size() << " tokenizer tokens]";
      value = v.str();
    } else {
      value = scalar_to_string(c, type);
    }
    if (key == "general.alignment") impl->alignment = static_cast<std::uint32_t>(std::stoul(value));
    impl->metadata.push_back({std::move(key), std::move(value)});
  }

  impl->tensors.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(impl->tensor_count, 1000000)));
  for (std::uint64_t i = 0; i < impl->tensor_count; ++i) {
    GgufTensorInfo t;
    t.name = c.read_string();
    const auto nd = c.read<std::uint32_t>();
    t.shape.reserve(nd);
    for (std::uint32_t d = 0; d < nd; ++d) t.shape.push_back(c.read<std::uint64_t>());
    t.ggml_type = c.read<std::uint32_t>();
    t.dtype = ggml_to_dtype(t.ggml_type);
    t.offset = c.read<std::uint64_t>();
    t.bytes = tensor_nbytes(t.dtype, t.shape);
    impl->tensors.push_back(std::move(t));
  }

  impl->data_start = align_up(c.pos, impl->alignment);
  for (auto& t : impl->tensors) {
    if (t.bytes == 0) continue;
    const auto abs = impl->data_start + t.offset;
    if (abs > impl->size || t.bytes > impl->size - abs) {
      throw std::runtime_error("GGUF tensor points outside file");
    }
  }
  impl->valid = true;
  GgufFile out;
  out.impl_ = std::move(impl);
  return out;
}

bool GgufFile::valid() const noexcept { return impl_ && impl_->valid; }
std::uint32_t GgufFile::version() const noexcept { return impl_ ? impl_->version : 0; }
std::uint64_t GgufFile::tensor_count() const noexcept { return impl_ ? impl_->tensor_count : 0; }
std::uint64_t GgufFile::metadata_count() const noexcept { return impl_ ? impl_->metadata_count : 0; }
std::uint32_t GgufFile::alignment() const noexcept { return impl_ ? impl_->alignment : 0; }
std::span<const GgufTensorInfo> GgufFile::tensors() const noexcept { return impl_ ? std::span<const GgufTensorInfo>(impl_->tensors) : std::span<const GgufTensorInfo>(); }
std::span<const GgufMetadataEntry> GgufFile::metadata() const noexcept { return impl_ ? std::span<const GgufMetadataEntry>(impl_->metadata) : std::span<const GgufMetadataEntry>(); }

std::optional<std::string> GgufFile::metadata_value(std::string_view key) const {
  if (!impl_) return std::nullopt;
  for (const auto& e : impl_->metadata) if (e.key == key) return e.value;
  return std::nullopt;
}

std::span<const std::string> GgufFile::tokenizer_tokens() const noexcept {
  return impl_ ? std::span<const std::string>(impl_->tokenizer_tokens) : std::span<const std::string>();
}

const GgufTensorInfo* GgufFile::find_tensor(std::string_view name) const noexcept {
  if (!impl_) return nullptr;
  for (const auto& t : impl_->tensors) if (t.name == name) return &t;
  return nullptr;
}

std::span<const std::byte> GgufFile::tensor_bytes(const GgufTensorInfo& tensor) const {
  if (!impl_) throw std::runtime_error("GGUF file is not open");
  const auto abs = impl_->data_start + tensor.offset;
  if (abs > impl_->size || tensor.bytes > impl_->size - abs) throw std::runtime_error("GGUF tensor out of bounds");
  return {impl_->data + abs, static_cast<std::size_t>(tensor.bytes)};
}


std::uint64_t GgufFile::prefetch_tensors(bool touch_pages) const {
  if (!impl_) throw std::runtime_error("GGUF file is not open");
  std::uint64_t total = 0;
  constexpr std::size_t page = 4096;
  volatile std::uint8_t sink = 0;
  for (const auto& t : impl_->tensors) {
    if (t.bytes == 0) continue;
    const auto abs = impl_->data_start + t.offset;
    if (abs > impl_->size || t.bytes > impl_->size - abs) throw std::runtime_error("GGUF tensor out of bounds");
    const auto* ptr = impl_->data + abs;
#if !defined(_WIN32)
    (void)madvise(const_cast<std::byte*>(ptr), static_cast<std::size_t>(t.bytes), MADV_WILLNEED);
#endif
    if (touch_pages) {
      for (std::uint64_t off = 0; off < t.bytes; off += page) {
        sink ^= static_cast<std::uint8_t>(ptr[off]);
      }
      sink ^= static_cast<std::uint8_t>(ptr[t.bytes - 1]);
    }
    total += t.bytes;
  }
  (void)sink;
  return total;
}

std::string GgufFile::summary() const {
  std::ostringstream out;
  out << "GGUF v" << version() << " tensors=" << tensor_count() << " metadata=" << metadata_count()
      << " alignment=" << alignment();
  if (auto arch = metadata_value("general.architecture")) out << " arch=" << *arch;
  if (auto name = metadata_value("general.name")) out << " name=\"" << *name << '"';
  return out.str();
}

GgufProbe probe_gguf(std::string_view path) {
  try {
    auto file = GgufFile::open(path);
    return {.valid = file.valid(), .version = file.version(), .tensor_count = file.tensor_count(), .metadata_count = file.metadata_count()};
  } catch (...) {
    return {};
  }
}

std::size_t dtype_size(DataType dtype) {
  switch (dtype) {
    case DataType::f32: return sizeof(float);
    case DataType::f16: return 2;
    case DataType::q4_0: return 18;
    case DataType::q4_1: return sizeof(float) * 2 + 16;
    case DataType::q8_0: return sizeof(float) + 32;
    case DataType::unknown: return 0;
  }
  return 0;
}

} // namespace cpullm
