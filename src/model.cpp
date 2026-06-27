#include "cpullm/cpullm.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cpullm {

static std::string trim(std::string s) {
  const auto first = s.find_first_not_of(" \t\r\n\"");
  const auto last = s.find_last_not_of(" \t\r\n\",");
  if (first == std::string::npos) return {};
  return s.substr(first, last - first + 1);
}

static bool ends_with(std::string_view s, std::string_view suffix) {
  return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
}

Model Model::load_manifest(std::string_view path) {
  std::ifstream in(std::string{path});
  if (!in) throw std::runtime_error("failed to open model manifest");

  Model model;
  std::string line;
  while (std::getline(in, line)) {
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    const auto key = trim(line.substr(0, colon));
    const auto value = trim(line.substr(colon + 1));
    if (key == "name") model.metadata_.name = value;
    else if (key == "architecture") model.metadata_.architecture = value;
    else if (key == "parameters") model.metadata_.parameters = static_cast<std::size_t>(std::stoull(value));
    else if (key == "context_length") model.metadata_.context_length = static_cast<std::size_t>(std::stoull(value));
  }
  return model;
}

Model Model::load(std::string_view path) {
  if (ends_with(path, ".gguf")) {
    Model model;
    model.metadata_.format = ModelFormat::gguf_probe;
    model.gguf_file_ = std::make_unique<GgufFile>(GgufFile::open(path));
    model.metadata_.gguf = {.valid = model.gguf_file_->valid(),
                            .version = model.gguf_file_->version(),
                            .tensor_count = model.gguf_file_->tensor_count(),
                            .metadata_count = model.gguf_file_->metadata_count()};
    model.metadata_.name = model.gguf_file_->metadata_value("general.name").value_or(std::string{path});
    model.metadata_.architecture = model.gguf_file_->metadata_value("general.architecture").value_or("unknown");
    auto parse_u32 = [&](const char* key) -> std::uint32_t {
      auto v = model.gguf_file_->metadata_value(key);
      if (!v) return 0u;
      const auto first = v->find_first_of("0123456789");
      if (first == std::string::npos) return 0u;
      const auto last = v->find_first_not_of("0123456789", first);
      return static_cast<std::uint32_t>(std::stoul(v->substr(first, last - first)));
    };
    model.metadata_.block_count = parse_u32("lfm2.block_count");
    model.metadata_.context_length = parse_u32("lfm2.context_length");
    model.metadata_.embedding_length = parse_u32("lfm2.embedding_length");
    model.metadata_.feed_forward_length = parse_u32("lfm2.feed_forward_length");
    model.metadata_.attention_heads = parse_u32("lfm2.attention.head_count");
    model.metadata_.attention_kv_heads = parse_u32("lfm2.attention.head_count_kv");
    model.metadata_.vocab_size = parse_u32("lfm2.vocab_size");
    model.metadata_.parameters = static_cast<std::size_t>(model.gguf_file_->tensor_count());
    return model;
  }
  return load_manifest(path);
}

const ModelMetadata& Model::metadata() const noexcept { return metadata_; }
const TensorStore& Model::tensors() const noexcept { return tensors_; }
TensorStore& Model::tensors() noexcept { return tensors_; }
const GgufFile* Model::gguf_file() const noexcept { return gguf_file_.get(); }

} // namespace cpullm
