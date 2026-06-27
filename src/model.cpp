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
    model.metadata_.name = std::string{path};
    model.metadata_.gguf = probe_gguf(path);
    if (!model.metadata_.gguf.valid) throw std::runtime_error("invalid or unsupported GGUF file");
    model.metadata_.parameters = static_cast<std::size_t>(model.metadata_.gguf.tensor_count);
    return model;
  }
  return load_manifest(path);
}

const ModelMetadata& Model::metadata() const noexcept { return metadata_; }
const TensorStore& Model::tensors() const noexcept { return tensors_; }
TensorStore& Model::tensors() noexcept { return tensors_; }

} // namespace cpullm
