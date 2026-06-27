#include "cpullm/cpullm.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cpullm {

static std::string trim(std::string s) {
  const auto first = s.find_first_not_of(" \t\r\n\"");
  const auto last = s.find_last_not_of(" \t\r\n\",");
  if (first == std::string::npos) return {};
  return s.substr(first, last - first + 1);
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

const ModelMetadata& Model::metadata() const noexcept { return metadata_; }

} // namespace cpullm
