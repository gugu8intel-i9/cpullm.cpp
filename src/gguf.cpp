#include "cpullm/cpullm.hpp"

#include <fstream>

namespace cpullm {

GgufProbe probe_gguf(std::string_view path) {
  GgufProbe probe{};
  std::ifstream in(std::string{path}, std::ios::binary);
  if (!in) return probe;
  char magic[4]{};
  in.read(magic, 4);
  if (!in || magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') return probe;
  probe.valid = true;
  in.read(reinterpret_cast<char*>(&probe.version), sizeof(probe.version));
  in.read(reinterpret_cast<char*>(&probe.tensor_count), sizeof(probe.tensor_count));
  in.read(reinterpret_cast<char*>(&probe.metadata_count), sizeof(probe.metadata_count));
  if (!in) probe.valid = false;
  return probe;
}

} // namespace cpullm
