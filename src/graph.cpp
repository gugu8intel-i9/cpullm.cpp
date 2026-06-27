#include "cpullm/cpullm.hpp"

namespace cpullm {

std::uint32_t Graph::add_node(std::string op_name) {
  nodes_.push_back(std::move(op_name));
  return static_cast<std::uint32_t>(nodes_.size() - 1);
}

std::size_t Graph::size() const noexcept { return nodes_.size(); }

} // namespace cpullm
