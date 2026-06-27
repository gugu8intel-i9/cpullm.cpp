#include "cpullm/cpullm.hpp"

#include <cstring>

namespace cpullm {

void TensorStore::add(std::string name, DataType dtype, std::vector<std::size_t> shape, std::vector<std::byte> data) {
  entries_.push_back({std::move(name), dtype, std::move(shape), std::move(data)});
}

const TensorView* TensorStore::find(std::string_view name) const noexcept {
  for (const auto& e : entries_) {
    if (e.name == name) {
      static thread_local TensorView view;
      view.dtype = e.dtype;
      view.shape = e.shape;
      view.bytes = std::span<const std::byte>(e.data.data(), e.data.size());
      return &view;
    }
  }
  return nullptr;
}

std::size_t TensorStore::size() const noexcept { return entries_.size(); }

} // namespace cpullm
