#include "cpullm/cpullm.hpp"

#include <cstdlib>
#include <new>
#include <utility>

namespace cpullm {

Arena::Arena(std::size_t bytes) : capacity_(bytes) {
#if defined(_MSC_VER)
  data_ = static_cast<std::byte*>(_aligned_malloc(bytes, 64));
  if (!data_) throw std::bad_alloc();
#else
  data_ = static_cast<std::byte*>(std::aligned_alloc(64, ((bytes + 63) / 64) * 64));
  if (!data_) throw std::bad_alloc();
#endif
}

Arena::Arena(Arena&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      capacity_(std::exchange(other.capacity_, 0)),
      offset_(std::exchange(other.offset_, 0)) {}

Arena& Arena::operator=(Arena&& other) noexcept {
  if (this != &other) {
#if defined(_MSC_VER)
    _aligned_free(data_);
#else
    std::free(data_);
#endif
    data_ = std::exchange(other.data_, nullptr);
    capacity_ = std::exchange(other.capacity_, 0);
    offset_ = std::exchange(other.offset_, 0);
  }
  return *this;
}

Arena::~Arena() {
#if defined(_MSC_VER)
  _aligned_free(data_);
#else
  std::free(data_);
#endif
}

void* Arena::allocate(std::size_t bytes, std::size_t alignment) {
  const std::size_t mask = alignment - 1;
  const std::size_t aligned = (offset_ + mask) & ~mask;
  if (aligned + bytes > capacity_) throw std::bad_alloc();
  offset_ = aligned + bytes;
  return data_ + aligned;
}

void Arena::reset() noexcept { offset_ = 0; }
std::size_t Arena::used() const noexcept { return offset_; }
std::size_t Arena::capacity() const noexcept { return capacity_; }

} // namespace cpullm
