#include "cpullm/cpullm.hpp"

#include <sstream>
#include <unordered_map>

namespace cpullm {

Tokenizer::Tokenizer(std::vector<std::string> vocab) : vocab_(std::move(vocab)) {
  if (vocab_.empty()) vocab_ = {"<unk>", "<bos>", "<eos>"};
}

std::vector<std::uint32_t> Tokenizer::encode(std::string_view text) const {
  std::unordered_map<std::string_view, std::uint32_t> ids;
  for (std::uint32_t i = 0; i < vocab_.size(); ++i) ids.emplace(vocab_[i], i);

  std::vector<std::uint32_t> out;
  std::istringstream in(std::string{text});
  std::string piece;
  while (in >> piece) {
    auto it = ids.find(piece);
    out.push_back(it == ids.end() ? 0u : it->second);
  }
  return out;
}

std::string Tokenizer::decode(std::span<const std::uint32_t> ids) const {
  std::string out;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i) out += ' ';
    const auto id = ids[i];
    out += id < vocab_.size() ? vocab_[id] : "<unk>";
  }
  return out;
}

std::size_t Tokenizer::vocab_size() const noexcept { return vocab_.size(); }

} // namespace cpullm
