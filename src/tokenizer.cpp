#include "cpullm/cpullm.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace cpullm {
namespace {

std::string byte_piece(unsigned char b) {
  // GPT-2 byte-unicode compatible for visible ASCII; private lightweight fallback for the rest.
  if (b >= 33 && b <= 126) return std::string(1, static_cast<char>(b));
  return std::string{"<0x"} + "0123456789ABCDEF"[b >> 4] + "0123456789ABCDEF"[b & 15] + ">";
}

std::pair<std::string, std::string> split_merge(std::string_view merge) {
  const auto sp = merge.find(' ');
  if (sp == std::string_view::npos) return {std::string(merge), {}};
  return {std::string(merge.substr(0, sp)), std::string(merge.substr(sp + 1))};
}

} // namespace

Tokenizer Tokenizer::from_gguf(const GgufFile& gguf) {
  auto toks = gguf.tokenizer_tokens();
  if (toks.empty()) throw std::runtime_error("GGUF tokenizer.ggml.tokens not found");
  auto merges = gguf.tokenizer_merges();
  auto model = gguf.metadata_value("tokenizer.ggml.model").value_or("unknown");
  return Tokenizer(std::vector<std::string>(toks.begin(), toks.end()),
                   std::vector<std::string>(merges.begin(), merges.end()), model);
}

Tokenizer::Tokenizer(std::vector<std::string> vocab) : vocab_(std::move(vocab)) {
  status_.model = "simple";
  if (vocab_.empty()) vocab_ = {"<unk>", "<bos>", "<eos>"};
  build_indices();
}

Tokenizer::Tokenizer(std::vector<std::string> vocab, std::vector<std::string> merges, std::string model)
    : vocab_(std::move(vocab)) {
  status_.model = std::move(model);
  std::uint32_t rank = 0;
  for (const auto& merge : merges) {
    auto [a, b] = split_merge(merge);
    if (!a.empty() && !b.empty()) merge_ranks_.emplace(a + '\n' + b, rank++);
  }
  status_.has_merges = !merge_ranks_.empty();
  status_.exact_bpe = status_.has_merges && (status_.model == "gpt2" || status_.model == "llama" || status_.model == "qwen2" || status_.model == "qwen3" || status_.model == "lfm2");
  build_indices();
}

void Tokenizer::build_indices() {
  token_to_id_.clear();
  token_to_id_.reserve(vocab_.size() * 2 + 1);
  trie_.clear();
  trie_.push_back({});
  for (std::uint32_t id = 0; id < vocab_.size(); ++id) {
    token_to_id_.emplace(vocab_[id], id);
    std::uint32_t node = 0;
    for (unsigned char ch : vocab_[id]) {
      auto it = trie_[node].next.find(ch);
      if (it == trie_[node].next.end()) {
        const std::uint32_t next = static_cast<std::uint32_t>(trie_.size());
        trie_[node].next.emplace(ch, next);
        trie_.push_back({});
        node = next;
      } else {
        node = it->second;
      }
    }
    trie_[node].token = id;
  }
  status_.vocab_size = vocab_.size();
  status_.merge_count = merge_ranks_.size();
}

std::vector<std::uint32_t> Tokenizer::encode_bpe(std::string_view text) const {
  std::vector<std::string> pieces;
  pieces.reserve(text.size());
  for (unsigned char b : text) pieces.push_back(byte_piece(b));

  while (pieces.size() > 1) {
    std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
    std::size_t best = std::numeric_limits<std::size_t>::max();
    for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
      auto it = merge_ranks_.find(pieces[i] + '\n' + pieces[i + 1]);
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best = i;
      }
    }
    if (best == std::numeric_limits<std::size_t>::max()) break;
    pieces[best] += pieces[best + 1];
    pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(best + 1));
  }

  std::vector<std::uint32_t> out;
  out.reserve(pieces.size());
  for (const auto& p : pieces) {
    auto it = token_to_id_.find(p);
    if (it != token_to_id_.end()) out.push_back(it->second);
    else {
      for (unsigned char b : p) {
        auto bit = token_to_id_.find(byte_piece(b));
        out.push_back(bit == token_to_id_.end() ? 0u : bit->second);
      }
    }
  }
  return out;
}

std::vector<std::uint32_t> Tokenizer::encode_trie(std::string_view text) const {
  std::vector<std::uint32_t> out;
  out.reserve(text.size());
  std::size_t i = 0;
  while (i < text.size()) {
    std::uint32_t node = 0;
    std::uint32_t best = UINT32_MAX;
    std::size_t best_pos = i;
    for (std::size_t j = i; j < text.size(); ++j) {
      auto it = trie_[node].next.find(static_cast<unsigned char>(text[j]));
      if (it == trie_[node].next.end()) break;
      node = it->second;
      if (trie_[node].token != UINT32_MAX) {
        best = trie_[node].token;
        best_pos = j + 1;
      }
    }
    if (best != UINT32_MAX) {
      out.push_back(best);
      i = best_pos;
    } else {
      auto bit = token_to_id_.find(byte_piece(static_cast<unsigned char>(text[i])));
      out.push_back(bit == token_to_id_.end() ? 0u : bit->second);
      ++i;
    }
  }
  return out;
}

std::vector<std::uint32_t> Tokenizer::encode(std::string_view text) const {
  if (!merge_ranks_.empty()) return encode_bpe(text);
  return encode_trie(text);
}

std::string Tokenizer::decode(std::span<const std::uint32_t> ids) const {
  std::string out;
  for (const auto id : ids) {
    if (id < vocab_.size()) out += vocab_[id];
    else out += "<unk>";
  }
  return out;
}

std::size_t Tokenizer::vocab_size() const noexcept { return vocab_.size(); }
const TokenizerStatus& Tokenizer::status() const noexcept { return status_; }

} // namespace cpullm
