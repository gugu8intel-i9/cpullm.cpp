#include "cpullm/cpullm.hpp"

#include <cassert>
#include <iostream>

int main() {
  cpullm::Tokenizer trie({"<unk>", "a", "b", "ab", "abc", " ", "c"});
  auto ids = trie.encode("abc ab");
  assert(ids.size() == 3);
  assert(trie.decode(ids) == "abc ab");
  assert(trie.status().vocab_size == 7);

  cpullm::Tokenizer bpe({"<unk>", "a", "b", "c", "ab", "abc"}, {"a b", "ab c"}, "gpt2");
  auto bid = bpe.encode("abc");
  assert(bid.size() == 1);
  assert(bpe.decode(bid) == "abc");
  assert(bpe.status().has_merges);
  assert(bpe.status().exact_bpe);

  std::cout << "cpullm tokenizer tests passed\n";
}
