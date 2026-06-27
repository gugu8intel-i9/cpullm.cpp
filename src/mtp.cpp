#include "cpullm/cpullm.hpp"

#include <algorithm>
#include <stdexcept>

namespace cpullm {

MtpStats mtp_greedy_decode(std::vector<std::uint32_t>& context, std::size_t target_tokens,
                           std::size_t draft_n_max, const MtpDraftFn& draft,
                           const MtpVerifyFn& verify, const MtpAcceptFn& accept) {
  if (target_tokens == 0) return {};
  if (draft_n_max == 0) throw std::invalid_argument("MTP draft_n_max must be greater than zero");
  if (!draft || !verify) throw std::invalid_argument("MTP requires draft and verify callbacks");

  MtpStats stats{};
  stats.target_tokens = target_tokens;

  while (stats.accepted_tokens + stats.rejected_tokens < target_tokens) {
    const std::size_t remaining = target_tokens - stats.accepted_tokens - stats.rejected_tokens;
    const std::size_t want = std::min(draft_n_max, remaining);
    auto proposed = draft(context, want);
    if (proposed.empty()) throw std::runtime_error("MTP draft callback returned no tokens");
    if (proposed.size() > want) proposed.resize(want);
    stats.drafted_tokens += proposed.size();

    auto verified = verify(context, proposed);
    ++stats.verifier_steps;
    if (verified.empty()) throw std::runtime_error("MTP verify callback returned no tokens");

    bool mismatch = false;
    const std::size_t compare = std::min(proposed.size(), verified.size());
    for (std::size_t i = 0; i < compare && stats.accepted_tokens + stats.rejected_tokens < target_tokens; ++i) {
      if (proposed[i] == verified[i]) {
        context.push_back(proposed[i]);
        ++stats.accepted_tokens;
        if (accept && !accept(proposed[i], true)) return stats;
      } else {
        context.push_back(verified[i]);
        ++stats.rejected_tokens;
        if (accept && !accept(verified[i], false)) return stats;
        mismatch = true;
        break;
      }
    }

    if (!mismatch && verified.size() > proposed.size() && stats.accepted_tokens + stats.rejected_tokens < target_tokens) {
      context.push_back(verified[proposed.size()]);
      ++stats.rejected_tokens;
      if (accept && !accept(verified[proposed.size()], false)) return stats;
    }
  }
  return stats;
}

} // namespace cpullm
