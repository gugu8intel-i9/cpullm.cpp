#include "cpullm/plan.hpp"

#include <algorithm>
#include <sstream>
#include <string_view>
#include <unordered_set>

namespace cpullm {
namespace {

void issue(ExecutionPlan& plan, std::string code, std::string message) {
  plan.issues.push_back({std::move(code), std::move(message)});
}

bool has_tensor(const GgufFile& gguf, std::string_view name) {
  return gguf.find_tensor(name) != nullptr;
}

std::string esc(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    if (c == '\\' || c == '"') { out.push_back('\\'); out.push_back(c); }
    else if (c == '\n') out += "\\n";
    else out.push_back(c);
  }
  return out;
}

} // namespace

ExecutionPlan build_execution_plan(const Model& model) {
  ExecutionPlan plan;
  const auto& md = model.metadata();
  plan.architecture = md.architecture;
  plan.layers = md.block_count;
  plan.required_ops = {"token_embedding", "rms_norm", "rope", "causal_attention", "short_convolution", "swiglu_mlp", "logits_projection", "greedy_decode"};

  const auto* gguf = model.gguf_file();
  if (!gguf) {
    issue(plan, "model.not_gguf", "production execution requires a GGUF model");
    return plan;
  }

  plan.tensors = gguf->tensor_count();
  plan.tokenizer_tokens = gguf->tokenizer_tokens().size();
  if (plan.tokenizer_tokens == 0) issue(plan, "tokenizer.missing", "GGUF tokenizer.ggml.tokens is missing or empty");

  for (const auto& t : gguf->tensors()) {
    switch (t.dtype) {
      case DataType::f32: plan.kernels.f32 = true; break;
      case DataType::q4_0: plan.kernels.q4_0 = true; break;
      case DataType::q4_1: plan.kernels.q4_1 = true; break;
      case DataType::q8_0: plan.kernels.q8_0 = true; break;
      default: plan.kernels.unknown_types = true; break;
    }
  }
  if (plan.kernels.q4_1) issue(plan, "kernel.q4_1.missing", "GGUF contains Q4_1 tensors but cpullm has no Q4_1 matvec/dequant kernel yet");
  if (plan.kernels.q8_0) issue(plan, "kernel.q8_0.missing", "GGUF contains Q8_0 tensors but cpullm has no Q8_0 matvec/dequant kernel yet");
  if (plan.kernels.unknown_types) issue(plan, "kernel.unknown_type", "GGUF contains tensor types not mapped to cpullm kernels yet");

  if (md.architecture == "lfm2") {
    if (md.block_count == 0) issue(plan, "lfm2.block_count", "lfm2.block_count is missing");
    if (md.embedding_length == 0) issue(plan, "lfm2.embedding_length", "lfm2.embedding_length is missing");
    if (md.vocab_size == 0) issue(plan, "lfm2.vocab_size", "lfm2.vocab_size is missing");

    const std::vector<std::string> global = {"token_embd.weight", "token_embd_norm.weight"};
    for (const auto& name : global) if (!has_tensor(*gguf, name)) issue(plan, "tensor.missing", "required tensor missing: " + name);

    const std::size_t layers = md.block_count;
    for (std::size_t i = 0; i < layers; ++i) {
      const std::string p = "blk." + std::to_string(i) + ".";
      const std::vector<std::string> required = {
        p + "attn_norm.weight",
        p + "ffn_norm.weight",
        p + "ffn_gate.weight",
        p + "ffn_up.weight",
        p + "ffn_down.weight",
        p + "shortconv.conv.weight",
        p + "shortconv.in_proj.weight",
        p + "shortconv.out_proj.weight",
      };
      for (const auto& name : required) if (!has_tensor(*gguf, name)) issue(plan, "tensor.missing", "required tensor missing: " + name);
    }

    issue(plan, "executor.lfm2.graph_unwired", "LFM2 graph scheduling is not wired yet: exact short-convolution/attention/residual ordering must be implemented before real decode");
    issue(plan, "tokenizer.bpe_unwired", "GGUF GPT-2/LFM2 BPE merge/rank handling is not wired yet; token array loading alone is insufficient for parity");
  } else if (md.architecture == "cpullm_tiny") {
    issue(plan, "executor.cpullm_tiny.missing", "cpullm_tiny executor tensor contract is reserved but no complete tensor set was found");
  } else {
    issue(plan, "architecture.unsupported", "unsupported architecture: " + md.architecture);
  }

  plan.status = plan.issues.empty() ? PlanStatus::runnable : PlanStatus::blocked;
  return plan;
}

std::string ExecutionPlan::to_text() const {
  std::ostringstream out;
  out << "execution_plan_status=" << (runnable() ? "runnable" : "blocked") << '\n';
  out << "architecture=" << architecture << '\n';
  out << "layers=" << layers << '\n';
  out << "tensors=" << tensors << '\n';
  out << "tokenizer_tokens=" << tokenizer_tokens << '\n';
  out << "kernels=f32:" << (kernels.f32 ? "yes" : "no")
      << ",q4_0:" << (kernels.q4_0 ? "yes" : "no")
      << ",q4_1:" << (kernels.q4_1 ? "yes" : "no")
      << ",q8_0:" << (kernels.q8_0 ? "yes" : "no")
      << ",unknown:" << (kernels.unknown_types ? "yes" : "no") << '\n';
  out << "required_ops=";
  for (std::size_t i = 0; i < required_ops.size(); ++i) out << (i ? "," : "") << required_ops[i];
  out << '\n';
  for (const auto& it : issues) out << "issue " << it.code << ": " << it.message << '\n';
  return out.str();
}

std::string ExecutionPlan::to_json() const {
  std::ostringstream out;
  out << "{\n";
  out << "  \"status\": \"" << (runnable() ? "runnable" : "blocked") << "\",\n";
  out << "  \"architecture\": \"" << esc(architecture) << "\",\n";
  out << "  \"layers\": " << layers << ",\n";
  out << "  \"tensors\": " << tensors << ",\n";
  out << "  \"tokenizer_tokens\": " << tokenizer_tokens << ",\n";
  out << "  \"kernels\": {\"f32\": " << (kernels.f32 ? "true" : "false")
      << ", \"q4_0\": " << (kernels.q4_0 ? "true" : "false")
      << ", \"q4_1\": " << (kernels.q4_1 ? "true" : "false")
      << ", \"q8_0\": " << (kernels.q8_0 ? "true" : "false")
      << ", \"unknown\": " << (kernels.unknown_types ? "true" : "false") << "},\n";
  out << "  \"issues\": [";
  for (std::size_t i = 0; i < issues.size(); ++i) {
    if (i) out << ",";
    out << "{\"code\": \"" << esc(issues[i].code) << "\", \"message\": \"" << esc(issues[i].message) << "\"}";
  }
  out << "]\n}";
  return out.str();
}

} // namespace cpullm
