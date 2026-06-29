#include "cpullm/plan.hpp"

#include <algorithm>
#include <array>
#include <sstream>
#include <string_view>

namespace cpullm {
namespace {

using std::string;

const std::vector<std::string>& decoder_ops() {
  static const std::vector<std::string> ops = {
      "token_embedding", "rms_norm", "rope", "causal_attention", "swiglu_mlp", "logits_projection", "greedy_decode"};
  return ops;
}

const std::vector<std::string>& hybrid_ops() {
  static const std::vector<std::string> ops = {
      "token_embedding", "rms_norm", "rope", "causal_attention", "short_convolution", "swiglu_mlp", "logits_projection", "greedy_decode"};
  return ops;
}

const std::vector<std::string>& moe_ops() {
  static const std::vector<std::string> ops = {
      "token_embedding", "rms_norm", "rope", "causal_attention", "moe_router", "expert_mlp", "logits_projection", "greedy_decode"};
  return ops;
}

const std::vector<ArchitectureProfile>& profiles_vec() {
  static const std::vector<ArchitectureProfile> profiles = {
      {"llama", "llama", true, false, false, true, true, decoder_ops(), {"Covers LLaMA, LLaMA 2/3-style GGUF tensor layouts."}},
      {"qwen2", "qwen", true, false, false, true, true, decoder_ops(), {"Qwen2/Qwen2.5 dense decoder family."}},
      {"qwen3", "qwen", true, false, false, true, true, decoder_ops(), {"Qwen3 dense decoder family; MTP variants need draft-head wiring."}},
      {"qwen3moe", "qwen", true, false, true, true, true, moe_ops(), {"Qwen MoE requires router and expert scheduling."}},
      {"mistral", "mistral", true, false, false, true, true, decoder_ops(), {"Mistral-style sliding-window attention metadata may be present."}},
      {"mixtral", "mistral", true, false, true, true, true, moe_ops(), {"Mixtral requires MoE router/expert kernels."}},
      {"gemma", "gemma", true, false, false, true, true, decoder_ops(), {"Gemma decoder family."}},
      {"gemma2", "gemma", true, false, false, true, true, decoder_ops(), {"Gemma 2/3 may require soft-capping and architecture-specific norms."}},
      {"gemma3", "gemma", true, false, false, true, true, decoder_ops(), {"Gemma 3 variants may include multimodal-adjacent metadata; text decoder remains transformer-like."}},
      {"phi2", "phi", true, false, false, true, true, decoder_ops(), {"Phi decoder family."}},
      {"phi3", "phi", true, false, false, true, true, decoder_ops(), {"Phi-3/3.5 decoder family."}},
      {"deepseek2", "deepseek", true, false, true, true, true, moe_ops(), {"DeepSeek V2/V3-style models require MLA/MoE-specific execution."}},
      {"granite", "granite", true, false, false, true, true, decoder_ops(), {"IBM Granite dense decoder family."}},
      {"gptneox", "gpt-neox", true, false, false, true, true, decoder_ops(), {"GPT-NeoX-style tensor layouts."}},
      {"falcon", "falcon", true, false, false, true, false, decoder_ops(), {"Falcon may use parallel attention/MLP and non-LLaMA normalization details."}},
      {"baichuan", "baichuan", true, false, false, true, true, decoder_ops(), {"Baichuan decoder family."}},
      {"mpt", "mpt", true, false, false, false, false, decoder_ops(), {"MPT uses ALiBi-style attention rather than RoPE in many checkpoints."}},
      {"starcoder", "starcoder", true, false, false, false, false, decoder_ops(), {"StarCoder/GPTBigCode-style decoder family."}},
      {"lfm2", "lfm", true, true, false, true, true, hybrid_ops(), {"LFM2 hybrid decoder requires short-convolution block scheduling."}},
      {"cpullm_tiny", "cpullm", true, false, false, true, true, decoder_ops(), {"Internal tiny executor contract for tests and bring-up."}},
  };
  return profiles;
}

void issue(ExecutionPlan& plan, std::string code, std::string message) {
  plan.issues.push_back({std::move(code), std::move(message)});
}

bool has_tensor(const GgufFile& gguf, std::string_view name) {
  return gguf.find_tensor(name) != nullptr;
}

bool has_any_tensor(const GgufFile& gguf, std::span<const std::string_view> names) {
  for (auto name : names) if (has_tensor(gguf, name)) return true;
  return false;
}

bool has_prefix_contains(const GgufFile& gguf, std::string_view prefix, std::string_view needle) {
  for (const auto& t : gguf.tensors()) {
    if (t.name.rfind(prefix, 0) == 0 && t.name.find(needle) != std::string::npos) return true;
  }
  return false;
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

void analyze_kernels(ExecutionPlan& plan, const GgufFile& gguf) {
  for (const auto& t : gguf.tensors()) {
    switch (t.dtype) {
      case DataType::f32: plan.kernels.f32 = true; break;
      case DataType::f16: plan.kernels.f16 = true; break;
      case DataType::q4_0: plan.kernels.q4_0 = true; break;
      case DataType::q4_1: plan.kernels.q4_1 = true; break;
      case DataType::q8_0: plan.kernels.q8_0 = true; break;
      default: plan.kernels.unknown_types = true; break;
    }
  }
  if (plan.kernels.f16) issue(plan, "kernel.f16.matmul_missing", "GGUF contains F16 tensors but cpullm has no F16 matvec/matmul kernel yet");
  if (plan.kernels.q4_1) issue(plan, "kernel.q4_1.missing", "GGUF contains Q4_1 tensors but cpullm has no Q4_1 matvec/dequant kernel yet");
  if (plan.kernels.q8_0) issue(plan, "kernel.q8_0.missing", "GGUF contains Q8_0 tensors but cpullm has no Q8_0 matvec/dequant kernel yet");
  if (plan.kernels.unknown_types) issue(plan, "kernel.unknown_type", "GGUF contains tensor types not mapped to cpullm kernels yet");
}

void validate_generic_decoder(ExecutionPlan& plan, const ModelMetadata& md, const GgufFile& gguf) {
  if (md.block_count == 0) issue(plan, "metadata.block_count", "block/layer count metadata is missing");
  if (md.embedding_length == 0) issue(plan, "metadata.embedding_length", "embedding length metadata is missing");
  if (md.vocab_size == 0) issue(plan, "metadata.vocab_size", "vocab size metadata is missing");

  static constexpr std::string_view global_a[] = {"token_embd.weight"};
  if (!has_any_tensor(gguf, global_a)) issue(plan, "tensor.token_embedding.missing", "token embedding tensor is missing");

  static constexpr std::string_view norm_names[] = {"output_norm.weight", "token_embd_norm.weight"};
  if (!has_any_tensor(gguf, norm_names)) issue(plan, "tensor.final_norm.missing", "final/output norm tensor is missing");

  const std::size_t layers = md.block_count;
  for (std::size_t i = 0; i < layers; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    if (!has_tensor(gguf, p + "attn_norm.weight")) issue(plan, "tensor.attn_norm.missing", "missing attention norm tensor in " + p);
    if (!has_tensor(gguf, p + "ffn_norm.weight")) issue(plan, "tensor.ffn_norm.missing", "missing FFN norm tensor in " + p);

    const bool split_qkv = has_tensor(gguf, p + "attn_q.weight") && has_tensor(gguf, p + "attn_k.weight") && has_tensor(gguf, p + "attn_v.weight");
    const bool fused_qkv = has_tensor(gguf, p + "attn_qkv.weight") || has_prefix_contains(gguf, p, "attn_qkv");
    if (!split_qkv && !fused_qkv) issue(plan, "tensor.attention_qkv.missing", "missing split or fused QKV tensors in " + p);
    if (!has_tensor(gguf, p + "attn_output.weight")) issue(plan, "tensor.attention_output.missing", "missing attention output tensor in " + p);

    const bool swiglu = has_tensor(gguf, p + "ffn_gate.weight") && has_tensor(gguf, p + "ffn_up.weight") && has_tensor(gguf, p + "ffn_down.weight");
    const bool dense_ffn = has_tensor(gguf, p + "ffn_up.weight") && has_tensor(gguf, p + "ffn_down.weight");
    const bool moe = has_prefix_contains(gguf, p, "ffn_gate_inp") || has_prefix_contains(gguf, p, "ffn_gate_exps") || has_prefix_contains(gguf, p, "expert");
    if (!swiglu && !dense_ffn && !moe) issue(plan, "tensor.ffn.missing", "missing FFN/MLP tensors in " + p);
  }
}

void validate_hybrid_lfm(ExecutionPlan& plan, const ModelMetadata& md, const GgufFile& gguf) {
  validate_generic_decoder(plan, md, gguf);
  for (std::size_t i = 0; i < md.block_count; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    const std::vector<std::string> required = {p + "shortconv.conv.weight", p + "shortconv.in_proj.weight", p + "shortconv.out_proj.weight"};
    for (const auto& name : required) if (!has_tensor(gguf, name)) issue(plan, "tensor.shortconv.missing", "required hybrid tensor missing: " + name);
  }
}

} // namespace

std::span<const ArchitectureProfile> architecture_profiles() noexcept { return profiles_vec(); }

const ArchitectureProfile* find_architecture_profile(std::string_view architecture) noexcept {
  const auto& profiles = profiles_vec();
  auto it = std::find_if(profiles.begin(), profiles.end(), [&](const auto& p) { return p.name == architecture; });
  return it == profiles.end() ? nullptr : &*it;
}

ExecutionPlan build_execution_plan(const Model& model) {
  ExecutionPlan plan;
  const auto& md = model.metadata();
  plan.architecture = md.architecture;
  plan.layers = md.block_count;

  if (const auto* profile = find_architecture_profile(md.architecture)) {
    plan.recognized_architecture = true;
    plan.family = profile->family;
    plan.transformer = profile->transformer;
    plan.hybrid = profile->hybrid;
    plan.moe = profile->moe;
    plan.required_ops = profile->required_ops;
  } else {
    plan.family = "unknown";
    plan.required_ops = decoder_ops();
  }

  const auto* gguf = model.gguf_file();
  if (!gguf) {
    issue(plan, "model.not_gguf", "production execution requires a GGUF model");
    return plan;
  }

  plan.tensors = gguf->tensor_count();
  plan.tokenizer_tokens = gguf->tokenizer_tokens().size();
  if (plan.tokenizer_tokens == 0) issue(plan, "tokenizer.missing", "GGUF tokenizer.ggml.tokens is missing or empty");
  issue(plan, "tokenizer.bpe_unwired", "GGUF BPE/SentencePiece merge/rank handling is not wired yet; token array loading alone is insufficient for parity across model families");

  analyze_kernels(plan, *gguf);

  if (!plan.recognized_architecture) {
    issue(plan, "architecture.unrecognized", "architecture is not in cpullm's profile registry: " + md.architecture);
  } else if (plan.hybrid && md.architecture == "lfm2") {
    validate_hybrid_lfm(plan, md, *gguf);
    issue(plan, "executor.hybrid.graph_unwired", "hybrid graph scheduling is not wired yet for family: " + plan.family);
  } else if (plan.moe) {
    validate_generic_decoder(plan, md, *gguf);
    issue(plan, "executor.moe.unwired", "MoE router/expert scheduling is not wired yet for family: " + plan.family);
  } else if (md.architecture == "cpullm_tiny") {
    issue(plan, "executor.cpullm_tiny.missing", "cpullm_tiny executor tensor contract is reserved but no complete tensor set was found");
  } else {
    validate_generic_decoder(plan, md, *gguf);
    issue(plan, "executor.transformer.graph_unwired", "generic transformer graph scheduling is not wired yet for architecture: " + md.architecture);
  }

  plan.status = plan.issues.empty() ? PlanStatus::runnable : PlanStatus::blocked;
  return plan;
}

std::string ExecutionPlan::to_text() const {
  std::ostringstream out;
  out << "execution_plan_status=" << (runnable() ? "runnable" : "blocked") << '\n';
  out << "architecture=" << architecture << '\n';
  out << "family=" << family << '\n';
  out << "recognized_architecture=" << (recognized_architecture ? "yes" : "no") << '\n';
  out << "transformer=" << (transformer ? "yes" : "no") << '\n';
  out << "hybrid=" << (hybrid ? "yes" : "no") << '\n';
  out << "moe=" << (moe ? "yes" : "no") << '\n';
  out << "layers=" << layers << '\n';
  out << "tensors=" << tensors << '\n';
  out << "tokenizer_tokens=" << tokenizer_tokens << '\n';
  out << "kernels=f32:" << (kernels.f32 ? "yes" : "no")
      << ",f16:" << (kernels.f16 ? "yes" : "no")
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
  out << "  \"family\": \"" << esc(family) << "\",\n";
  out << "  \"recognized_architecture\": " << (recognized_architecture ? "true" : "false") << ",\n";
  out << "  \"transformer\": " << (transformer ? "true" : "false") << ",\n";
  out << "  \"hybrid\": " << (hybrid ? "true" : "false") << ",\n";
  out << "  \"moe\": " << (moe ? "true" : "false") << ",\n";
  out << "  \"layers\": " << layers << ",\n";
  out << "  \"tensors\": " << tensors << ",\n";
  out << "  \"tokenizer_tokens\": " << tokenizer_tokens << ",\n";
  out << "  \"kernels\": {\"f32\": " << (kernels.f32 ? "true" : "false")
      << ", \"f16\": " << (kernels.f16 ? "true" : "false")
      << ", \"q4_0\": " << (kernels.q4_0 ? "true" : "false")
      << ", \"q4_1\": " << (kernels.q4_1 ? "true" : "false")
      << ", \"q8_0\": " << (kernels.q8_0 ? "true" : "false")
      << ", \"unknown\": " << (kernels.unknown_types ? "true" : "false") << "},\n";
  out << "  \"required_ops\": [";
  for (std::size_t i = 0; i < required_ops.size(); ++i) out << (i ? ", " : "") << "\"" << esc(required_ops[i]) << "\"";
  out << "],\n";
  out << "  \"issues\": [";
  for (std::size_t i = 0; i < issues.size(); ++i) {
    if (i) out << ",";
    out << "{\"code\": \"" << esc(issues[i].code) << "\", \"message\": \"" << esc(issues[i].message) << "\"}";
  }
  out << "]\n}";
  return out.str();
}

} // namespace cpullm
