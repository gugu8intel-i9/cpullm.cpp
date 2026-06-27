#include "cpullm/cpullm.hpp"

#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

struct CliOptions {
  std::string model_path;
  std::string prompt;
  cpullm::GenerationConfig generation;
  std::size_t context_size = 2048;
  std::size_t threads = 0;
  bool show_help = false;
  bool show_version = false;
  bool stream = false;
  bool inspect = false;
};

void print_help(const char* argv0) {
  std::cout
      << "cpullm.cpp - lightweight llama.cpp-compatible CPU LLM runtime\n\n"
      << "usage:\n"
      << "  " << argv0 << " -m <model-or-manifest> -p <prompt> [options]\n"
      << "  " << argv0 << " <model-or-manifest> <prompt>\n\n"
      << "llama.cpp-compatible options:\n"
      << "  -m, --model <path>       model path or cpullm YAML manifest\n"
      << "  -p, --prompt <text>      prompt text\n"
      << "  -n, --n-predict <count>  maximum tokens to generate\n"
      << "  --temp <value>           sampling temperature\n"
      << "  -c, --ctx-size <count>   context size hint\n"
      << "  -t, --threads <count>    worker thread hint; 0 means auto\n"
      << "  -h, --help               show help\n"
      << "  --version                show version\n"
      << "  --stream                 stream tokens as they are produced\n"
      << "  --inspect                print model/GGUF metadata and exit\n\n"
      << "status:\n"
      << "  This foundation accepts llama.cpp-style invocations, but full GGUF\n"
      << "  inference compatibility is still on the roadmap. YAML manifests work now.\n";
}

std::optional<std::string> require_value(int& i, int argc, char** argv, std::string_view flag) {
  if (i + 1 >= argc) {
    std::cerr << "missing value for " << flag << '\n';
    return std::nullopt;
  }
  return std::string{argv[++i]};
}

std::optional<CliOptions> parse_args(int argc, char** argv) {
  CliOptions opt;
  std::vector<std::string> positional;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") opt.show_help = true;
    else if (arg == "--version") opt.show_version = true;
    else if (arg == "--stream") opt.stream = true;
    else if (arg == "--inspect") opt.inspect = true;
    else if (arg == "-m" || arg == "--model") {
      auto value = require_value(i, argc, argv, arg);
      if (!value) return std::nullopt;
      opt.model_path = *value;
    } else if (arg == "-p" || arg == "--prompt") {
      auto value = require_value(i, argc, argv, arg);
      if (!value) return std::nullopt;
      opt.prompt = *value;
    } else if (arg == "-n" || arg == "--n-predict") {
      auto value = require_value(i, argc, argv, arg);
      if (!value) return std::nullopt;
      opt.generation.max_tokens = static_cast<std::size_t>(std::stoull(*value));
    } else if (arg == "--temp") {
      auto value = require_value(i, argc, argv, arg);
      if (!value) return std::nullopt;
      opt.generation.temperature = std::stof(*value);
    } else if (arg == "-c" || arg == "--ctx-size") {
      auto value = require_value(i, argc, argv, arg);
      if (!value) return std::nullopt;
      opt.context_size = static_cast<std::size_t>(std::stoull(*value));
    } else if (arg == "-t" || arg == "--threads") {
      auto value = require_value(i, argc, argv, arg);
      if (!value) return std::nullopt;
      opt.threads = static_cast<std::size_t>(std::stoull(*value));
    } else if (!arg.empty() && arg[0] == '-') {
      std::cerr << "warning: option '" << arg << "' is not implemented yet and will be ignored\n";
    } else {
      positional.push_back(arg);
    }
  }

  if (opt.model_path.empty() && !positional.empty()) opt.model_path = positional[0];
  if (opt.prompt.empty() && positional.size() > 1) opt.prompt = positional[1];
  return opt;
}

} // namespace

int main(int argc, char** argv) {
  try {
    auto parsed = parse_args(argc, argv);
    if (!parsed) return 2;
    const auto& opt = *parsed;

    if (opt.show_version) {
      std::cout << "cpullm.cpp 0.1.0 foundation (llama.cpp-compatible CLI scaffold)\n";
      return 0;
    }
    if (opt.show_help || opt.model_path.empty()) {
      print_help(argv[0]);
      return opt.show_help ? 0 : 2;
    }

    auto model = cpullm::Model::load(opt.model_path);
    if (opt.inspect) {
      const auto& md = model.metadata();
      std::cout << "name=" << md.name << '\n'
                << "architecture=" << md.architecture << '\n'
                << "context_length=" << md.context_length << '\n'
                << "blocks=" << md.block_count << '\n'
                << "embedding_length=" << md.embedding_length << '\n'
                << "feed_forward_length=" << md.feed_forward_length << '\n'
                << "attention_heads=" << md.attention_heads << '\n'
                << "attention_kv_heads=" << md.attention_kv_heads << '\n'
                << "vocab_size=" << md.vocab_size << '\n';
      if (const auto* gguf = model.gguf_file()) {
        std::cout << gguf->summary() << '\n';
        std::size_t shown = 0;
        for (const auto& t : gguf->tensors()) {
          if (shown++ >= 12) break;
          std::cout << "tensor " << t.name << " type=" << t.ggml_type << " bytes=" << t.bytes << " shape=";
          for (auto d : t.shape) std::cout << d << 'x';
          std::cout << '\n';
        }
      }
      return 0;
    }
    cpullm::Engine engine(std::move(model));
    if (opt.stream) {
      std::cout << opt.prompt;
      engine.generate_stream(opt.prompt, opt.generation, [](const cpullm::TokenEvent& event) {
        std::cout << event.text << std::flush;
        return true;
      });
      std::cout << '\n';
    } else {
      std::cout << engine.generate(opt.prompt, opt.generation) << '\n';
    }
    std::cerr << cpullm::detect_cpu_features().summary()
              << " ctx=" << opt.context_size
              << " threads=" << (opt.threads == 0 ? std::string{"auto"} : std::to_string(opt.threads))
              << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "cpullm error: " << e.what() << '\n';
    return 1;
  }
}
