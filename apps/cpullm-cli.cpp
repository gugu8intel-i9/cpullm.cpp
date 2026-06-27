#include "cpullm/cpullm.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      std::cerr << "usage: cpullm-cli <manifest.yml> <prompt>\n";
      std::cerr << cpullm::detect_cpu_features().summary() << '\n';
      return 2;
    }
    auto model = cpullm::Model::load_manifest(argv[1]);
    cpullm::Engine engine(std::move(model));
    std::cout << engine.generate(argv[2]) << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "cpullm error: " << e.what() << '\n';
    return 1;
  }
}
