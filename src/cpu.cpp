#include "cpullm/cpullm.hpp"

#include <sstream>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
  #if defined(_MSC_VER)
    #include <intrin.h>
  #else
    #include <cpuid.h>
  #endif
#endif

namespace cpullm {

CpuFeatures detect_cpu_features() {
  CpuFeatures f{};
#if defined(__aarch64__) || defined(_M_ARM64)
  f.neon = true;
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
  #if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 1);
    f.sse2 = (regs[3] & (1 << 26)) != 0;
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    __cpuidex(regs, 7, 0);
    f.avx2 = osxsave && ((regs[1] & (1 << 5)) != 0);
    f.avx512f = osxsave && ((regs[1] & (1 << 16)) != 0);
  #else
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    f.sse2 = (edx & bit_SSE2) != 0;
    const bool osxsave = (ecx & bit_OSXSAVE) != 0;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
      f.avx2 = osxsave && ((ebx & bit_AVX2) != 0);
      f.avx512f = osxsave && ((ebx & bit_AVX512F) != 0);
    }
  #endif
#endif
  return f;
}

std::string CpuFeatures::summary() const {
  std::ostringstream out;
  out << "cpu[";
  bool first = true;
  auto add = [&](bool enabled, const char* name) {
    if (!enabled) return;
    if (!first) out << ',';
    out << name;
    first = false;
  };
  add(sse2, "sse2"); add(avx2, "avx2"); add(avx512f, "avx512f"); add(neon, "neon");
  if (first) out << "portable";
  out << "]";
  return out.str();
}

} // namespace cpullm
