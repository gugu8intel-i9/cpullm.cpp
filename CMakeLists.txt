cmake_minimum_required(VERSION 3.16)
project(cpullm VERSION 0.1.0 LANGUAGES CXX)

option(CPULLM_BUILD_TESTS "Build cpullm tests" ON)
option(CPULLM_BUILD_EXAMPLES "Build cpullm examples" ON)
option(CPULLM_NATIVE "Enable native CPU tuning flags" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cpullm
  src/arena.cpp
  src/cpu.cpp
  src/engine.cpp
  src/graph.cpp
  src/llama_compat.cpp
  src/matmul.cpp
  src/model.cpp
  src/tokenizer.cpp
)

target_include_directories(cpullm PUBLIC include)
target_compile_features(cpullm PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(cpullm PRIVATE /W4 /permissive-)
else()
  target_compile_options(cpullm PRIVATE -Wall -Wextra -Wpedantic)
  if(CPULLM_NATIVE)
    target_compile_options(cpullm PRIVATE -march=native)
  endif()
endif()

add_executable(cpullm-cli apps/cpullm-cli.cpp)
target_link_libraries(cpullm-cli PRIVATE cpullm)

add_executable(llama-cli apps/cpullm-cli.cpp)
target_link_libraries(llama-cli PRIVATE cpullm)

if(CPULLM_BUILD_EXAMPLES)
  add_executable(cpullm-minimal examples/minimal.cpp)
  target_link_libraries(cpullm-minimal PRIVATE cpullm)
endif()

if(CPULLM_BUILD_TESTS)
  enable_testing()
  add_executable(cpullm-tests tests/test_core.cpp)
  target_link_libraries(cpullm-tests PRIVATE cpullm)
  add_test(NAME cpullm-core COMMAND cpullm-tests)
endif()
