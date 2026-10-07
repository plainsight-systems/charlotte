# llama.cpp's logits for the reference test (tests/gpu/reference_test.cpp),
# built against llama.cpp at the commit tools/make_reference_logits.sh pins.
# A development tool: never part of the harness's build.
cmake_minimum_required(VERSION 3.20)
project(reference_logits CXX C)

set(CMAKE_CXX_STANDARD 20)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_SOURCE} llama EXCLUDE_FROM_ALL)

add_executable(reference_logits main.cpp)
target_link_libraries(reference_logits PRIVATE llama)
