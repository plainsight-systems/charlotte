# llama.cpp's preparation and embedding of images for the vision tests
# (tests/media_image_test.cpp, runtime/vision.h), built against llama.cpp
# and its multimodal library at the commit tools/make_reference_vision.sh
# pins. Static, so the tool can reach mtmd's own preprocessor, which the
# library does not export. A development tool: never part of the harness's
# build.
cmake_minimum_required(VERSION 3.20)
project(reference_vision CXX C)

set(CMAKE_CXX_STANDARD 20)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_SOURCE} llama EXCLUDE_FROM_ALL)

add_executable(reference_vision main.cpp)
target_include_directories(reference_vision PRIVATE
    ${LLAMA_SOURCE}/tools/mtmd
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(reference_vision PRIVATE mtmd llama)
