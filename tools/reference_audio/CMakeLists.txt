# llama.cpp's preparation and embedding of audio for the audio tests
# (tests/media_audio_test.cpp, runtime/audio.h), built against llama.cpp
# and its multimodal library at the commit tools/make_reference_audio.sh
# pins, with tools/reference_audio/mel_floor.patch applied — or without, for
# the unchanged library's embeddings the fixture compares against. Static,
# so the tool can reach mtmd's own preprocessor, which the library does not
# export. A development tool: never part of the harness's build.
cmake_minimum_required(VERSION 3.20)
project(reference_audio CXX C)

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

add_executable(reference_audio main.cpp)
target_include_directories(reference_audio PRIVATE
    ${LLAMA_SOURCE}/tools/mtmd
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(reference_audio PRIVATE mtmd llama)
