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

# llama.cpp's own resize, img_tool, which mtmd-image.cpp keeps to itself:
# copied verbatim from the pinned source when the tool is configured, so a
# video frame's reference is stretched to its grid by llama.cpp's resampler
# (main.cpp's video mode), where its preprocessors only letterbox.
file(READ ${LLAMA_SOURCE}/tools/mtmd/mtmd-image.cpp MTMD_IMAGE)
string(FIND "${MTMD_IMAGE}" "struct img_tool {" IMG_TOOL_BEGIN)
if(IMG_TOOL_BEGIN EQUAL -1)
    message(FATAL_ERROR "mtmd-image.cpp has no struct img_tool at this commit")
endif()
string(SUBSTRING "${MTMD_IMAGE}" ${IMG_TOOL_BEGIN} -1 IMG_TOOL_REST)
string(FIND "${IMG_TOOL_REST}" "\n};\n" IMG_TOOL_LENGTH)
if(IMG_TOOL_LENGTH EQUAL -1)
    message(FATAL_ERROR "mtmd-image.cpp's struct img_tool does not end where expected")
endif()
math(EXPR IMG_TOOL_LENGTH "${IMG_TOOL_LENGTH} + 3")
string(SUBSTRING "${IMG_TOOL_REST}" 0 ${IMG_TOOL_LENGTH} IMG_TOOL)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/generated/img_tool.inc
     "// Copied by CMakeLists.txt from llama.cpp's tools/mtmd/mtmd-image.cpp. Do not edit.\n${IMG_TOOL}\n")
target_include_directories(reference_vision PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
