cmake_minimum_required(VERSION 3.25)
project(charlotte LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(EmbedShaders)

# ---------------------------------------------------------------------------
# Core: platform-neutral C++. Compiles unchanged in both configurations.
#
# Invariant: nothing here may include <emscripten.h> or anything under
# src/wasm/ or web/. tools/check_boundaries.sh enforces this in CI.
# ---------------------------------------------------------------------------
# Diagnostic instrumentation is compiled out unless explicitly requested. See
# docs/research/2026-08-31-measurement-build-configurations.md — the axis is
# instrumentation, not optimisation, so a diagnostic build is still Release.
option(BLLM_ENABLE_DIAGNOSTICS
       "Compile diagnostic instrumentation into the build (perturbs timing)" OFF)

add_library(charlotte_core STATIC
    src/core/arch/describe.cpp
    src/core/arch/gemma3/gemma3.cpp
    src/core/arch/llama/llama.cpp
    src/core/arch/qwen3/qwen3.cpp
    src/core/graph/graph.cpp
    src/core/capability/capability.cpp
    src/core/gguf/arrays.cpp
    src/core/gguf/index.cpp
    src/core/gguf/reader.cpp
    src/core/preflight/preflight.cpp
    src/core/tokenizer/bpe/byte_level_bpe.cpp
    src/core/tokenizer/bpe/merge.cpp
    src/core/tokenizer/bpe/merge_table.cpp
    src/core/tokenizer/bpe/piece_cache.cpp
    src/core/tokenizer/bpe/sentencepiece_bpe.cpp
    src/core/tokenizer/bpe/sentencepiece_merges.cpp
    src/core/tokenizer/load.cpp
    src/core/tokenizer/nfc.cpp
    src/core/tokenizer/pretokenize.cpp
    src/core/tokenizer/special.cpp
    src/core/tokenizer/unicode.cpp
    src/core/tokenizer/unicode_tables.cpp
    src/core/tokenizer/utf8_stream.cpp
    src/core/tokenizer/vocabulary.cpp
    src/core/residency/plan.cpp
    src/core/cache/kv.cpp
    src/core/cache/prefix.cpp
    src/core/formats/q6_k/q6_k_repack.cpp
    src/core/residency/piece_writer.cpp
    src/core/kernels/gather/gather.cpp
    src/core/kernels/matmul/matmul.cpp
    src/core/kernels/norm/norm.cpp
    src/core/kernels/attention/attention.cpp
    src/core/kernels/rope/rope.cpp
    src/core/kernels/schedule.cpp
    src/core/residency/routes.cpp
    src/core/gpu/dispatch_math.cpp
)

if(BLLM_ENABLE_DIAGNOSTICS)
    target_compile_definitions(charlotte_core PUBLIC BLLM_DIAGNOSTICS_ENABLED=1)
    message(STATUS
        "charlotte: diagnostics ENABLED — timings from this build are not quotable")
endif()

target_include_directories(charlotte_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

bllm_embed_shaders(
    TARGET  charlotte_core
    OUTPUT  "${CMAKE_BINARY_DIR}/generated/bllm/shaders_generated.h"
    SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/vector_add.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/formats/f32/f32.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/formats/f16/f16.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/formats/f16/f16_pack.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/formats/q4_0/q4_0.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/formats/q4_1/q4_1.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/formats/q6_k/q6_k.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/formats/q8_0/q8_0.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernels/step.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernels/gather/gather.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernels/matmul/matmul.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernels/norm/norm.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernels/attention/attention.wgsl"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kernels/rope/rope.wgsl"
)

# ---------------------------------------------------------------------------
# GPU: the core code that holds WebGPU objects, apart from the core that needs
# none. It needs an implementation of webgpu.h: in the wasm configuration
# --use-port=emdawnwebgpu, over the browser's WebGPU; natively, Dawn itself,
# prebuilt and pinned by cmake/dawn.json to the release the port comes from,
# so both configurations run one webgpu.h and GPU code is tested natively.
# ---------------------------------------------------------------------------
add_library(charlotte_gpu STATIC
    src/core/gpu/device.cpp
    src/core/gpu/self_check.cpp
    src/core/kernels/program.cpp
    src/core/residency/mapping.cpp
    src/core/residency/upload.cpp
    src/core/residency/witness.cpp
)
target_link_libraries(charlotte_gpu PUBLIC charlotte_core)
# Not compiled at all in a clean build (upload_check.h, TLM.8).
if(BLLM_ENABLE_DIAGNOSTICS)
    target_sources(charlotte_gpu PRIVATE src/core/residency/upload_check.cpp)
endif()

# Every public header that holds WebGPU objects compiles on its own, with
# nothing included before it, in both configurations: one generated
# translation unit each, against Dawn natively and the port in wasm, built
# with everything else, so neither half rests on a manual check.
set(BLLM_GPU_HEADERS
    core/gpu/callback_mode.h
    core/gpu/device.h
    core/gpu/readback_bench.h
    core/gpu/self_check.h
    core/gpu/userdata.h
    core/gpu/wgpu_handles.h
    core/kernels/program.h
    core/residency/mapping.h
    core/residency/upload.h
    core/residency/upload_check.h
)
set(BLLM_GPU_HEADER_CHECKS "")
foreach(header IN LISTS BLLM_GPU_HEADERS)
    string(MAKE_C_IDENTIFIER "${header}" check)
    set(check "${CMAKE_BINARY_DIR}/header_checks/${check}.cpp")
    file(CONFIGURE OUTPUT "${check}" CONTENT "#include \"${header}\"\n")
    list(APPEND BLLM_GPU_HEADER_CHECKS "${check}")
endforeach()
add_library(charlotte_gpu_header_checks OBJECT ${BLLM_GPU_HEADER_CHECKS})
target_link_libraries(charlotte_gpu_header_checks PRIVATE charlotte_gpu)

if(EMSCRIPTEN)
    target_compile_options(charlotte_gpu PUBLIC --use-port=emdawnwebgpu)
    # The browser's event loop runs every callback (core/gpu/callback_mode.h).
    target_compile_definitions(charlotte_gpu PUBLIC BLLM_GPU_PROCESS_EVENTS=0)

    # Not merely preprocessed away — not compiled at all in a clean build, so
    # the symbol cannot reach the shipped module.
    if(BLLM_ENABLE_DIAGNOSTICS)
        target_sources(charlotte_gpu PRIVATE src/core/gpu/readback_bench.cpp)
    endif()

    # -----------------------------------------------------------------------
    # Browser target. The only Emscripten-aware translation unit.
    # -----------------------------------------------------------------------
    add_executable(charlotte src/wasm/bindings.cpp)
    target_link_libraries(charlotte PRIVATE charlotte_gpu)

    target_link_options(charlotte PRIVATE
        --use-port=emdawnwebgpu
        -sMODULARIZE=1
        -sEXPORT_ES6=1
        -sENVIRONMENT=web,worker
        # WASM.1: growth is a FALLBACK, not a default. It converts a
        # deterministic out-of-memory at startup into a nondeterministic one
        # under load. It is enabled because -sINITIAL_HEAP is sized only from
        # the loader's measured peak, never from a figure chosen for comfort,
        # and no measured peak exists to size it from.
        #
        # MAXIMUM_MEMORY bounds the damage and makes an out-of-memory
        # reproducible. The default is 2GB, which is a ceiling inherited rather
        # than chosen.
        -sALLOW_MEMORY_GROWTH=1
        -sMAXIMUM_MEMORY=536870912
        -sEXPORTED_RUNTIME_METHODS=['HEAPU8']
        -sEXPORTED_FUNCTIONS=['_main','_malloc','_free']
        # Deliberately absent: -pthread. GitHub Pages cannot set COOP/COEP,
        # so SharedArrayBuffer is unavailable and pthreads cannot be used.
    )
    set_target_properties(charlotte PROPERTIES
        OUTPUT_NAME "charlotte"
        SUFFIX ".mjs"
    )
else()
    # -----------------------------------------------------------------------
    # Native: tests only. The platform-neutral tests need no GPU; the GPU
    # tests run on Dawn, through this machine's GPU, or Mesa's software Vulkan
    # driver in CI.
    # -----------------------------------------------------------------------
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dawn.json" BLLM_DAWN_PINS)
    string(JSON BLLM_DAWN_RELEASE GET "${BLLM_DAWN_PINS}" release)
    string(TOLOWER "${CMAKE_HOST_SYSTEM_PROCESSOR}" BLLM_HOST_CPU)
    if(BLLM_HOST_CPU STREQUAL "amd64")
        set(BLLM_HOST_CPU x86_64)
    elseif(BLLM_HOST_CPU STREQUAL "aarch64")
        set(BLLM_HOST_CPU arm64)
    endif()
    set(BLLM_DAWN_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/.cache/dawn/${BLLM_DAWN_RELEASE}/${CMAKE_HOST_SYSTEM_NAME}-${BLLM_HOST_CPU}")
    # lib/ on macOS, lib64/ on Linux, as Dawn's own release packages it.
    find_package(Dawn CONFIG PATHS "${BLLM_DAWN_DIR}"
                 PATH_SUFFIXES lib/cmake/Dawn lib64/cmake/Dawn NO_DEFAULT_PATH)
    if(NOT Dawn_FOUND)
        message(FATAL_ERROR
            "charlotte: native Dawn ${BLLM_DAWN_RELEASE} not found in ${BLLM_DAWN_DIR}.\n"
            "Fetch it with: python3 tools/fetch_dawn.py (or make dawn).")
    endif()
    find_package(Threads REQUIRED)
    target_link_libraries(charlotte_gpu PUBLIC dawn::webgpu_dawn)
    # Callbacks given a mode run only inside wgpuInstanceProcessEvents, on the
    # caller's thread; the uncaptured-error callback is the one exception
    # (core/gpu/callback_mode.h).
    target_compile_definitions(charlotte_gpu PUBLIC BLLM_GPU_PROCESS_EVENTS=1)

    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(tests/gpu)
    add_subdirectory(bench)
endif()
