include(FetchContent)

# Pinned by tag. Dependency resolution must be deterministic; a floating ref
# would make the test build a moving target.
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        1da23a3e8119ec5cce4f9388e91b065e20bf06f5
)
FetchContent_MakeAvailable(doctest)

add_executable(charlotte_tests
    dispatch_math_test.cpp
    shader_embed_test.cpp
    unique_handle_test.cpp
    run_guard_test.cpp
    quant_q4_0_test.cpp
    gguf_checked_test.cpp
    gguf_arrays_test.cpp
    test_data_test.cpp
    tokenizer_bpe_test.cpp
    tokenizer_merge_table_test.cpp
    tokenizer_merge_test.cpp
    tokenizer_nfc_test.cpp
    tokenizer_piece_cache_test.cpp
    tokenizer_sentencepiece_test.cpp
    tokenizer_sentencepiece_merges_test.cpp
    tokenizer_split_test.cpp
    tokenizer_unicode_test.cpp
    tokenizer_utf8_stream_test.cpp
    tokenizer_vocabulary_test.cpp
    gguf_parse_test.cpp
    gguf_formats_test.cpp
    gguf_malformed_test.cpp
    contract_headers_test.cpp
    preflight_test.cpp
    arch_describe_test.cpp
    graph_test.cpp
    kernels_schedule_test.cpp
    kernels_topk_test.cpp
    kernels_order_test.cpp
    runtime_turn_test.cpp
    sampler_test.cpp
    residency_plan_test.cpp
    cache_kv_test.cpp
    cache_prefix_test.cpp
    formats_q6_k_repack_test.cpp
    kernels_gather_test.cpp
    kernels_attention_test.cpp
    kernels_geometry_test.cpp
    kernels_matmul_test.cpp
    kernels_norm_test.cpp
    kernels_rope_test.cpp
    residency_piece_writer_test.cpp
    residency_routes_test.cpp
    main_test.cpp
)

target_link_libraries(charlotte_tests PRIVATE charlotte_core doctest::doctest)

# The shader test reads the on-disk .wgsl to compare against the embedded copy.
target_include_directories(charlotte_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_compile_definitions(charlotte_tests PRIVATE
    BLLM_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../src/shaders"
    BLLM_GGUF_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/gguf"
    # Fetched, not committed: tools/fetch_test_data.py (make test-data).
    BLLM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../.cache/test-data"
)

add_test(NAME charlotte_tests COMMAND charlotte_tests)
