#!/bin/sh
# CI's smoke run of the GPU tests, on the runners' software Vulkan driver.
#
#   gpu_smoke.sh <preset>     e.g. native-release, native-sanitize
#
# Every GPU test file but the four that check a whole model against
# llama.cpp, which take minutes each on a software driver and run on the
# development machine (make test). Each file runs in a process of its own:
# run whole, the suite's memory grows past a runner's 16 GB.
set -eu
cd "$(dirname "$0")/.."

bin="build/$1/tests/gpu/charlotte_gpu_tests"
for file in tests/gpu/*_test.cpp; do
    case "$file" in
        */reference_test.cpp | */vision_embed_test.cpp | */audio_embed_test.cpp | */video_embed_test.cpp) continue ;;
    esac
    echo "== $(basename "$file")"
    "$bin" -sf="*/$(basename "$file")"
done
