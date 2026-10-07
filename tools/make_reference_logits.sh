#!/bin/sh
# Writes tests/fixtures/reference/qwen3-0.6b-q4_0.inc: llama.cpp's logits for
# a pinned prompt on the pinned Qwen3 0.6B file, on its CPU backend and on
# Metal, for the reference test (tests/gpu/reference_test.cpp, graph/graph.h).
# Run on a Mac with Metal; the fixture is committed, so the test needs none of
# this. llama.cpp is fetched at the commit below into .cache/llama.cpp.
set -eu
cd "$(dirname "$0")/.."

LLAMA_COMMIT=988190680d5a89fce97de3c20df2c2813731fd61
MODEL_SHA256=33bcc57074ec7b6eada5a90651ee546ec0c2b271002c22baf9f1b2dd1e8f75cb
MODEL=.cache/test-data/models/qwen3-0.6b-q4_0.gguf
TOKENS=64
TEXT="The lighthouse keeper climbed the spiral stairs each evening at dusk, counting the steps as he went. \
There were one hundred and twelve of them, worn smooth in the middle by a century of boots. At the top he \
trimmed the wick, polished the great lens until it shone, and watched the first ships of the night turn \
toward the harbour, their lanterns small and steady against the dark water."

python3 tools/fetch_test_data.py
SRC=.cache/llama.cpp
if [ ! -d "$SRC/.git" ]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp
fi
git -C "$SRC" fetch -q --depth 1 origin "$LLAMA_COMMIT"
git -C "$SRC" checkout -q FETCH_HEAD

cmake -S tools/reference_logits -B .cache/reference_logits -DLLAMA_SOURCE="$PWD/$SRC" -DCMAKE_BUILD_TYPE=Release
cmake --build .cache/reference_logits --target reference_logits
mkdir -p tests/fixtures/reference
.cache/reference_logits/reference_logits "$MODEL" "$LLAMA_COMMIT" "$MODEL_SHA256" "$TOKENS" "$TEXT" \
    > tests/fixtures/reference/qwen3-0.6b-q4_0.inc
