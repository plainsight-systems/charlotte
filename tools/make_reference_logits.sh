#!/bin/sh
# Writes tests/fixtures/reference/<model>.inc for each listed model: llama.cpp's
# logits for a pinned prompt on the pinned file, on its CPU backend and on
# Metal, for the reference test (tests/gpu/reference_test.cpp, graph/graph.h):
# Qwen3 0.6B and Llama 3.2 1B over 64 tokens, Gemma 3 1B over 600, past its
# 512-key window, each model's context sized to its prompt. Run on a Mac with
# Metal; the fixtures are committed, so the test needs none of this. llama.cpp
# is fetched at the commit below into .cache/llama.cpp.
set -eu
cd "$(dirname "$0")/.."

LLAMA_COMMIT=988190680d5a89fce97de3c20df2c2813731fd61
TEXT="The lighthouse keeper climbed the spiral stairs each evening at dusk, counting the steps as he went. \
There were one hundred and twelve of them, worn smooth in the middle by a century of boots. At the top he \
trimmed the wick, polished the great lens until it shone, and watched the first ships of the night turn \
toward the harbour, their lanterns small and steady against the dark water."
# Gemma 3's prompt runs past its 512-key window: the paragraph, eight times
# over, each time a chapter.
LONG_TEXT=""
for chapter in one two three four five six seven eight; do
    LONG_TEXT="${LONG_TEXT}Chapter ${chapter}. ${TEXT} "
done

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
# Each model: its file, its SHA-256, the tokens decoded, and the text.
reference() {
    .cache/reference_logits/reference_logits ".cache/test-data/models/$1.gguf" "$LLAMA_COMMIT" "$2" "$3" "$4" \
        > "tests/fixtures/reference/$1.inc"
}
reference qwen3-0.6b-q4_0 33bcc57074ec7b6eada5a90651ee546ec0c2b271002c22baf9f1b2dd1e8f75cb 64 "$TEXT"
reference llama-3.2-1b-instruct-q4_0 fa0390e7c043f89ae1847bd6682d748041a99d4ef3de0e0b27d33b6af97a8be8 64 "$TEXT"
reference gemma-3-1b-it-q4_0 27ee88e03be02e9ba73def9a819d570d8ad73716e50769e87f374ae394b0276e 600 "$LONG_TEXT"
