#!/bin/sh
# Writes tests/fixtures/reference/embeddinggemma-2-q8_0.inc: llama.cpp's
# embeddings for pinned texts on the pinned file, on its CPU backend and on
# Metal, for the embedder's reference test (tests/gpu/embedder_test.cpp,
# runtime/embedder.h): a short query, a passage of a few hundred tokens, one
# of exactly 512 tokens with its ends — the most one encode holds — and text
# in three scripts, each in the task prompt EmbeddingGemma 2's card gives it.
# Run on a Mac with Metal; the fixture is committed, so the test needs none of
# this. llama.cpp is fetched at the embeddings' own commit, the first holding
# gemma-embedding2, into .cache/llama.cpp-embeddings; the logits fixtures
# (make_reference_logits.sh) keep theirs.
set -eu
cd "$(dirname "$0")/.."

LLAMA_COMMIT=a9d27ac693ff0a1d3cba449c912c2b891904a4c7
MODEL=embeddinggemma-2-q8_0
SHA256=2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135
PARAGRAPH="The lighthouse keeper climbed the spiral stairs each evening at dusk, counting the steps as he went. \
There were one hundred and twelve of them, worn smooth in the middle by a century of boots. At the top he \
trimmed the wick, polished the great lens until it shone, and watched the first ships of the night turn \
toward the harbour, their lanterns small and steady against the dark water."
PASSAGE=""
for chapter in one two three four; do
    PASSAGE="${PASSAGE}Chapter ${chapter}. ${PARAGRAPH} "
done
LONG=""
for chapter in one two three four five six seven eight; do
    LONG="${LONG}Chapter ${chapter}. ${PARAGRAPH} "
done

python3 tools/fetch_test_data.py
SRC=.cache/llama.cpp-embeddings
if [ ! -d "$SRC/.git" ]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp
fi
git -C "$SRC" fetch -q --depth 1 origin "$LLAMA_COMMIT"
git -C "$SRC" checkout -q FETCH_HEAD

cmake -S tools/reference_embeddings -B .cache/reference_embeddings -DLLAMA_SOURCE="$PWD/$SRC" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build .cache/reference_embeddings --target reference_embeddings
mkdir -p tests/fixtures/reference
.cache/reference_embeddings/reference_embeddings ".cache/test-data/models/$MODEL.gguf" "$LLAMA_COMMIT" "$SHA256" \
    query "task: search result | query: how many steps lead to the top of the lighthouse?" \
    passage "title: none | text: $PASSAGE" \
    context "@512 title: none | text: $LONG" \
    scripts "title: none | text: 灯台守は毎晩階段を上った。Смотритель маяка поднимался по лестнице. حارس المنارة صعد الدرج كل مساء." \
    > "tests/fixtures/reference/$MODEL.inc"
