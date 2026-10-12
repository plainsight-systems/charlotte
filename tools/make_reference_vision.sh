#!/bin/sh
# Writes the vision fixtures from llama.cpp at the embeddings' commit
# (runtime/vision.h): tests/fixtures/reference/vision_prepare.inc, llama.cpp's
# preparation of the images of tests/fixtures/vision/scenes.h — the grid its
# rule gives each and a hash of the bytes it resizes them to — for
# tests/media_image_test.cpp. Run on a Mac; the fixtures are committed, so
# the tests need none of this. llama.cpp is fetched into
# .cache/llama.cpp-embeddings, as tools/make_reference_embeddings.sh fetches
# it, at the same commit.
set -eu
cd "$(dirname "$0")/.."

LLAMA_COMMIT=a9d27ac693ff0a1d3cba449c912c2b891904a4c7
MMPROJ=mmproj-embeddinggemma-2-q8_0
SHA256=c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f

python3 tools/fetch_test_data.py
SRC=.cache/llama.cpp-embeddings
if [ ! -d "$SRC/.git" ]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp
fi
git -C "$SRC" fetch -q --depth 1 origin "$LLAMA_COMMIT"
git -C "$SRC" checkout -q FETCH_HEAD

cmake -S tools/reference_vision -B .cache/reference_vision -DLLAMA_SOURCE="$PWD/$SRC" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build .cache/reference_vision --target reference_vision
mkdir -p tests/fixtures/reference
.cache/reference_vision/reference_vision prepare ".cache/test-data/models/$MMPROJ.gguf" "$LLAMA_COMMIT" "$SHA256" \
    > tests/fixtures/reference/vision_prepare.inc

# The embeddings: tests/fixtures/reference/vision_embeddings.inc, for
# tests/gpu/vision_embed_test.cpp (runtime/vision.h). llama.cpp is given a
# copy of the mmproj with clip.use_gelu set, so it computes the activation
# the model was trained with (arch/gemma4v); the published file is given too,
# for the cosine the fixture records beside each image.
MODEL=embeddinggemma-2-q8_0
MODEL_SHA256=2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135
PATCHED=.cache/reference_vision/$MMPROJ-use-gelu.gguf
python3 tools/gguf_add_bool.py ".cache/test-data/models/$MMPROJ.gguf" "$PATCHED" clip.use_gelu true
.cache/reference_vision/reference_vision embed ".cache/test-data/models/$MODEL.gguf" "$PATCHED" \
    ".cache/test-data/models/$MMPROJ.gguf" "$LLAMA_COMMIT" "$MODEL_SHA256" "$SHA256" \
    > tests/fixtures/reference/vision_embeddings.inc

# The video embeddings: tests/fixtures/reference/video_embeddings.inc, for
# tests/gpu/video_embed_test.cpp (runtime/video.h), from the same patched
# mmproj: each clip's frames sized by Google's rule, stretched by llama.cpp's
# resampler, encoded one a frame.
.cache/reference_vision/reference_vision video ".cache/test-data/models/$MODEL.gguf" "$PATCHED" \
    "$LLAMA_COMMIT" "$MODEL_SHA256" "$SHA256" > tests/fixtures/reference/video_embeddings.inc

# The page's video check (web/search.js): llama.cpp's scores for the NASA
# clips of tests/fixtures/video/, their frames decoded by FFmpeg as Google's
# processor picks them — frame ⌊i × 30⌋ of each 30 fps clip, one a second —
# 4 to a clip, a 16:9 frame's 120 tokens leaving room for 4 (runtime/
# video.h), each query with its task prompt: tests/fixtures/video/
# page_scores.json.
FRAMES=.cache/reference_vision/nasa-frames
rm -rf "$FRAMES"
mkdir -p "$FRAMES"
MANIFEST="$FRAMES/manifest.txt"
: > "$MANIFEST"
for name in launch spacewalk landing earth; do
    mkdir -p "$FRAMES/$name"
    ffmpeg -v error -i "tests/fixtures/video/$name.mp4" -vf "select='not(mod(n\,30))'" -fps_mode passthrough \
        "$FRAMES/$name/%02d.ppm"
    set -- "$FRAMES/$name"/*.ppm
    first=0
    while [ $# -gt 0 ]; do
        line="clip $name-$first"
        n=0
        while [ $# -gt 0 ] && [ $n -lt 4 ]; do
            line="$line $1"
            shift
            n=$((n + 1))
        done
        echo "$line" >> "$MANIFEST"
        first=$((first + n))
    done
done
for query in "a rocket lifting off its launch pad at night" "astronauts working outside the space station" \
    "a rover lowered onto the surface of Mars" "the curve of the Earth and clouds seen from orbit"; do
    echo "query task: search result | query: $query" >> "$MANIFEST"
done
.cache/reference_vision/reference_vision frames ".cache/test-data/models/$MODEL.gguf" "$PATCHED" "$MANIFEST" \
    > tests/fixtures/video/page_scores.json
