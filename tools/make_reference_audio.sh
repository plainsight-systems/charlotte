#!/bin/sh
# Writes the audio fixtures from llama.cpp at the embeddings' pin
# (runtime/audio.h). Run on a Mac; the fixtures are committed, so the tests
# need none of this.
#
#   - tests/fixtures/audio/*.wav: four public-domain recordings from NASA's
#     historical sounds page, each fetched and checked against its SHA-256
#     below, decoded by llama.cpp's own decoder to 16 kHz mono and cut to
#     its first 20 s (tests/fixtures/audio/SOURCES.md).
#   - tests/fixtures/reference/audio_prepare.inc: llama.cpp's gemma4a
#     spectrograms and clip sizes, for tests/media_audio_test.cpp.
#
# llama.cpp is fetched into .cache/llama.cpp-audio at the same commit as
# tools/make_reference_embeddings.sh's, and tools/reference_audio/
# mel_floor.patch applied: the one line that makes its log take Google's
# floor (media/audio.h).
set -eu
cd "$(dirname "$0")/.."

LLAMA_COMMIT=a9d27ac693ff0a1d3cba449c912c2b891904a4c7
MODEL=embeddinggemma-2-q8_0
MMPROJ=mmproj-embeddinggemma-2-q8_0
MMPROJ_SHA256=c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f
NASA=https://www.nasa.gov/wp-content/uploads/2015/01

python3 tools/fetch_test_data.py
SRC=.cache/llama.cpp-audio
if [ ! -d "$SRC/.git" ]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp
fi
git -C "$SRC" fetch -q --depth 1 origin "$LLAMA_COMMIT"
git -C "$SRC" checkout -q -f FETCH_HEAD
git -C "$SRC" apply "$PWD/tools/reference_audio/mel_floor.patch"

cmake -S tools/reference_audio -B .cache/reference_audio -DLLAMA_SOURCE="$PWD/$SRC" -DCMAKE_BUILD_TYPE=Release
cmake --build .cache/reference_audio --target reference_audio
TOOL=.cache/reference_audio/reference_audio
MODELS=.cache/test-data/models

# name, file on NASA's page, its SHA-256
mkdir -p .cache/reference_audio/sources tests/fixtures/audio
while read -r name file sha; do
    mp3=.cache/reference_audio/sources/$file
    [ -f "$mp3" ] || curl -sfL -o "$mp3" "$NASA/$file"
    echo "$sha  $mp3" | shasum -a 256 -c - >/dev/null
    "$TOOL" decode "$MODELS/$MODEL.gguf" "$MODELS/$MMPROJ.gguf" "$mp3" "tests/fixtures/audio/$name.wav" 20
done <<LIST
kennedy 586447main_JFKwechoosemoonspeech.mp3 b2e6f87c7254870ce6183ffaff839d255cef00a8c53aefd26d0d3898f33b7949
christmas 581549main_Apollo-8_Merry-Christmas.mp3 a551004c3a02af8f1afb517d218485383306729cfc735f54296a5bb9dbb051cd
countdown 590318main_ringtone_135_launch.mp3 9132e401834205b9bd0dbf93e2ab895516932975cc8de0b5cc50aee54eeb3d98
sputnik 578626main_sputnik-beep.mp3 84ab44b58396913c75ba0d8f5c29b18bca382269a10527bcf90a39c5c1460215
LIST

mkdir -p tests/fixtures/reference
"$TOOL" prepare "$MODELS/$MMPROJ.gguf" tests/fixtures/audio "$LLAMA_COMMIT" "$MMPROJ_SHA256" \
    > tests/fixtures/reference/audio_prepare.inc
