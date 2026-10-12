#!/bin/sh
# Cuts the four NASA videos of tests/fixtures/video/ — the page's video
# fixtures (web/search.js's verification, runtime/video.h) — from the files
# NASA's Image and Video Library publishes, each pinned by SHA-256. Each is a
# few seconds of its source, 640 × 360 at 30 frames a second, H.264 with no
# sound, so Chromium and Safari both play it. The cuts are committed; this
# records how they were made, and with what, in tests/fixtures/video/SOURCES.md.
# A source whose hash differs stops the run.
set -eu
cd "$(dirname "$0")/.."

SRC=.cache/nasa-video
OUT=tests/fixtures/video
mkdir -p "$SRC" "$OUT"

# name | NASA ID | SHA-256 of its ~medium.mp4 | start s | length s
CLIPS='launch|Artemis I Launch 2022 CU tracking from Press Site_compressed|a02fd793f0b33ce3ec213d9c77eba9426329412f7cd33428422494258af6bb72|46|8
spacewalk|NHQ_2019_1018_all woman spacewalk|f88fc55479af4c58e23b98f7ed7927a349c2bab46fe4cbe16aec2d16fbbf301f|13|7
landing|NHQ_2021_0222_VF_NASA Provides First Front-Row Seat to Perseverance Rover’s Landing on Mars|4f7253f3c4e8dce6d46c57bcba40e1711b3b27ed2c323b4d44ac73196806583f|229|8
earth|Earth Views from the International Space Station|caf71b75b1cd19816c55179384ea41e606d6ecd1088432b1e8e85e566917ce98|13|8'

quote() { python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$1"; }

echo "$CLIPS" | while IFS='|' read -r name id sha start length; do
    source="$SRC/$name-source.mp4"
    if [ ! -f "$source" ] || [ "$(shasum -a 256 "$source" | awk '{print $1}')" != "$sha" ]; then
        curl -sSf --max-time 600 -o "$source" \
            "https://images-assets.nasa.gov/video/$(quote "$id")/$(quote "$id~medium.mp4")"
    fi
    got=$(shasum -a 256 "$source" | awk '{print $1}')
    if [ "$got" != "$sha" ]; then
        echo "make_video_fixtures: $name's source has SHA-256 $got, not $sha" >&2
        rm -f "$source"
        exit 1
    fi
    ffmpeg -v error -y -ss "$start" -i "$source" -t "$length" -an -vf "scale=640:360:flags=lanczos" \
        -c:v libx264 -preset slow -crf 24 -pix_fmt yuv420p -profile:v main -r 30 -movflags +faststart \
        "$OUT/$name.mp4"
    echo "$name: $(shasum -a 256 "$OUT/$name.mp4" | awk '{print $1}')"
done
ffmpeg -version | head -1
