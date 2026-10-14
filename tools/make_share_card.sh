#!/bin/sh
# Cuts the page's share card, web/assets/charlotte-card.jpg, from the README's
# hero, web/assets/charlotte-readme.png. The hero is 3:1; a share card is
# 1200 × 630, and a platform that center-crops the hero to that shape cuts the
# web in half. So the card is the hero's web and spider, cropped inside its
# border and past the window on its left, at 1200 × 630. web/index.html names
# the card and its size in its Open Graph tags.
set -eu
cd "$(dirname "$0")/.."

ffmpeg -v error -y -i web/assets/charlotte-readme.png \
    -vf "crop=1273:666:885:26,scale=1200:630:flags=lanczos" -q:v 3 \
    web/assets/charlotte-card.jpg
ffmpeg -version | head -1
