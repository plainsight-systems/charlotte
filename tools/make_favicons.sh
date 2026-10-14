#!/bin/sh
# Renders the page's icons from web/assets/favicon.svg, which browsers that
# take SVG icons use as it is: favicon-32.png for those that don't, and
# apple-touch-icon.png, 180 × 180, for a phone's home screen. The home-screen
# icon is drawn square, without the SVG's rounded corners, since iOS rounds
# it itself. web/index.html links all three.
set -eu
cd "$(dirname "$0")/.."

rsvg-convert -w 32 -h 32 web/assets/favicon.svg -o web/assets/favicon-32.png
sed 's/ rx="6"//' web/assets/favicon.svg \
    | rsvg-convert -w 180 -h 180 -o web/assets/apple-touch-icon.png
rsvg-convert --version | head -1
