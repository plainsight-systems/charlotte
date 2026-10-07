#!/bin/sh
# Assembles the site from web/ and the built module. One script for
# `make dist` and the Pages workflow, so what is served locally is what is
# deployed.
#
#   assemble_site.sh         the deployable site, in dist/, without web/dev/
#   assemble_site.sh --dev   a development site, in dist-dev/, with web/dev/
#   assemble_site.sh --diag  a diagnostic site, in dist-diag/, with web/dev/
#                            and the wasm-diag module, for ?profile
#                            (web/dev/step_profile.js); never deployed
#
# tools/check_site.sh proves dist/ carries no development tooling.
set -eu
cd "$(dirname "$0")/.."

if [ "${1:-}" = "--dev" ]; then
    out=dist-dev
    files="$(cd web && find . -type f)"
else
    out=dist
    files="$(cd web && find . -path ./dev -prune -o -type f -print)"
fi

rm -rf "${out}"
mkdir -p "${out}"
for file in ${files}; do
    mkdir -p "${out}/$(dirname "${file}")"
    cp "web/${file}" "${out}/${file}"
done
cp build/wasm-release/charlotte.mjs build/wasm-release/charlotte.wasm "${out}/"
touch "${out}/.nojekyll"
