# Thin task wrapper. Every target is a one-liner you could run by hand.
#
# On the wasm targets, local and CI build the same thing with the same preset
# and the same pinned toolchain image. They differ trivially in who supplies
# the container: CI's job already runs inside the emsdk image and invokes
# cmake directly, while these targets wrap the identical commands in
# `docker run` to put the same image around them locally.
#
# Two divergences that are NOT trivial, recorded so the line above is not read
# more broadly than it is meant:
#
#   - `make test` configures native-debug; CI (test.yml) tests native-release.
#     Different optimisation levels, so the two can genuinely disagree.
#   - `make check` runs six guards; CI runs three, split across test.yml and
#     pages.yml. The rest are local-only.
#
# Both are stated rather than fixed. Closing them is a CI change, not a
# comment.

EMSDK_IMAGE := emscripten/emsdk:6.0.8
# Where the repo is mounted inside the image. Fixed rather than $(CURDIR)
# on purpose: a constant mount point keeps artifacts identical across
# machines, where the host's own path would bake a home directory into the
# build. The cost is that a host-toolchain build configured in the same
# directory is incompatible -- see tools/ensure_container_cache.sh.
CONTAINER_SRC := /src

.PHONY: test test-data dawn test-native test-web bench profile check wasm wasm-diag dist serve serve-dev clean

## Every unit test. No browser; the GPU tests run on this machine's GPU,
## through native Dawn.
test: test-native test-web

## Test data too large to commit, fetched and verified against its pinned
## SHA-256 (tests/fixtures/external.json).
test-data:
	python3 tools/fetch_test_data.py

## Native Dawn, prebuilt and pinned (cmake/dawn.json): the webgpu.h the
## native GPU tests run on.
dawn:
	python3 tools/fetch_dawn.py

## C++ build and unit tests, the GPU tests among them.
test-native: test-data dawn
	cmake --preset native-debug
	cmake --build --preset native-debug
	ctest --preset native-debug

## Tokenizer throughput on a pinned corpus, release build. The corpus is this
## repository's docs and sources at BENCH_COMMIT, so every run measures the
## same bytes; the figures in the tokenizer's optimization commits come from it.
BENCH_COMMIT := 1d74007d019ab5687f2f9d901f4c499b6e62e3b7
bench: test-data dawn
	cmake --preset native-release
	cmake --build --preset native-release --target charlotte_bench_tokenizer
	./tools/make_bench_corpus.sh $(BENCH_COMMIT) .cache/bench/corpus.txt
	@echo "machine: $$(uname -sm), $$(sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
	./build/native-release/bench/charlotte_bench_tokenizer .cache/bench/corpus.txt
	cmake --build --preset native-release --target charlotte_bench_piece_writer
	./build/native-release/bench/charlotte_bench_piece_writer

## JavaScript unit tests, in Node. No dependencies to install; the template
## test reads the pinned model headers the test data holds.
test-web: test-data
	node --test tests/web/*.test.js

## Structural invariants, plus the tests proving each guard actually fires.
check:
	./tools/check_boundaries.sh
	./tools/check_diagnostics_excluded.sh
	./tests/test_check_boundaries.sh
	./tests/test_codex_review_preflight.sh
	./tests/test_ensure_container_cache.sh
	./tests/test_check_site.sh

## The forward pass's GPU time, launch by launch, on Qwen3: DIAGNOSTIC, never a
## throughput figure (bench/forward_profile.cpp). The model's SHA-256 heads the
## report; each launch timed, with its uncertainty, goes to
## build/native-diag/profile.csv.
PROFILE_MODEL := .cache/test-data/models/qwen3-0.6b-q4_0.gguf
profile: test-data dawn
	cmake --preset native-diag
	cmake --build --preset native-diag --target charlotte_profile_forward
	shasum -a 256 $(PROFILE_MODEL)
	./build/native-diag/bench/charlotte_profile_forward $(PROFILE_MODEL) --csv build/native-diag/profile.csv

## Diagnostic wasm build: same optimisation, instrumentation compiled in.
## Timings from this build are diagnostic and are not quotable as throughput.
wasm-diag:
	./tools/ensure_container_cache.sh build/wasm-diag $(CONTAINER_SRC)
	docker run --rm -v "$(CURDIR)":$(CONTAINER_SRC) -w $(CONTAINER_SRC) $(EMSDK_IMAGE) \
		sh -c "emcmake cmake --preset wasm-diag && cmake --build --preset wasm-diag"

## WebAssembly build, inside the pinned toolchain image.
wasm:
	./tools/ensure_container_cache.sh build/wasm-release $(CONTAINER_SRC)
	docker run --rm -v "$(CURDIR)":$(CONTAINER_SRC) -w $(CONTAINER_SRC) $(EMSDK_IMAGE) \
		sh -c "emcmake cmake --preset wasm-release && cmake --build --preset wasm-release"

## Assemble the deployable static site.
dist: wasm
	./tools/assemble_site.sh
	./tools/check_site.sh

## Serve dist/ locally. The page cannot run from file:// — module loading and
## the model fetch both fail against a null origin.
serve: dist
	cd dist && python3 -m http.server 8080

## Serve a development site that includes web/dev/, for ?fake-runtime.
## Never deployed.
serve-dev: wasm
	./tools/assemble_site.sh --dev
	cd dist-dev && python3 -m http.server 8081

clean:
	rm -rf build dist dist-dev
