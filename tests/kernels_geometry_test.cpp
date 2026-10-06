// The launch contract's geometry, on the CPU (kernels/interface.h): a layer's
// key chunks for a step, and the invocations a launch runs.

#include <doctest/doctest.h>

#include "core/kernels/interface.h"

using namespace bllm::kernels;

TEST_CASE("a layer's key chunks run from its first row's earliest key to its last row") {
    struct Case {
        std::uint32_t position, tokens, window;
        KeyChunks want;
    };
    for (const Case& c : {
             // Decode, full attention: every chunk through the token's.
             Case{0, 1, 40960, {0, 1, 1}},
             Case{255, 1, 40960, {0, 1, 1}},
             Case{256, 1, 40960, {0, 2, 2}},
             Case{4095, 1, 40960, {0, 16, 16}},
             // A window of 512: at most three chunks, however deep.
             Case{32767, 1, 512, {126, 2, 2}},   // the window starts a chunk
             Case{32768, 1, 512, {126, 3, 3}},
             Case{32768 + 255, 1, 512, {127, 2, 2}},
             Case{511, 1, 512, {0, 2, 2}},
             Case{512, 1, 512, {0, 3, 3}},
             // Prefill: split while every row's partials fit 512 rows.
             Case{0, 512, 40960, {0, 2, 1}},
             Case{0, 256, 40960, {0, 1, 1}},
             Case{0, 128, 40960, {0, 1, 1}},
             Case{3840, 128, 40960, {0, 16, 1}},
             Case{3840, 32, 40960, {0, 16, 16}},
             Case{3840, 33, 40960, {0, 16, 1}},
         }) {
        CAPTURE(c.position);
        CAPTURE(c.tokens);
        CAPTURE(c.window);
        const KeyChunks got = key_chunks(c.position, c.tokens, c.window);
        CHECK(got.first == c.want.first);
        CHECK(got.count == c.want.count);
        CHECK(got.splits == c.want.splits);
    }
}

TEST_CASE("a launch runs its rows, its tiles, its splits, or not at all") {
    // Over rows, every token or the last.
    CHECK(invocations_for({Rows::EveryToken, 256, 0, KeySplit::None, 0}, 0, 5) == 1280);
    CHECK(invocations_for({Rows::LastToken, 256, 0, KeySplit::None, 0}, 0, 5) == 256);
    // Tiles of 4 rows, 128 invocations a row: 5 rows take two tiles.
    CHECK(invocations_for({Rows::EveryToken, 128, 4, KeySplit::None, 0}, 0, 1) == 512);
    CHECK(invocations_for({Rows::EveryToken, 128, 4, KeySplit::None, 0}, 0, 4) == 512);
    CHECK(invocations_for({Rows::EveryToken, 128, 4, KeySplit::None, 0}, 0, 5) == 1024);
    // Split by chunks: decode at 4,096 takes 16.
    CHECK(invocations_for({Rows::EveryToken, 128, 4, KeySplit::PerChunk, 40960}, 4095, 1) == 512 * 16);
    CHECK(invocations_for({Rows::EveryToken, 128, 4, KeySplit::PerChunk, 40960}, 0, 512) == 512 * 128);
    // The combine: only when the step splits.
    CHECK(invocations_for({Rows::EveryToken, 512, 0, KeySplit::WhenSplit, 40960}, 4095, 1) == 512);
    CHECK(invocations_for({Rows::EveryToken, 512, 0, KeySplit::WhenSplit, 40960}, 100, 1) == 0);
    CHECK(invocations_for({Rows::EveryToken, 512, 0, KeySplit::WhenSplit, 40960}, 0, 512) == 0);
}
