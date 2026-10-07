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

TEST_CASE("a launch runs its rows, its tiles, its splits, or not at all, in whole workgroups") {
    // Over rows, every token or the last, workgroups of 64.
    CHECK(workgroups_for({Rows::EveryToken, 256, 0, KeySplit::None, 0}, 64, 0, 5, true) == 20);
    CHECK(workgroups_for({Rows::LastToken, 256, 0, KeySplit::None, 0}, 64, 0, 5, true) == 4);
    CHECK(workgroups_for({Rows::EveryToken, 32, 0, KeySplit::None, 0}, 64, 0, 5, true) == 3);   // 160 invocations
    // Tiles of 4 rows, 128 invocations a row: 5 rows take two tiles.
    CHECK(workgroups_for({Rows::EveryToken, 128, 4, KeySplit::None, 0}, 64, 0, 1, true) == 8);
    CHECK(workgroups_for({Rows::EveryToken, 128, 4, KeySplit::None, 0}, 64, 0, 4, true) == 8);
    CHECK(workgroups_for({Rows::EveryToken, 128, 4, KeySplit::None, 0}, 64, 0, 5, true) == 16);
    // Split by chunks: decode at 4,096 takes 16, each split whole workgroups
    // even when its invocations fill less than one.
    CHECK(workgroups_for({Rows::EveryToken, 128, 4, KeySplit::PerChunk, 40960}, 64, 4095, 1, true) == 8 * 16);
    CHECK(workgroups_for({Rows::EveryToken, 1, 0, KeySplit::PerChunk, 40960}, 64, 4095, 1, true) == 16);
    CHECK(workgroups_for({Rows::EveryToken, 128, 4, KeySplit::PerChunk, 40960}, 64, 0, 512, true) == 1024);
    // The combine: only when the step splits.
    CHECK(workgroups_for({Rows::EveryToken, 512, 0, KeySplit::WhenSplit, 40960}, 64, 4095, 1, true) == 8);
    CHECK(workgroups_for({Rows::EveryToken, 512, 0, KeySplit::WhenSplit, 40960}, 64, 100, 1, true) == 0);
    CHECK(workgroups_for({Rows::EveryToken, 512, 0, KeySplit::WhenSplit, 40960}, 64, 0, 512, true) == 0);
}

TEST_CASE("a step of one token decodes and of more prefills, and a launch runs in its token range alone") {
    CHECK(regime_for(1) == Regime::Decode);
    CHECK(regime_for(2) == Regime::Prefill);
    CHECK(regime_for(512) == Regime::Prefill);
    const Geometry decode{Rows::EveryToken, 64, 0, KeySplit::None, 0, tokens_of(Regime::Decode)};
    const Geometry narrow{Rows::EveryToken, 64, 0, KeySplit::None, 0, {2, 8}};
    const Geometry wide{Rows::EveryToken, 64, 0, KeySplit::None, 0, {9, 512}};
    const Geometry every{Rows::LastToken, 64, 0, KeySplit::None, 0};
    CHECK(workgroups_for(decode, 64, 0, 1, true) == 1);
    CHECK(workgroups_for(decode, 64, 0, 2, true) == 0);
    CHECK(workgroups_for(narrow, 64, 0, 1, true) == 0);
    CHECK(workgroups_for(narrow, 64, 0, 2, true) == 2);
    CHECK(workgroups_for(narrow, 64, 0, 8, true) == 8);
    CHECK(workgroups_for(narrow, 64, 0, 9, true) == 0);
    CHECK(workgroups_for(wide, 64, 0, 9, true) == 9);
    CHECK(workgroups_for(every, 64, 0, 1, true) == 1);
    CHECK(workgroups_for(every, 64, 0, 7, true) == 1);
}

TEST_CASE("a launch over the last token alone runs only in a step that asks for logits") {
    const Geometry last{Rows::LastToken, 256, 0, KeySplit::None, 0};
    const Geometry every{Rows::EveryToken, 256, 0, KeySplit::None, 0};
    CHECK(workgroups_for(last, 64, 0, 512, true) == 4);
    CHECK(workgroups_for(last, 64, 0, 512, false) == 0);
    CHECK(workgroups_for(every, 64, 0, 512, false) == 2048);
}
