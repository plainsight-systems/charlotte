#include <doctest/doctest.h>

#include <cstdint>
#include <initializer_list>
#include <vector>

#include "core/cache/kv.h"

using namespace bllm;
using cache::KvCache;

namespace {

struct LayerShape {
    std::uint32_t window;
    std::uint32_t slots;
};

// A model and plan of these layers; only what the cache reads is filled.
struct Fixture {
    model::ModelDescription model{};
    residency::ResidencyPlan plan{};
};

Fixture make(std::initializer_list<LayerShape> layers) {
    Fixture f;
    std::uint32_t buffer = 0;
    for (const LayerShape l : layers) {
        model::LayerDescription d{};
        d.attention_window = l.window;
        f.model.layers.push_back(d);
        f.plan.cache.push_back({{residency::BufferIndex{buffer}, 0, 4}, {residency::BufferIndex{buffer + 1}, 0, 4},
                                l.slots});
        buffer += 2;
    }
    return f;
}

// What a ring holds after `length` tokens, simulated slot by slot: the
// position each slot was last written with.
bool ring_still_holds(std::uint32_t slots, std::uint32_t length, std::uint32_t position) {
    std::vector<std::int64_t> slot(slots, -1);
    for (std::uint32_t p = 0; p < length; ++p) slot[p % slots] = p;
    return slot[position % slots] == position;
}

}  // namespace

TEST_CASE("a new cache is empty, and reports its capacity, precision and each layer's storage") {
    const Fixture f = make({{100, 100}, {8, 20}});
    const KvCache cache(f.model, f.plan, policy::CachePrecision::F16, 100);
    CHECK(cache.length() == 0);
    CHECK(cache.capacity() == 100);
    CHECK(cache.precision() == policy::CachePrecision::F16);
    CHECK(cache.layer(model::LayerIndex{1}).slots == 20);
    CHECK(cache.layer(model::LayerIndex{1}).keys.buffer == residency::BufferIndex{2});
    CHECK(cache.layer(model::LayerIndex{0}).values.buffer == residency::BufferIndex{1});
}

TEST_CASE("advancing adds each step's tokens; truncating within full-attention layers always keeps") {
    const Fixture f = make({{1000, 1000}, {1000, 1000}});
    KvCache cache(f.model, f.plan, policy::CachePrecision::F16, 1000);
    cache.advance(512);
    cache.advance(1);
    CHECK(cache.length() == 513);
    CHECK(cache.truncate(513) == 513);
    CHECK(cache.truncate(37) == 37);
    CHECK(cache.length() == 37);
    CHECK(cache.truncate(0) == 0);
    CHECK(cache.length() == 0);
}

TEST_CASE("a sliding layer keeps a rollback while the next query's window is still in its ring") {
    // Window 4, ring of 10: after 25 tokens it holds positions 15 .. 24. The
    // next query at t reads t - 3 .. t - 1, so t = 18 is the deepest kept.
    for (const std::uint32_t t : {25u, 24u, 18u}) {
        const Fixture f = make({{4, 10}});
        KvCache cache(f.model, f.plan, policy::CachePrecision::F16, 100);
        cache.advance(25);
        CAPTURE(t);
        CHECK(cache.truncate(t) == t);
        CHECK(cache.length() == t);
    }
    const Fixture f = make({{4, 10}});
    KvCache cache(f.model, f.plan, policy::CachePrecision::F16, 100);
    cache.advance(25);
    CHECK(cache.truncate(17) == 0);   // would need position 14, overwritten by 24
    CHECK(cache.length() == 0);
}

TEST_CASE("one sliding layer that has lost the window empties the whole cache") {
    const Fixture f = make({{100, 100}, {4, 10}, {100, 100}});
    KvCache cache(f.model, f.plan, policy::CachePrecision::F16, 100);
    cache.advance(25);
    CHECK(cache.truncate(17) == 0);
    CHECK(cache.length() == 0);
}

TEST_CASE("truncation agrees with a slot-by-slot simulation of every ring, length and rollback") {
    // The rule against the ring itself: keep exactly when every earlier
    // position the next query reads is still in its slot.
    for (const std::uint32_t window : {1u, 2u, 4u, 7u}) {
        for (const std::uint32_t slots : {window, window + 1, window + 5, 3 * window + 2}) {
            for (std::uint32_t length = 0; length <= 3 * slots + 3; ++length) {
                for (std::uint32_t t = 0; t <= length; ++t) {
                    bool held = true;
                    const std::uint32_t first = t >= window ? t - window + 1 : 0;
                    for (std::uint32_t p = first; p < t; ++p) held = held && ring_still_holds(slots, length, p);
                    const Fixture f = make({{window, slots}});
                    KvCache cache(f.model, f.plan, policy::CachePrecision::F16, 1000);
                    cache.advance(length);
                    CAPTURE(window);
                    CAPTURE(slots);
                    CAPTURE(length);
                    CAPTURE(t);
                    REQUIRE(cache.truncate(t) == (held ? t : 0));
                }
            }
        }
    }
}

TEST_CASE("a second rollback is judged by what the ring was written to, not by the length the first left") {
    // Window 4, ring of 10. After 25 tokens the ring holds 15 .. 24, and
    // rolling back to 18 keeps it. Rolling back again to 12 needs 9 .. 11,
    // overwritten when 19 .. 21 were written: the length is 18, but the ring
    // was written to 25.
    const Fixture f = make({{4, 10}});
    KvCache cache(f.model, f.plan, policy::CachePrecision::F16, 100);
    cache.advance(25);
    REQUIRE(cache.truncate(18) == 18);
    CHECK(cache.truncate(12) == 0);
    CHECK(cache.length() == 0);
}

TEST_CASE("any history of steps and rollbacks agrees with a slot-by-slot simulation of the ring") {
    // Random histories against the ring itself: a slot holds the last
    // position written to it, and a rollback keeps the cache exactly when
    // every earlier position the next query reads is still in its slot.
    std::uint32_t state = 0x6C8E9CF5u;
    const auto next = [&](std::uint32_t below) {
        state = state * 1664525u + 1013904223u;
        return (state >> 8) % below;
    };
    for (const LayerShape shape : {LayerShape{2, 2}, LayerShape{4, 10}, LayerShape{7, 9}, LayerShape{3, 20}}) {
        for (int history = 0; history < 200; ++history) {
            const Fixture f = make({shape});
            KvCache cache(f.model, f.plan, policy::CachePrecision::F16, 100000);
            std::vector<std::int64_t> slot(shape.slots, -1);
            std::uint32_t length = 0;
            for (int op = 0; op < 40; ++op) {
                if (next(2) == 0) {
                    const std::uint32_t tokens = 1 + next(2 * shape.slots);
                    for (std::uint32_t p = length; p < length + tokens; ++p) slot[p % shape.slots] = p;
                    length += tokens;
                    cache.advance(tokens);
                } else {
                    const std::uint32_t t = next(length + 1);
                    const std::uint32_t first = t >= shape.window ? t - shape.window + 1 : 0;
                    bool held = true;
                    for (std::uint32_t p = first; p < t; ++p) held = held && slot[p % shape.slots] == p;
                    length = held ? t : 0;
                    CAPTURE(shape.window);
                    CAPTURE(shape.slots);
                    CAPTURE(history);
                    CAPTURE(op);
                    REQUIRE(cache.truncate(t) == length);
                }
                REQUIRE(cache.length() == length);
            }
        }
    }
}
