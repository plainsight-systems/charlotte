// The order a program reports its steps in (kernels/order.h): the order they
// ran, whatever order they settle in — here out of order, as WebGPU may
// settle two buffers' mappings.

#include <doctest/doctest.h>

#include "core/kernels/order.h"

using bllm::kernels::Order;

TEST_CASE("a step that settles first is held until the one run before it is reported") {
    Order order;
    const auto first = order.run(), second = order.run();
    CHECK(order.full());
    order.settle(second);
    CHECK_FALSE(order.next().has_value());   // the first has not settled
    order.settle(first);
    CHECK(order.next() == first);
    CHECK(order.next() == second);
    CHECK_FALSE(order.next().has_value());
    CHECK(order.idle());
}

TEST_CASE("steps in order are reported as they settle, and a slot is reused only once reported") {
    Order order;
    for (std::uint64_t n = 0; n < 6; ++n) {
        CHECK(order.run() == n);
        order.settle(n);
        CHECK(order.next() == n);
        CHECK(order.idle());
    }
    // Two outstanding, one reported: a third may run, into the reported slot.
    const auto a = order.run(), b = order.run();
    order.settle(a);
    CHECK(order.next() == a);
    CHECK_FALSE(order.full());
    const auto c = order.run();
    CHECK(c % Order::kOutstanding == a % Order::kOutstanding);
    order.settle(c);
    CHECK_FALSE(order.next().has_value());   // b still outstanding ahead of c
    order.settle(b);
    CHECK(order.next() == b);
    CHECK(order.next() == c);
}
