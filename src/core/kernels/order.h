#pragma once

#include <array>
#include <cstdint>
#include <optional>

namespace bllm::kernels {

// Axis D: changes with the WebGPU surface or the limits a device grants.
//
// The order a program reports its outstanding steps in (kernels/program.h):
// the order they were run, whatever order they settle in. WebGPU does not
// order the mappings of two buffers, so a step may settle before the one
// run ahead of it; it is held until that one has been reported. Pure, so
// the order is tested with steps settling out of order, which a device
// cannot be made to do on demand.
//
//   - At most kOutstanding steps outstanding, numbered from 0 as they are
//     run: step n is slot n % kOutstanding.
//   - Invariant: next_report <= every outstanding step < next_run, and
//     next_run − next_report <= kOutstanding.
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     C.2    Use class if the class has an invariant — the one above.
//     F.8    Prefer pure functions — no GPU object; the program calls it.
//   C++ performance guidelines
//     GPU.7  Pipeline CPU and GPU work — two steps in flight, reported in
//            the order they ran.
class Order {
public:
    static constexpr std::uint64_t kOutstanding = 2;

    [[nodiscard]] bool full() const noexcept { return next_run_ - next_report_ >= kOutstanding; }
    [[nodiscard]] bool idle() const noexcept { return next_run_ == next_report_; }

    // A step run: its number. Precondition: !full().
    std::uint64_t run() noexcept {
        settled_[next_run_ % kOutstanding] = false;
        return next_run_++;
    }

    // Step `step` has settled. Precondition: it is outstanding, not yet
    // settled.
    void settle(std::uint64_t step) noexcept { settled_[step % kOutstanding] = true; }

    // The next step to report, if it has settled, now counted reported;
    // otherwise none, an earlier step still to settle.
    std::optional<std::uint64_t> next() noexcept {
        if (idle() || !settled_[next_report_ % kOutstanding]) return std::nullopt;
        return next_report_++;
    }

private:
    std::uint64_t next_run_ = 0;
    std::uint64_t next_report_ = 0;
    std::array<bool, kOutstanding> settled_{};
};

}  // namespace bllm::kernels
