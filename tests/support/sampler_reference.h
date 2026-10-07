#pragma once

#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include "core/policy/policy.h"

// The draw's reference — TEST SUPPORT ONLY (sampler/sampler.h): Philox4x32-10
// and the uniform the draw makes of it, and the draw's steps in f64, given
// the uniform. The harness draws on the GPU; this is what it is checked
// against.
namespace bllm::testing {

struct Philox {
    std::uint32_t words[4];
};

inline std::uint32_t mulhi(std::uint32_t a, std::uint32_t b) {
    return static_cast<std::uint32_t>((std::uint64_t{a} * b) >> 32);
}

// Random123's Philox4x32 with 10 rounds.
inline Philox philox4x32_10(Philox c, std::uint32_t key0, std::uint32_t key1) {
    for (int round = 0; round < 10; ++round) {
        const std::uint32_t hi0 = mulhi(0xD2511F53u, c.words[0]), lo0 = 0xD2511F53u * c.words[0];
        const std::uint32_t hi1 = mulhi(0xCD9E8D57u, c.words[2]), lo1 = 0xCD9E8D57u * c.words[2];
        c = {{hi1 ^ c.words[1] ^ key0, lo1, hi0 ^ c.words[3] ^ key1, lo0}};
        key0 += 0x9E3779B9u;
        key1 += 0xBB67AE85u;
    }
    return c;
}

// The draw's uniform for a seed and a position, as the GPU makes it.
inline float uniform(std::uint64_t seed, std::uint32_t position) {
    const Philox p = philox4x32_10({{position, 0, 0, 0}}, static_cast<std::uint32_t>(seed),
                                   static_cast<std::uint32_t>(seed >> 32));
    return static_cast<float>(p.words[0] >> 9) * (1.0f / 8388608.0f) + (1.0f / 16777216.0f);
}

struct Candidate {
    float logit;
    std::uint32_t token;
};

// The draw in f64, given `u`: the index chosen, each candidate's probability
// of being chosen, and how near the draw came to a decision going the other
// way — the least distance, as a fraction of the sums compared, of top-p's
// running sum from top_p, of a min-p test from its threshold, and of the
// chosen running sum from u × the total. Near 0, f32 on the GPU may decide
// otherwise.
struct Reference {
    std::size_t chosen = 0;
    std::vector<double> probability;
    double margin = 1;
};

inline Reference reference_draw(std::span<const Candidate> c, const policy::SamplingSettings& s, double u) {
    Reference r;
    const std::size_t k = s.top_k;
    const double first = c[0].logit;
    std::vector<double> w(k);
    double total = 0;
    for (std::size_t i = 0; i < k; ++i) total += w[i] = std::exp(c[i].logit - first);
    std::size_t kept = k;
    if (s.top_p < 1) {
        double cumulative = 0;
        for (std::size_t i = 0; i < k; ++i) {
            cumulative += w[i] / total;
            r.margin = std::min(r.margin, std::abs(cumulative - s.top_p));
            if (cumulative >= s.top_p) {
                kept = i + 1;
                break;
            }
        }
    }
    std::size_t survivors = 1;
    const double threshold = s.min_p > 0 ? first + std::log(static_cast<double>(s.min_p)) : -INFINITY;
    while (survivors < kept && c[survivors].logit >= threshold) {
        if (s.min_p > 0) r.margin = std::min(r.margin, std::abs(c[survivors].logit - threshold));
        ++survivors;
    }
    if (survivors < kept && s.min_p > 0) r.margin = std::min(r.margin, std::abs(c[survivors].logit - threshold));
    r.probability.assign(c.size(), 0);
    if (s.temperature == 0) {
        r.probability[0] = 1;
        return r;
    }
    std::vector<double> running(survivors);
    double sum = 0;
    for (std::size_t i = 0; i < survivors; ++i) running[i] = sum += std::exp((c[i].logit - first) / s.temperature);
    for (std::size_t i = 0; i < survivors; ++i) r.probability[i] = (running[i] - (i ? running[i - 1] : 0)) / sum;
    const double target = u * sum;
    r.chosen = survivors - 1;
    for (std::size_t i = 0; i < survivors; ++i) {
        if (running[i] > target) {
            r.chosen = i;
            break;
        }
    }
    r.margin = std::min(r.margin, std::abs(running[r.chosen] - target) / sum);
    if (r.chosen > 0) r.margin = std::min(r.margin, std::abs(running[r.chosen - 1] - target) / sum);
    return r;
}

}  // namespace bllm::testing
