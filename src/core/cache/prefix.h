#pragma once

#include <cstddef>
#include <span>

#include "core/tokenizer/tokenizer.h"

namespace bllm::cache {

// Axis G: changes with a new cache or context mechanism.
//
// The diff that keys the KV cache on the token sequence. JavaScript sends the
// whole rendered conversation every turn and holds no cache state; this
// function finds how much of it the cache already holds.
//
// A pure function over two token sequences: no GPU, no model, no file. It is
// the most correctness-critical logic in the chat loop, and it is kept apart
// from the cache's storage so it is tested directly.
//
// The runtime truncates the cache to the returned length and prefills the
// rest. The cache may keep less than the common prefix: a rollback deeper
// than the rollback reserve empties it, and the runtime prefills from the
// first token. When the whole incoming sequence is cached, the runtime still
// recomputes its last token: generation needs that token's logits, and the
// cache holds keys and values, not logits.
//
// Attention is causal, so a token's keys and values depend only on the tokens
// before it: everything before the first difference is exactly reusable, and
// everything from it on is recomputed, matching or not.
//
// What it costs, once per turn: one pass over at most the shorter sequence,
// comparing 4-byte identifiers — at most the context offered, 40,960 for
// Qwen3, 160 KiB read from each side. At one comparison a cycle that is about
// 15 µs, an estimate, not a measurement. What it saves grows with the tokens
// it reuses: each reused token is a row that every layer's projections,
// attention and feed-forward no longer compute, about 1.2 GFLOP for Qwen3,
// and each 512 reused tokens a prefill pass that no longer reads the
// weights. Both sequences already live in the module's memory: the page sends
// text, the module tokenizes it, and no token crosses the JavaScript boundary
// (WASM.2).
//
// Guidelines, by corpus:
//   C++ Core Guidelines
//     ES.1   Prefer the standard library to handcrafted code — std::mismatch.
//     Per.2  Don't optimize prematurely — the diff's 8 bytes a compared
//            token are far below the work one reused token saves, so it is a
//            plain comparison, not a vectorized or hashed one.
//   C++ performance guidelines
//     GDSA.6 Account the bytes a stage moves — the count above.
//     WASM.2 Batch work across the JS boundary — the sequences never cross it.

[[nodiscard]] std::size_t longest_common_prefix(std::span<const tokenizer::TokenId> cached,
                                                std::span<const tokenizer::TokenId> incoming) noexcept;

}  // namespace bllm::cache
