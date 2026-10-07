#include "core/kernels/matmul/matmul.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "bllm/shaders_generated.h"
#include "core/capability/capability.h"
#include "core/gguf/types.h"

namespace bllm::kernels {
namespace {

constexpr std::uint32_t kWorkgroupSize = 64;
constexpr std::uint32_t kDecodeRows = 8;        // a decode workgroup's rows
constexpr std::uint32_t kDecodePairs = 4;       // or gate-and-up pairs
// Prefill's tiles, by tokens, and the steps each takes: the narrowest tile
// that holds the step whole, or the widest when none does (matmul.h,
// prefill) (GDSA.6).
struct Tile {
    std::uint32_t tokens;
    TokenRange steps;
};
constexpr std::array<Tile, 3> kTiles{{{8, {2, 8}}, {16, {9, 16}}, {32, {17, UINT32_MAX}}}};
constexpr std::uint32_t kTileOutputs = 64;      // and outputs
constexpr std::uint32_t kTilePairs = 32;        // or gate-and-up pairs

// Binding 1, as matmul.wgsl's Matmul lays it out: for each member its first
// word in the binding, its blocks, its first output row and its rows.
struct Member {
    std::uint32_t base, blocks, first_row, rows;
};
using Constants = std::array<Member, 3>;
static_assert(sizeof(Constants) == 48);

struct Entry {
    std::string_view decode, prefill;
};

Entry entry_of(Epilogue e) {
    switch (e) {
        case Epilogue::Write: return {"decode_write", "prefill_write"};
        case Epilogue::QKV: return {"decode_qkv", "prefill_qkv"};
        case Epilogue::GatedActivation: return {"decode_gated", "prefill_gated"};
    }
    return {"decode_write", "prefill_write"};
}

std::uint32_t ceil_div(std::uint64_t a, std::uint32_t b) { return static_cast<std::uint32_t>((a + b - 1) / b); }

// The product over one binding: its members, the launch pair (or the head's
// one launch) that reads them.
void add_launches(const MatmulLaunch& m, const Binding& weights, const Constants& members, std::uint64_t rows,
                  std::vector<Launch>& out) {
    const residency::WeightView& head = *m.weights[0];
    const std::uint64_t columns = head.shape().dimensions[0];
    const bool gated = m.epilogue == Epilogue::GatedActivation;
    const auto bytes = std::as_bytes(std::span(members));
    std::vector<Binding> bindings{weights, {m.input.buffer, m.input.offset, m.input.length}};
    const std::uint32_t outputs = m.epilogue == Epilogue::QKV ? 3 : 1;
    for (std::uint32_t o = 0; o < outputs; ++o) {
        bindings.push_back({m.outputs[o].buffer, m.outputs[o].offset, m.outputs[o].length});
    }
    std::vector<Override> overrides{
        {"columns", static_cast<double>(columns)},
        {"out_width", static_cast<double>(head.shape().dimensions[1])},
        {"activation", m.activation == model::FeedForwardActivation::GeluTanh ? 1.0 : 0.0}};
    const Entry entry = entry_of(m.epilogue);
    const formats::Format* format = capability::find_format(head.format());

    // Decode: a workgroup a kDecodeRows rows, or kDecodePairs pairs.
    Launch decode{shaders::matmul,
                  format,
                  std::vector<std::byte>(bytes.begin(), bytes.end()),
                  bindings,
                  ceil_div(gated ? rows / 2 : rows, gated ? kDecodePairs : kDecodeRows) * kWorkgroupSize,
                  kWorkgroupSize,
                  m.rows,
                  overrides};
    decode.entry_point = entry.decode;
    if (m.rows == Rows::LastToken) {
        out.push_back(std::move(decode));   // the head: every regime
        return;
    }
    decode.tokens = tokens_of(Regime::Decode);
    out.push_back(std::move(decode));

    // Prefill: a launch a tile width, each workgroup a token tile an output
    // tile.
    const std::uint32_t out_tiles = gated ? ceil_div(rows / 2, kTilePairs) : ceil_div(rows, kTileOutputs);
    for (const auto [tile, steps] : kTiles) {
        std::vector<Override> with_tile = overrides;
        with_tile.push_back({"tile_tokens", static_cast<double>(tile)});
        Launch prefill{shaders::matmul,
                       format,
                       std::vector<std::byte>(bytes.begin(), bytes.end()),
                       bindings,
                       out_tiles * kWorkgroupSize / tile,
                       kWorkgroupSize,
                       Rows::EveryToken,
                       std::move(with_tile)};
        prefill.entry_point = entry.prefill;
        prefill.rows_per_tile = tile;
        prefill.tokens = steps;
        out.push_back(std::move(prefill));
    }
}

}  // namespace

std::vector<Launch> matmul_launches(const MatmulLaunch& m) {
    const residency::WeightView& head = *m.weights[0];
    const std::uint64_t columns = head.shape().dimensions[0];
    const std::uint64_t block = gguf::format_layout(head.format())->block_elements;
    std::vector<Launch> out;

    if (m.members == 1) {
        // One weight: a launch pair a piece, each binding its piece.
        for (const residency::WeightPiece& p : head.pieces()) {
            const Constants members{Member{0, static_cast<std::uint32_t>(p.row_count * columns / block),
                                           static_cast<std::uint32_t>(p.first_row),
                                           static_cast<std::uint32_t>(p.row_count)},
                                    Member{}, Member{}};
            add_launches(m, {p.buffer, p.offset, p.length}, members, p.row_count, out);
        }
        return out;
    }

    // A fused group: one binding spanning its members, each one piece in one
    // buffer, read from its own first word.
    std::uint64_t begin = UINT64_MAX, end = 0;
    for (std::uint32_t i = 0; i < m.members; ++i) {
        const residency::WeightPiece& p = m.weights[i]->pieces().front();
        begin = std::min(begin, p.offset);
        end = std::max(end, p.offset + p.length);
    }
    Constants members{};
    std::uint64_t rows = 0;
    for (std::uint32_t i = 0; i < m.members; ++i) {
        const residency::WeightPiece& p = m.weights[i]->pieces().front();
        members[i] = {static_cast<std::uint32_t>((p.offset - begin) / 4),
                      static_cast<std::uint32_t>(p.row_count * columns / block), static_cast<std::uint32_t>(rows),
                      static_cast<std::uint32_t>(p.row_count)};
        rows += p.row_count;
    }
    add_launches(m, {head.pieces().front().buffer, begin, end - begin}, members, rows, out);
    return out;
}

}  // namespace bllm::kernels
