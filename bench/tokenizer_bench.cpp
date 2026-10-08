// Tokenizer throughput on a pinned corpus: make bench.
//
// For each listed model whose tokenizer is byte-level BPE, it times loading
// and each stage of encoding over the corpus, then whole encodes with and
// without a piece cache. For each whose tokenizer is SentencePiece BPE, it
// times loading, encoding, and decoding the tokens back to text through a
// Utf8Stream. Every figure is printed with the conditions it was
// measured under (WASM.11): the build, the clock, how many runs were timed
// and how many warm-up runs were discarded, and the spread around the median.
// tools/make_bench_corpus.sh pins the corpus and prints its SHA-256; the
// Makefile prints the machine.
//
// It uses only the tokenizer's public interface, and checks that every way
// of encoding gives the same tokens, and that decoding gives the corpus back,
// before it reports any time.
//
//     charlotte_bench_tokenizer <corpus>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/capability/capability.h"
#include "core/gguf/reader.h"
#include "core/tokenizer/bpe/byte_level_bpe.h"
#include "core/tokenizer/bpe/byte_map.h"
#include "core/tokenizer/bpe/merge.h"
#include "core/tokenizer/bpe/merge_table.h"
#include "core/tokenizer/bpe/piece_cache.h"
#include "core/tokenizer/bpe/sentencepiece_bpe.h"
#include "core/tokenizer/nfc.h"
#include "core/tokenizer/pretokenize.h"
#include "core/tokenizer/special.h"
#include "core/tokenizer/unicode.h"
#include "core/tokenizer/vocabulary.h"

using namespace bllm;
using namespace bllm::tokenizer;

namespace {

constexpr int kWarmUp = 3;   // runs discarded before timing
constexpr int kRuns = 21;    // runs timed; the median is the 11th

struct ModelHeader {
    std::string_view model;
    std::string_view data_name;
    std::uint64_t file_size;
};

constexpr ModelHeader kModelHeaders[] = {
#include "fixtures/tokenizer/headers.inc"
};

// The median of the timed runs, with the 10th and 90th percentiles.
struct Timing {
    double median_ms;
    double p10_ms;
    double p90_ms;
};

template <typename Run>
Timing measure(Run run) {
    for (int i = 0; i < kWarmUp; ++i) run();
    std::array<double, kRuns> ms{};
    for (double& m : ms) {
        const auto start = std::chrono::steady_clock::now();
        run();
        m = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    std::sort(ms.begin(), ms.end());
    return {ms[kRuns / 2], ms[kRuns / 10], ms[kRuns * 9 / 10]};
}

std::string shown(const Timing& t) {
    char text[48];
    std::snprintf(text, sizeof text, "%.2f (%.2f-%.2f)", t.median_ms, t.p10_ms, t.p90_ms);
    return text;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// One model's header, read, with what the stages need from it.
struct Model {
    std::string_view id;
    std::string header;
    std::uint64_t file_size;
    gguf::TensorIndex index;
    std::string_view algorithm;                  // "gpt2" or "llama"
    const PreTokenizer* pretokenizer = nullptr;  // for gpt2
};

gguf::MemoryByteSource source_of(const Model& m) {
    return gguf::MemoryByteSource{std::as_bytes(std::span{m.header}), m.file_size};
}

// The model, if its header is fetched and its tokenizer is byte-level BPE
// with a pre-tokenizer the harness implements, or SentencePiece BPE.
bool open_model(const ModelHeader& h, Model& out) {
    out.id = h.model;
    out.header = read_file(std::string(BLLM_TEST_DATA_DIR) + "/" + std::string(h.data_name));
    out.file_size = h.file_size;
    if (out.header.empty()) return false;
    auto source = source_of(out);
    if (gguf::read_index(source, out.index).error != gguf::ReadError::Ok) return false;
    if (out.index.read_string("tokenizer.ggml.model", out.algorithm) != gguf::MetadataError::Ok) return false;
    if (out.algorithm == "llama") return true;
    if (out.algorithm != "gpt2") return false;
    std::string_view pre;
    if (out.index.read_string("tokenizer.ggml.pre", pre) != gguf::MetadataError::Ok) return false;
    out.pretokenizer = capability::find_pretokenizer(pre);
    return out.pretokenizer != nullptr;
}

std::vector<std::uint32_t> ids(const std::vector<TokenId>& tokens) {
    std::vector<std::uint32_t> out;
    out.reserve(tokens.size());
    for (const TokenId t : tokens) out.push_back(static_cast<std::uint32_t>(t));
    return out;
}

// Prints the model's row. False if the ways of encoding disagree.
bool bench_model(const Model& m, std::string_view corpus) {
    auto source = source_of(m);
    const PreTokenizer& pre = *m.pretokenizer;

    const Timing load = measure([&] {
        bpe::ByteLevelBpe fresh;
        (void)bpe::load_byte_level_bpe(source, m.index, pre, fresh);
    });
    bpe::ByteLevelBpe bpe;
    (void)bpe::load_byte_level_bpe(source, m.index, pre, bpe);

    // The stages, run over the whole corpus one at a time.
    const SpecialTokens special{bpe.vocabulary()};
    std::vector<Segment> segments;
    const Timing t_special = measure([&] {
        segments.clear();
        special.segment(corpus, segments);
    });
    std::string normalized(corpus);
    const bool nfc = pre.normalization == Normalization::Nfc;
    const Timing t_nfc = nfc ? measure([&] { (void)to_nfc(corpus, normalized); }) : Timing{};
    std::vector<Piece> pieces;
    const Timing t_split = measure([&] { (void)split(pre, normalized, pieces); });

    bpe::MergeTable merges;
    (void)bpe::load_merges(source, m.index, bpe.vocabulary(), merges);
    std::array<TokenId, 256> byte_tokens{};
    for (std::size_t b = 0; b < 256; ++b) {
        std::string text;
        append_utf8(bpe::kByteChars[b], text);
        byte_tokens[b] = *bpe.vocabulary().find(text);
    }
    std::vector<std::vector<TokenId>> piece_symbols;
    for (const Piece& p : pieces) {
        std::vector<TokenId> symbols;
        for (const char c : std::string_view(normalized).substr(p.offset, p.length)) {
            symbols.push_back(byte_tokens[static_cast<unsigned char>(c)]);
        }
        piece_symbols.push_back(std::move(symbols));
    }
    std::vector<TokenId> merged;
    const Timing t_merge = measure([&] {
        merged.clear();
        for (const auto& symbols : piece_symbols) bpe::merge(symbols, merges, merged);
    });

    // Whole encodes: without a cache, with a fresh one each run, and with one
    // kept from run to run, as a chat keeps it from turn to turn.
    std::vector<TokenId> plain, cold, warm;
    const Timing t_plain = measure([&] {
        plain.clear();
        (void)bpe.encode(corpus, kNoTokenLimit, plain);
    });
    const Timing t_cold = measure([&] {
        bpe::PieceCache cache;
        cold.clear();
        (void)bpe.encode(corpus, kNoTokenLimit, cold, cache);
    });
    bpe::PieceCache kept;
    const Timing t_warm = measure([&] {
        warm.clear();
        (void)bpe.encode(corpus, kNoTokenLimit, warm, kept);
    });

    const bool agree = ids(plain) == ids(cold) && ids(plain) == ids(warm);
    std::printf("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %zu | %s |\n", std::string(m.id).c_str(),
                shown(load).c_str(), shown(t_special).c_str(), nfc ? shown(t_nfc).c_str() : "-",
                shown(t_split).c_str(), shown(t_merge).c_str(), shown(t_plain).c_str(), shown(t_cold).c_str(),
                shown(t_warm).c_str(), plain.size(), agree ? "yes" : "NO");
    return agree;
}

// Prints the model's row. False if decoding does not give the corpus back.
bool bench_sentencepiece(const Model& m, std::string_view corpus) {
    auto source = source_of(m);
    const Timing load = measure([&] {
        bpe::SentencePieceBpe fresh;
        (void)bpe::load_sentencepiece_bpe(source, m.index, fresh);
    });
    bpe::SentencePieceBpe spm;
    (void)bpe::load_sentencepiece_bpe(source, m.index, spm);

    std::vector<TokenId> tokens;
    const Timing encode = measure([&] {
        tokens.clear();
        (void)spm.encode(corpus, kNoTokenLimit, tokens);
    });
    // A token at a time, as generation streams it.
    std::string text;
    std::string bytes;
    const Timing decode = measure([&] {
        text.clear();
        Utf8Stream stream;
        for (const TokenId t : tokens) {
            bytes.clear();
            spm.decode(t, bytes);
            stream.push(bytes, text);
        }
        (void)stream.finish(text);
    });

    const bool round_trip = text == corpus;
    std::printf("| %s | %s | %s | %s | %zu | %s |\n", std::string(m.id).c_str(), shown(load).c_str(),
                shown(encode).c_str(), shown(decode).c_str(), tokens.size(), round_trip ? "yes" : "NO");
    return round_trip;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <corpus>\n", argv[0]);
        return 2;
    }
    const std::string corpus = read_file(argv[1]);
    if (corpus.empty()) {
        std::fprintf(stderr, "cannot read the corpus %s\n", argv[1]);
        return 1;
    }

#if defined(NDEBUG)
    constexpr const char* kBuild = "release (NDEBUG)";
#else
    constexpr const char* kBuild = "debug: figures not quotable";
#endif
    std::printf("Tokenizer benchmark: %zu bytes of corpus, %s build, compiler %s\n", corpus.size(), kBuild,
                __VERSION__);
    std::printf("Each figure: median in ms of %d timed runs after %d discarded, with the 10th-90th percentile; "
                "std::chrono::steady_clock.\n\n",
                kRuns, kWarmUp);
    std::vector<Model> models;
    for (const ModelHeader& h : kModelHeaders) {
        Model m;
        if (open_model(h, m)) models.push_back(std::move(m));
    }
    if (models.empty()) {
        std::fprintf(stderr, "no listed model's header found under %s; run make test-data\n", BLLM_TEST_DATA_DIR);
        return 1;
    }

    bool all_agree = true;
    std::printf("Byte-level BPE:\n\n");
    std::printf("| model | load | special tokens | NFC | split | merge | encode | cached, cold | cached, warm "
                "| tokens | all agree |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|---|---|\n");
    for (const Model& m : models) {
        if (m.algorithm == "gpt2") all_agree = bench_model(m, corpus) && all_agree;
    }
    std::printf("\nSentencePiece BPE (decode: a token at a time through a Utf8Stream):\n\n");
    std::printf("| model | load | encode | decode | tokens | decodes to the corpus |\n");
    std::printf("|---|---|---|---|---|---|\n");
    for (const Model& m : models) {
        if (m.algorithm == "llama") all_agree = bench_sentencepiece(m, corpus) && all_agree;
    }
    return all_agree ? 0 : 1;
}
